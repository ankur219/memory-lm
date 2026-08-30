"""Print compact markdown tables for optional ~100M scale-point logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DATASETS = {
    "tinystories": {
        "title": "TinyStories",
        "logs": {
            "Baseline": "scale100_tinystories_baseline",
            "Per-token (Many-Small)": "scale100_tinystories_per_token",
            "Recurrent (Few-Rich, 208x2048)": "scale100_tinystories_recurrent",
        },
    },
    "wikitext": {
        "title": "WikiText-103",
        "logs": {
            "Baseline": "scale100_wikitext_baseline",
            "Per-token (Many-Small)": "scale100_wikitext_per_token",
            "Recurrent (Few-Rich, 208x2048)": "scale100_wikitext_recurrent",
        },
    },
}


def parse_summary(log_path: Path):
    if not log_path.exists():
        return None

    summary = None
    last_metrics = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("event") == "metrics":
                last_metrics = data
            elif data.get("event") == "summary":
                summary = data

    if not summary and last_metrics:
        summary = {"final_metrics": last_metrics}
    return summary


def candidate_log_paths(logs_dir: Path, stem: str, seed: int | None):
    if seed is not None:
        yield logs_dir / f"{stem}_seed{seed}.jsonl"
    yield logs_dir / f"{stem}.jsonl"


def row_for(name: str, stem: str, logs_dir: Path, seed: int | None):
    summary = None
    for path in candidate_log_paths(logs_dir, stem, seed):
        summary = parse_summary(path)
        if summary is not None:
            break
    if summary is None:
        return {
            "Model": name,
            "Val Loss": "missing",
            "Perplexity": "missing",
            "Tokens/sec": "missing",
            "Peak VRAM": "missing",
            "Params": "missing",
            "Mem Floats": "missing",
            "Full Val": "missing",
        }

    final_metrics = summary.get("final_metrics", {})
    val_loss = final_metrics.get("validation_loss", float("nan"))
    ppl = math.exp(val_loss) if not math.isnan(val_loss) else float("nan")
    tokens_per_sec = final_metrics.get("tokens_per_sec", 0.0)
    peak_vram = final_metrics.get("peak_gpu_memory_mb", 0.0)
    param_cnt = summary.get("parameter_count", {}).get("total", 0)
    mem_floats = summary.get("memory_budget", {}).get("floats", 0)
    val_full = final_metrics.get("validation_full")

    return {
        "Model": name,
        "Val Loss": f"{val_loss:.4f}",
        "Perplexity": f"{ppl:.2f}",
        "Tokens/sec": f"{tokens_per_sec / 1000:.1f}k" if tokens_per_sec > 0 else "N/A",
        "Peak VRAM": f"{peak_vram:.2f} MB" if peak_vram > 0 else "N/A",
        "Params": f"{param_cnt:,}",
        "Mem Floats": f"{mem_floats:,}",
        "Full Val": str(val_full),
    }


def print_table(title: str, rows: list[dict[str, str]]) -> None:
    print(f"\n## {title}")
    headers = ["Model", "Val Loss", "Perplexity", "Tokens/sec", "Peak VRAM", "Params", "Mem Floats", "Full Val"]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))

    print(" | ".join(f"{h:<{widths[h]}}" for h in headers))
    print("-|-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" | ".join(f"{row[h]:<{widths[h]}}" for h in headers))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--seed", type=int, default=None, help="Prefer *_seedN.jsonl logs.")
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    for dataset_key in args.datasets:
        spec = DATASETS[dataset_key]
        rows = [row_for(name, stem, logs_dir, args.seed) for name, stem in spec["logs"].items()]
        print_table(spec["title"], rows)


if __name__ == "__main__":
    main()
