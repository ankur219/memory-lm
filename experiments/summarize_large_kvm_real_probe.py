"""Print compact markdown tables for 35M-scale KVM real-LM probe logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LOGS = {
    "TinyStories KVM-style": "large_tinystories_kvm",
    "WikiText-103 KVM-style": "large_wikitext_kvm",
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
    if summary is None and last_metrics is not None:
        summary = {"final_metrics": last_metrics}
    return summary


def candidate_paths(logs_dir: Path, stem: str, suffix: str | None):
    if suffix:
        yield logs_dir / f"{stem}_{suffix}.jsonl"
    yield logs_dir / f"{stem}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--suffix", default=None)
    args = parser.parse_args()

    rows = []
    for name, stem in LOGS.items():
        summary = None
        for path in candidate_paths(Path(args.logs_dir), stem, args.suffix):
            summary = parse_summary(path)
            if summary is not None:
                break
        if summary is None:
            rows.append([name, "missing", "missing", "missing", "missing", "missing", "missing", "missing"])
            continue

        final_metrics = summary.get("final_metrics", {})
        val_loss = final_metrics.get("validation_loss", float("nan"))
        ppl = math.exp(val_loss) if not math.isnan(val_loss) else float("nan")
        rows.append(
            [
                name,
                f"{val_loss:.4f}",
                f"{ppl:.2f}",
                f"{final_metrics.get('tokens_per_sec', 0.0) / 1000:.1f}k",
                f"{final_metrics.get('peak_gpu_memory_mb', 0.0):.0f} MB",
                f"{summary.get('parameter_count', {}).get('total', 0):,}",
                f"{summary.get('memory_budget', {}).get('floats', 0):,}",
                str(final_metrics.get("validation_full")),
            ]
        )

    headers = ["Model", "Val Loss", "Perplexity", "Tokens/sec", "Peak VRAM", "Params", "Mem Floats", "Full Val"]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-|-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, widths)))


if __name__ == "__main__":
    main()
