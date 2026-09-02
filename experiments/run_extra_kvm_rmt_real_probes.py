"""Run optional 7M/100M KVM and RMT real-data probes.

The 35M KVM/RMT real-data configs have dedicated runners. This wrapper covers
the extra scale points so they can be queued without hand-writing a long shell
script. All configs disable checkpoint saving by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.trainer import train_language_model


CONFIG_GROUPS = {
    ("7m", "kvm"): ["tinystories_kvm.yaml", "wikitext_kvm.yaml"],
    ("7m", "rmt"): ["tinystories_rmt.yaml", "wikitext_rmt.yaml"],
    ("100m", "kvm"): ["scale100_tinystories_kvm.yaml", "scale100_wikitext_kvm.yaml"],
    ("100m", "rmt"): ["scale100_tinystories_rmt.yaml", "scale100_wikitext_rmt.yaml"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["7m", "100m", "all"], default="7m")
    parser.add_argument("--model", choices=["kvm", "rmt", "all"], default="all")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["tinystories", "wikitext"],
        default=["tinystories", "wikitext"],
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed in all configs.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size.")
    parser.add_argument(
        "--full-epoch",
        action="store_true",
        help="Clear max_steps/final_eval_max_batches. Most useful for RMT probe configs.",
    )
    return parser.parse_args()


def selected_configs(args: argparse.Namespace) -> list[str]:
    scales = ["7m", "100m"] if args.scale == "all" else [args.scale]
    models = ["kvm", "rmt"] if args.model == "all" else [args.model]
    configs: list[str] = []
    for scale in scales:
        for model in models:
            for config_name in CONFIG_GROUPS[(scale, model)]:
                if any(dataset in config_name for dataset in args.datasets):
                    configs.append(config_name)
    return configs


def apply_overrides(config: dict, config_name: str, args: argparse.Namespace) -> dict:
    config = dict(config)
    model = dict(config["model"])
    config["model"] = model

    suffix_parts = []
    if args.seed is not None:
        config["seed"] = args.seed
        suffix_parts.append(f"seed{args.seed}")
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
        suffix_parts.append(f"steps{args.max_steps}")
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
        suffix_parts.append(f"bs{args.batch_size}")
    if args.full_epoch:
        config["max_steps"] = None
        config["final_eval_max_batches"] = None
        suffix_parts.append("full")

    if config["model_name"] == "rmt":
        model["recurrent_memory_dim"] = model["hidden_size"]
    elif config["model_name"] == "kvm":
        model["recurrent_memory_dim"] = model.get("recurrent_memory_dim") or model["memory_dim"]

    config["checkpoint_dir"] = None
    config["save_every"] = 0

    if suffix_parts:
        stem = Path(config_name).stem
        suffix = "_".join(suffix_parts)
        config["log_path"] = f"logs/{stem}_{suffix}.jsonl"
        config["csv_path"] = f"logs/{stem}_{suffix}.csv"

    return config


def main() -> None:
    args = parse_args()
    configs = selected_configs(args)
    if not configs:
        raise SystemExit("No configs selected.")

    for config_name in configs:
        print(f"\n=== {config_name} ===", flush=True)
        with Path("configs", config_name).open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        train_language_model(apply_overrides(config, config_name, args))


if __name__ == "__main__":
    main()
