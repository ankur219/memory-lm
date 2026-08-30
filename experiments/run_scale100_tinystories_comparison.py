"""Run the optional ~100M TinyStories scale-point comparison.

This mirrors the 35M TinyStories comparison but uses a larger backbone. The
compressed variants are matched on persistent memory floats:

per-token: 128 tokens x 13 layers x 2(K,V) x 128 dim = 425,984 floats
recurrent: 208 memory slots x 2048 dim = 425,984 floats

Checkpoint saving is disabled in the configs to avoid filling small instance
disks during these long runs.
"""

from pathlib import Path
import argparse
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.trainer import train_language_model


CONFIGS = [
    "scale100_tinystories_baseline.yaml",
    "scale100_tinystories_per_token.yaml",
    "scale100_tinystories_recurrent.yaml",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Override seed in all configs.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=CONFIGS,
        help="Config filenames under configs/ to run. Defaults to the full comparison.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for config_name in args.configs:
        print(f"\n=== {config_name} ===", flush=True)
        with Path("configs", config_name).open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if args.seed is not None:
            config["seed"] = args.seed
            stem = Path(config_name).stem
            config["log_path"] = f"logs/{stem}_seed{args.seed}.jsonl"
            config["csv_path"] = f"logs/{stem}_seed{args.seed}.csv"
        train_language_model(config)
