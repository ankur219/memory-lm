"""Run 35M-scale KVM-style real-language-model probes.

KVM keeps layerwise compressed key/value state. These configs use the same
262,144 persistent-memory-float budget as the 35M per-token and recurrent
compressed runs:

    8 layers x 128 slots x 2(K,V) x 128 dim = 262,144 floats

Checkpoint saving is disabled by default to avoid filling small experiment
instances.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.trainer import train_language_model


CONFIGS = [
    "large_tinystories_kvm.yaml",
    "large_wikitext_kvm.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=CONFIGS,
        help="Config filenames under configs/ to run.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed in all configs.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size.")
    parser.add_argument(
        "--num-memory-tokens",
        type=int,
        default=None,
        help="Override KVM state slots. Memory floats = layers * slots * 2 * state_dim.",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="Override KVM recurrent_memory_dim and memory_dim.",
    )
    return parser.parse_args()


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
    if args.num_memory_tokens is not None:
        model["num_memory_tokens"] = args.num_memory_tokens
        suffix_parts.append(f"slots{args.num_memory_tokens}")
    if args.state_dim is not None:
        model["memory_dim"] = args.state_dim
        model["recurrent_memory_dim"] = args.state_dim
        suffix_parts.append(f"dim{args.state_dim}")

    model["recurrent_memory_dim"] = model.get("recurrent_memory_dim") or model["memory_dim"]
    config["checkpoint_dir"] = None
    config["save_every"] = 0

    if suffix_parts:
        stem = Path(config_name).stem
        suffix = "_".join(suffix_parts)
        config["log_path"] = f"logs/{stem}_{suffix}.jsonl"
        config["csv_path"] = f"logs/{stem}_{suffix}.csv"

    return config


if __name__ == "__main__":
    args = parse_args()
    for config_name in args.configs:
        print(f"\n=== {config_name} ===", flush=True)
        with Path("configs", config_name).open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        train_language_model(apply_overrides(config, config_name, args))
