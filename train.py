"""CLI entrypoint for the tiny synthetic overfit experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from training.trainer import train_language_model, train_synthetic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--task",
        choices=["synthetic", "real_lm"],
        default=None,
        help="Override task type. Defaults to config['task'] or synthetic.",
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    task = args.task or config.get("task", "synthetic")
    if task == "real_lm":
        summary = train_language_model(config)
    else:
        summary = train_synthetic(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
