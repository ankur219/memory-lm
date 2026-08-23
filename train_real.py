"""Convenience entrypoint for real-text language-model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from training.trainer import train_language_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_per_token.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    summary = train_language_model(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

