"""Run baseline/per-token/recurrent on the same real-text configuration."""

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.trainer import train_language_model


if __name__ == "__main__":
    for config_name in ["real_baseline.yaml", "real_per_token.yaml", "real_recurrent.yaml"]:
        print(f"\n=== {config_name} ===")
        with Path("configs", config_name).open("r", encoding="utf-8") as f:
            train_language_model(yaml.safe_load(f))
