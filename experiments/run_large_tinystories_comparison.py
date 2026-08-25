"""Run the larger TinyStories comparison near the initial research scale.

The small 7M-parameter configs are useful smoke tests. These configs move the
same comparison to roughly 35M-38M parameters while keeping the per-token and
recurrent memory-compressed variants matched on persistent memory floats and
total parameter count.
"""

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.trainer import train_language_model


CONFIGS = [
    "large_tinystories_baseline.yaml",
    "large_tinystories_per_token.yaml",
    "large_tinystories_recurrent.yaml",
]


if __name__ == "__main__":
    for config_name in CONFIGS:
        print(f"\n=== {config_name} ===", flush=True)
        with Path("configs", config_name).open("r", encoding="utf-8") as f:
            train_language_model(yaml.safe_load(f))
