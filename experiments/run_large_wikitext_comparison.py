"""Run the larger WikiText-103 comparison near the initial research scale.

This mirrors the larger TinyStories comparison, but swaps the corpus to
WikiText-103. The goal is to check whether the 35M-scale TinyStories ordering
replicates on a less story-like natural-language dataset.
"""

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.trainer import train_language_model


CONFIGS = [
    "large_wikitext_baseline.yaml",
    "large_wikitext_per_token.yaml",
    "large_wikitext_recurrent.yaml",
]


if __name__ == "__main__":
    for config_name in CONFIGS:
        print(f"\n=== {config_name} ===", flush=True)
        with Path("configs", config_name).open("r", encoding="utf-8") as f:
            train_language_model(yaml.safe_load(f))
