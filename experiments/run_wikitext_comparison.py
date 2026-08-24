"""Run the matched model comparison on WikiText-103.

This is the next real-data check after TinyStories. It uses the same model
sizes and memory budgets, but swaps in a less story-like corpus so we can see
whether the TinyStories result is dataset-specific.
"""

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.trainer import train_language_model


CONFIGS = [
    "wikitext_baseline.yaml",
    "wikitext_per_token.yaml",
    "wikitext_recurrent.yaml",
]


if __name__ == "__main__":
    for config_name in CONFIGS:
        print(f"\n=== {config_name} ===", flush=True)
        with Path("configs", config_name).open("r", encoding="utf-8") as f:
            train_language_model(yaml.safe_load(f))
