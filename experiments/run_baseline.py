from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.trainer import train_synthetic


if __name__ == "__main__":
    with Path("configs/baseline.yaml").open("r", encoding="utf-8") as f:
        train_synthetic(yaml.safe_load(f))
