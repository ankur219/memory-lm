from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.trainer import train_language_model

if __name__ == "__main__":
    configs = [
        "full_baseline.yaml",
        "full_per_token.yaml",
        "full_recurrent_mean_gru.yaml",
        "full_recurrent_cross_attn.yaml",
    ]
    for config_name in configs:
        print(f"\n=== {config_name} ===", flush=True)
        with Path("configs", config_name).open("r", encoding="utf-8") as f:
            train_language_model(yaml.safe_load(f))
