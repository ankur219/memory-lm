"""Report real-data token counts and run budgets for comparison configs."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.text import tokenizer_metadata
from evaluation.efficiency import parameter_breakdown
from evaluation.efficiency import per_token_memory_budget
from training.trainer import build_model, build_real_lm_datasets_from_config, memory_budget_for_model
from models import TransformerConfig


CONFIGS = ["real_baseline.yaml", "real_per_token.yaml", "real_recurrent.yaml"]


def load_description(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    context_length = int(cfg["model"]["context_length"])
    train_ds, val_ds, tokenizer = build_real_lm_datasets_from_config(cfg)

    model_cfg_dict = dict(cfg["model"])
    model_cfg_dict["vocab_size"] = tokenizer.vocab_size
    model_cfg = TransformerConfig(**model_cfg_dict)
    model = build_model(cfg["model_name"], model_cfg)
    batch_size = int(cfg.get("batch_size", 1))
    steps_per_epoch = (len(train_ds) + batch_size - 1) // batch_size
    num_epochs = int(cfg.get("num_epochs", 1))
    max_steps = cfg.get("max_steps")
    planned_steps = int(max_steps) if max_steps is not None else steps_per_epoch * num_epochs
    if max_steps is not None:
        planned_tokens = planned_steps * batch_size * context_length
    else:
        planned_tokens = len(train_ds) * context_length * num_epochs
    planned_data_passes = planned_steps / max(1, steps_per_epoch)

    return {
        "config_name": config_path.name,
        "model_name": cfg["model_name"],
        "tokenizer": tokenizer_metadata(tokenizer),
        "train_tokens": len(train_ds) * context_length,
        "validation_tokens": len(val_ds) * context_length,
        "train_blocks": len(train_ds),
        "validation_blocks": len(val_ds),
        "context_length": context_length,
        "block_stride": train_ds.stride,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "planned_steps": planned_steps,
        "planned_train_tokens": planned_tokens,
        "planned_data_passes": planned_data_passes,
        "params": parameter_breakdown(model)["total"],
        "param_breakdown": parameter_breakdown(model),
        "persistent_memory": memory_budget_for_model(cfg["model_name"], model_cfg, context_length),
    }


def print_description(desc: dict) -> None:
    print(f"\n{desc['config_name']}")
    print(f"  model: {desc['model_name']}")
    print(f"  tokenizer: {desc['tokenizer']}")
    print(f"  train tokens per epoch: {desc['train_tokens']:,}")
    print(f"  validation tokens per eval: {desc['validation_tokens']:,}")
    print(f"  train blocks: {desc['train_blocks']:,}")
    print(f"  validation blocks: {desc['validation_blocks']:,}")
    print(f"  context_length: {desc['context_length']}")
    print(f"  block_stride: {desc['block_stride']}")
    print(f"  batch_size: {desc['batch_size']}")
    print(f"  steps_per_epoch: {desc['steps_per_epoch']:,}")
    print(f"  planned_steps: {desc['planned_steps']:,}")
    print(f"  planned_train_tokens: {desc['planned_train_tokens']:,}")
    print(f"  planned_data_passes: {desc['planned_data_passes']:.2f}")
    print(f"  params: {desc['params']:,}")
    print(f"  param_breakdown: {desc['param_breakdown']}")
    print(f"  persistent_memory: {desc['persistent_memory']}")


if __name__ == "__main__":
    descriptions = [load_description(Path("configs") / name) for name in CONFIGS]
    for desc in descriptions:
        print_description(desc)

    per_token = next(d for d in descriptions if d["model_name"] == "per_token")
    recurrent = next(d for d in descriptions if d["model_name"] == "recurrent")
    print("\nmatched_many_small_vs_few_rich")
    print(f"  params_equal: {per_token['params'] == recurrent['params']}")
    print(f"  per_token_params: {per_token['params']:,}")
    print(f"  recurrent_params: {recurrent['params']:,}")
    print(
        "  memory_equal: "
        f"{per_token['persistent_memory']['floats'] == recurrent['persistent_memory']['floats']}"
    )
    print(f"  per_token_memory_floats: {per_token['persistent_memory']['floats']:,}")
    print(f"  recurrent_memory_floats: {recurrent['persistent_memory']['floats']:,}")
