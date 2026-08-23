"""Small, readable training loop for milestone experiments."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, random_split

from data.synthetic import KeyValueRetrievalDataset, collate_batch
from data.text import build_lm_datasets, load_text_file, load_tinystories_text, tokenizer_metadata
from evaluation.efficiency import (
    baseline_kv_memory_budget,
    parameter_breakdown,
    per_token_memory_budget,
    recurrent_memory_budget,
)
from evaluation.language_modeling import evaluate_loss
from evaluation.memory_tasks import token_accuracy
from models import DecoderOnlyTransformer, PerTokenMemoryTransformer, RecurrentMemoryTransformer, TransformerConfig


MODEL_REGISTRY = {
    "baseline": DecoderOnlyTransformer,
    "per_token": PerTokenMemoryTransformer,
    "recurrent": RecurrentMemoryTransformer,
}


def get_git_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def build_model(model_name: str, config: TransformerConfig):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name={model_name!r}. Choices: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](config)


def memory_budget_for_model(model_name: str, config: TransformerConfig, sequence_length: int) -> Dict:
    if model_name == "baseline":
        return baseline_kv_memory_budget(config, sequence_length)
    if model_name == "per_token":
        return per_token_memory_budget(config, sequence_length)
    if model_name == "recurrent":
        return recurrent_memory_budget(config)
    raise ValueError(model_name)


def append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_synthetic(config: Dict) -> Dict:
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)

    requested_device = config.get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    dataset_cfg = config.get("dataset", {})
    dataset = KeyValueRetrievalDataset(**dataset_cfg)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    batch_size = int(config.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model_cfg = TransformerConfig(**config["model"])
    model = build_model(config["model_name"], model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    tokens_processed = 0
    log_rows = []
    log_path = Path(config.get("log_path", "logs/synthetic_runs.jsonl"))
    max_steps = int(config.get("max_steps", 200))
    eval_every = int(config.get("eval_every", 50))
    step = 0

    while step < max_steps:
        for input_ids, targets in train_loader:
            step += 1
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            out = model(input_ids, targets=targets)
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            optimizer.step()
            tokens_processed += int(input_ids.numel())

            if step == 1 or step % eval_every == 0 or step == max_steps:
                val_loss = evaluate_loss(model, val_loader, device)
                val_acc = token_accuracy(model, val_loader, device)
                elapsed = time.time() - start_time
                peak_gpu_mb = (
                    torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
                )
                row = {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "validation_loss": float(val_loss),
                    "validation_answer_accuracy": float(val_acc),
                    "tokens_processed": tokens_processed,
                    "training_time_sec": elapsed,
                    "tokens_per_sec": tokens_processed / max(elapsed, 1e-9),
                    "peak_gpu_memory_mb": peak_gpu_mb,
                }
                log_rows.append(row)
                append_jsonl(log_path, {"event": "metrics", **row})
                print(
                    f"step {step:04d} | train {row['train_loss']:.4f} | "
                    f"val {val_loss:.4f} | acc {val_acc:.3f}"
                )

            if step >= max_steps:
                break

    summary = {
        "config": config,
        "git_hash": get_git_hash(),
        "parameter_count": parameter_breakdown(model),
        "memory_budget": memory_budget_for_model(
            config["model_name"], model_cfg, sequence_length=dataset[0][0].numel()
        ),
        "final_metrics": log_rows[-1] if log_rows else {},
    }
    append_jsonl(log_path, {"event": "summary", **summary})
    csv_path = Path(config.get("csv_path", "logs/synthetic_metrics.csv"))
    write_csv(csv_path, log_rows)
    return summary


def load_real_text_from_config(config: Dict) -> str:
    data_cfg = config.get("data", {})
    source = data_cfg.get("source", "tinystories")
    max_chars = data_cfg.get("max_chars")
    if source == "tinystories":
        return load_tinystories_text(
            split=data_cfg.get("split", "train"),
            max_examples=data_cfg.get("max_examples", 10_000),
            max_chars=max_chars,
            cache_dir=data_cfg.get("cache_dir", "data/hf_cache"),
            offline=bool(data_cfg.get("offline", False)),
        )
    if source == "text_file":
        return load_text_file(data_cfg["path"], max_chars=max_chars)
    raise ValueError("data.source must be 'tinystories' or 'text_file'")


def train_language_model(config: Dict) -> Dict:
    """Train next-token LM on real text.

    This uses the same model registry and accounting as the synthetic trainer,
    but the target is every next byte token in a real text stream.
    """

    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    requested_device = config.get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)

    text = load_real_text_from_config(config)
    train_ds, val_ds, tokenizer = build_lm_datasets(
        text,
        block_size=int(config["model"]["context_length"]),
        val_fraction=float(config.get("val_fraction", 0.05)),
        tokenizer_config=config.get("tokenizer", {"kind": "tiktoken", "encoding": "gpt2"}),
        block_stride=config.get("block_stride"),
    )
    model_cfg_dict = dict(config["model"])
    model_cfg_dict["vocab_size"] = tokenizer.vocab_size
    model_cfg = TransformerConfig(**model_cfg_dict)

    batch_size = int(config.get("batch_size", 8))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(config["model_name"], model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    tokens_processed = 0
    log_rows = []
    log_path = Path(config.get("log_path", "logs/real_lm_runs.jsonl"))
    max_steps_cfg = config.get("max_steps")
    max_steps = int(max_steps_cfg) if max_steps_cfg is not None else None
    num_epochs = int(config.get("num_epochs", 1))
    eval_every = int(config.get("eval_every", 50))
    step = 0

    for epoch in range(1, num_epochs + 1):
        for input_ids, targets in train_loader:
            step += 1
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            out = model(input_ids, targets=targets)
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            optimizer.step()
            tokens_processed += int(input_ids.numel())

            is_last_step = max_steps is not None and step >= max_steps
            is_epoch_end = step % len(train_loader) == 0
            if step == 1 or step % eval_every == 0 or is_last_step or is_epoch_end:
                val_loss = evaluate_loss(model, val_loader, device)
                elapsed = time.time() - start_time
                peak_gpu_mb = (
                    torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
                )
                row = {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    "validation_loss": float(val_loss),
                    "tokens_processed": tokens_processed,
                    "training_time_sec": elapsed,
                    "tokens_per_sec": tokens_processed / max(elapsed, 1e-9),
                    "peak_gpu_memory_mb": peak_gpu_mb,
                }
                log_rows.append(row)
                append_jsonl(log_path, {"event": "metrics", **row})
                print(f"step {step:04d} | train {row['train_loss']:.4f} | val {val_loss:.4f}")

            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    summary = {
        "config": config,
        "git_hash": get_git_hash(),
        "tokenizer": tokenizer_metadata(tokenizer),
        "dataset": {
            "num_train_blocks": len(train_ds),
            "num_validation_blocks": len(val_ds),
            "num_text_chars": len(text),
            "context_length": model_cfg.context_length,
            "block_stride": train_ds.stride,
            "train_tokens_per_epoch": len(train_ds) * model_cfg.context_length,
            "validation_tokens_per_eval": len(val_ds) * model_cfg.context_length,
        },
        "parameter_count": parameter_breakdown(model),
        "memory_budget": memory_budget_for_model(
            config["model_name"], model_cfg, sequence_length=model_cfg.context_length
        ),
        "final_metrics": log_rows[-1] if log_rows else {},
    }
    append_jsonl(log_path, {"event": "summary", **summary})
    csv_path = Path(config.get("csv_path", "logs/real_lm_metrics.csv"))
    write_csv(csv_path, log_rows)
    return summary
