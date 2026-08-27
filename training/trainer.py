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
from data.text import (
    build_lm_datasets,
    build_lm_datasets_from_texts,
    build_hf_memmap_datasets,
    build_tinystories_memmap_datasets,
    load_hf_text,
    load_text_file,
    load_tinystories_text,
    tokenizer_metadata,
)
from evaluation.efficiency import (
    baseline_kv_memory_budget,
    parameter_breakdown,
    per_token_memory_budget,
    recurrent_memory_budget,
)
from evaluation.language_modeling import evaluate_loss
from evaluation.memory_tasks import token_accuracy
from models import (
    AssociativeRecurrentMemoryTransformer,
    DecoderOnlyTransformer,
    PerTokenMemoryTransformer,
    RecurrentMemoryTransformer,
    RMTMemoryTransformer,
    TransformerConfig,
)


MODEL_REGISTRY = {
    "assoc_recurrent": AssociativeRecurrentMemoryTransformer,
    "baseline": DecoderOnlyTransformer,
    "per_token": PerTokenMemoryTransformer,
    "recurrent": RecurrentMemoryTransformer,
    "rmt": RMTMemoryTransformer,
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
    if model_name in {"recurrent", "assoc_recurrent", "rmt"}:
        return recurrent_memory_budget(config, per_layer_memory=config.per_layer_memory)
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


def save_training_checkpoint(
    path: Path,
    model,
    optimizer,
    config: Dict,
    step: int,
    epoch: int,
    tokens_processed: int,
    latest_metrics: Dict,
) -> None:
    """Save enough state to inspect or continue a run later.

    The current trainer does not yet implement automatic resume semantics, but
    long real-data runs should still leave recoverable model and optimizer
    snapshots instead of only CSV metrics.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "tokens_processed": tokens_processed,
            "config": config,
            "git_hash": get_git_hash(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "latest_metrics": latest_metrics,
        },
        path,
    )


@torch.no_grad()
def generate_token_sample(model, tokenizer, prompt: str, config: Dict, device: torch.device):
    """Generate token ids from the current checkpoint.

    This deliberately uses simple full-context decoding instead of a KV cache so
    the same code works for baseline, many-small, and few-rich models.
    """

    gen_cfg = config.get("generation", {})
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 80))
    temperature = float(gen_cfg.get("temperature", 0.8))
    top_k = gen_cfg.get("top_k", 50)
    top_k = int(top_k) if top_k is not None else None
    context_length = int(config["model"]["context_length"])

    model_was_training = model.training
    model.eval()
    prompt_token_ids = tokenizer.encode(prompt, add_eos=False) or [tokenizer.eos_token]
    token_ids = list(prompt_token_ids)

    for _ in range(max_new_tokens):
        context = token_ids[-context_length:]
        input_ids = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(input_ids)["logits"][0, -1]
        logits = logits / max(temperature, 1e-6)
        if top_k is not None and top_k < logits.numel():
            values, _ = torch.topk(logits, top_k)
            logits = logits.masked_fill(logits < values[-1], torch.finfo(logits.dtype).min)
        probs = torch.softmax(logits, dim=-1)
        next_token = int(torch.multinomial(probs, num_samples=1).item())
        token_ids.append(next_token)
        if next_token == tokenizer.eos_token:
            break

    if model_was_training:
        model.train()
    return prompt_token_ids, token_ids


def generate_text_sample(model, tokenizer, prompt: str, config: Dict, device: torch.device) -> str:
    """Return the full decoded prompt plus continuation sample."""

    _, token_ids = generate_token_sample(model, tokenizer, prompt, config, device)
    return tokenizer.decode(token_ids)


def maybe_print_generation_sample(model, tokenizer, config: Dict, step: int, device: torch.device) -> None:
    gen_cfg = config.get("generation", {})
    if not gen_cfg or not bool(gen_cfg.get("enabled", False)):
        return
    every_steps = int(gen_cfg.get("every_steps", config.get("eval_every", 50)))
    print_at_step_one = bool(gen_cfg.get("print_at_step_one", False))
    if step == 1 and not print_at_step_one:
        return
    if every_steps > 0 and step % every_steps != 0:
        return
    prompt = gen_cfg.get("prompt", "Once upon a time")
    prompt_token_ids, sample_token_ids = generate_token_sample(model, tokenizer, prompt, config, device)
    continuation_token_ids = sample_token_ids[len(prompt_token_ids) :]
    continuation = tokenizer.decode(continuation_token_ids).replace("\n", " ").strip()
    print(f"\n--- sample step {step:04d} ---")
    print("fed prompt:")
    print(prompt.replace("\n", " ").strip())
    print("\npredicted continuation:")
    print(continuation)
    print("--- end sample ---\n")


def train_synthetic(config: Dict) -> Dict:
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)

    requested_device = config.get("device", "auto")
    if requested_device == "auto":
        if torch.cuda.is_available():
            requested_device = "cuda"
        elif torch.backends.mps.is_available():
            requested_device = "mps"
        else:
            requested_device = "cpu"
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
    if source == "hf_text":
        return load_hf_text(
            dataset_name=data_cfg["dataset_name"],
            dataset_config=data_cfg.get("dataset_config"),
            split=data_cfg.get("split", "train"),
            text_field=data_cfg.get("text_field", "text"),
            max_examples=data_cfg.get("max_examples"),
            max_chars=max_chars,
            cache_dir=data_cfg.get("cache_dir", "data/hf_cache"),
            offline=bool(data_cfg.get("offline", False)),
        )
    raise ValueError("data.source must be 'tinystories', 'hf_text', or 'text_file'")


def load_validation_text_from_config(config: Dict) -> Optional[str]:
    val_cfg = config.get("validation_data")
    if not val_cfg:
        return None
    source = val_cfg.get("source", "tinystories")
    max_chars = val_cfg.get("max_chars")
    if source == "tinystories":
        return load_tinystories_text(
            split=val_cfg.get("split", "validation"),
            max_examples=val_cfg.get("max_examples"),
            max_chars=max_chars,
            cache_dir=val_cfg.get("cache_dir", "data/hf_cache"),
            offline=bool(val_cfg.get("offline", config.get("data", {}).get("offline", False))),
        )
    if source == "text_file":
        return load_text_file(val_cfg["path"], max_chars=max_chars)
    if source == "hf_text":
        return load_hf_text(
            dataset_name=val_cfg["dataset_name"],
            dataset_config=val_cfg.get("dataset_config"),
            split=val_cfg.get("split", "validation"),
            text_field=val_cfg.get("text_field", "text"),
            max_examples=val_cfg.get("max_examples"),
            max_chars=max_chars,
            cache_dir=val_cfg.get("cache_dir", "data/hf_cache"),
            offline=bool(val_cfg.get("offline", config.get("data", {}).get("offline", False))),
        )
    raise ValueError("validation_data.source must be 'tinystories', 'hf_text', or 'text_file'")


def build_real_lm_datasets_from_config(config: Dict):
    """Build real-data train/validation datasets from a config.

    For full TinyStories runs, prefer the memmap path. It uses the official
    TinyStories train and validation splits without materializing all token ids
    as a Python list.
    """

    block_size = int(config["model"]["context_length"])
    block_stride = config.get("block_stride")
    data_cfg = config.get("data", {})
    if data_cfg.get("cache_tokens", False):
        if data_cfg.get("source", "tinystories") == "hf_text":
            return build_hf_memmap_datasets(config, block_size=block_size, block_stride=block_stride)
        if data_cfg.get("source", "tinystories") != "tinystories":
            raise ValueError("cache_tokens is currently implemented for TinyStories and hf_text only.")
        return build_tinystories_memmap_datasets(config, block_size=block_size, block_stride=block_stride)

    train_text = load_real_text_from_config(config)
    validation_text = load_validation_text_from_config(config)
    if validation_text is not None:
        return build_lm_datasets_from_texts(
            train_text,
            validation_text,
            block_size=block_size,
            tokenizer_config=config.get("tokenizer", {"kind": "tiktoken", "encoding": "gpt2"}),
            block_stride=block_stride,
        )
    return build_lm_datasets(
        train_text,
        block_size=block_size,
        val_fraction=float(config.get("val_fraction", 0.05)),
        tokenizer_config=config.get("tokenizer", {"kind": "tiktoken", "encoding": "gpt2"}),
        block_stride=block_stride,
    )


def train_language_model(config: Dict) -> Dict:
    """Train next-token LM on real text.

    This uses the same model registry and accounting as the synthetic trainer,
    but the target is every next byte token in a real text stream.
    """

    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    requested_device = config.get("device", "auto")
    if requested_device == "auto":
        if torch.cuda.is_available():
            requested_device = "cuda"
        elif torch.backends.mps.is_available():
            requested_device = "mps"
        else:
            requested_device = "cpu"
    device = torch.device(requested_device)

    train_ds, val_ds, tokenizer = build_real_lm_datasets_from_config(config)
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
    eval_max_batches_cfg = config.get("eval_max_batches")
    eval_max_batches = int(eval_max_batches_cfg) if eval_max_batches_cfg is not None else None
    final_eval_max_batches_cfg = config.get("final_eval_max_batches")
    final_eval_max_batches = (
        int(final_eval_max_batches_cfg) if final_eval_max_batches_cfg is not None else None
    )
    checkpoint_dir_cfg = config.get("checkpoint_dir")
    checkpoint_dir = Path(checkpoint_dir_cfg) if checkpoint_dir_cfg else None
    save_every = int(config.get("save_every", 0))
    step = 0

    epoch = 0
    while True:
        epoch += 1
        for batch_idx, (input_ids, targets) in enumerate(train_loader, start=1):
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
            is_epoch_end = batch_idx == len(train_loader)
            is_final_step = is_last_step or (max_steps is None and is_epoch_end and epoch >= num_epochs)
            should_eval = step == 1 or step % eval_every == 0 or is_final_step or is_epoch_end
            if should_eval:
                val_limit = final_eval_max_batches if is_final_step else eval_max_batches
                val_loss = evaluate_loss(model, val_loader, device, max_batches=val_limit)
                elapsed = time.time() - start_time
                peak_gpu_mb = (
                    torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
                )
                row = {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    "validation_loss": float(val_loss),
                    "validation_batches": len(val_loader) if val_limit is None else min(val_limit, len(val_loader)),
                    "validation_full": val_limit is None or val_limit >= len(val_loader),
                    "tokens_processed": tokens_processed,
                    "training_time_sec": elapsed,
                    "tokens_per_sec": tokens_processed / max(elapsed, 1e-9),
                    "peak_gpu_memory_mb": peak_gpu_mb,
                }
                log_rows.append(row)
                append_jsonl(log_path, {"event": "metrics", **row})
                print(f"step {step:04d} | train {row['train_loss']:.4f} | val {val_loss:.4f}")
                if checkpoint_dir is not None and save_every > 0 and step % save_every == 0:
                    save_training_checkpoint(
                        checkpoint_dir / f"step_{step:06d}.pt",
                        model,
                        optimizer,
                        config,
                        step,
                        epoch,
                        tokens_processed,
                        row,
                    )
            maybe_print_generation_sample(model, tokenizer, config, step, device)

            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break
        if max_steps is None and epoch >= num_epochs:
            break

    if checkpoint_dir is not None:
        latest_metrics = log_rows[-1] if log_rows else {}
        save_training_checkpoint(
            checkpoint_dir / "final.pt",
            model,
            optimizer,
            config,
            step,
            epoch,
            tokens_processed,
            latest_metrics,
        )

    summary = {
        "config": config,
        "git_hash": get_git_hash(),
        "tokenizer": tokenizer_metadata(tokenizer),
        "dataset": {
            "num_train_blocks": len(train_ds),
            "num_validation_blocks": len(val_ds),
            "num_text_chars": None,
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
