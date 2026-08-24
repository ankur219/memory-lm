"""Run a synthetic key-value retrieval difficulty sweep.

This probe asks a sharper memory question than TinyStories LM loss:

    how many exact key-value facts can each architecture store and retrieve?

For each num_pairs setting we train fresh small models on the same task and log
answer-token accuracy. The per-token and recurrent configs are matched by
persistent-memory floats for the actual sequence length of that difficulty.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from data.synthetic import KeyValueRetrievalDataset, collate_batch
from evaluation.efficiency import (
    matched_recurrent_dim_for_per_token,
    parameter_breakdown,
)
from evaluation.memory_tasks import token_accuracy
from models import TransformerConfig
from training.trainer import build_model, memory_budget_for_model


def make_config(model_name: str, num_pairs: int, vocab_size: int) -> TransformerConfig:
    seq_len = 2 * num_pairs + 5
    context_length = max(32, seq_len)
    base = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        context_length=context_length,
        mlp_ratio=4.0,
        dropout=0.0,
        tie_embeddings=True,
        memory_dim=64,
        num_memory_tokens=8,
        recurrent_update_rank=4,
        recurrent_compressed_attention=True,
        recurrent_learned_initial=False,
        recurrent_update_style="cross_attention",
        chunk_size=32,
    )
    if model_name == "recurrent":
        base.recurrent_memory_dim = matched_recurrent_dim_for_per_token(
            base,
            sequence_length=seq_len,
            num_memory_tokens=base.num_memory_tokens,
        )
    return base


def train_one(model_name: str, num_pairs: int, args) -> dict:
    seed = args.seed + num_pairs * 100
    torch.manual_seed(seed)
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)

    train_val = KeyValueRetrievalDataset(
        num_examples=args.num_examples,
        num_pairs=num_pairs,
        num_keys=args.num_keys,
        num_values=args.num_values,
        seed=seed,
    )
    test_ds = KeyValueRetrievalDataset(
        num_examples=args.test_examples,
        num_pairs=num_pairs,
        num_keys=args.num_keys,
        num_values=args.num_values,
        seed=seed + 1_000_000,
    )
    val_size = max(1, int(0.1 * len(train_val)))
    train_size = len(train_val) - val_size
    train_ds, val_ds = random_split(
        train_val,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    cfg = make_config(model_name, num_pairs, train_val.vocab.size)
    model = build_model(model_name, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    step = 0
    last_loss = None
    while step < args.steps:
        for input_ids, targets in train_loader:
            step += 1
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            out = model(input_ids, targets=targets)
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            last_loss = float(loss.item())
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                val_acc = token_accuracy(model, val_loader, device)
                print(
                    f"pairs {num_pairs:03d} | {model_name:9s} | "
                    f"step {step:04d} | loss {last_loss:.4f} | val_acc {val_acc:.3f}",
                    flush=True,
                )
            if step >= args.steps:
                break

    test_acc = token_accuracy(model, test_loader, device)
    elapsed = time.time() - start
    seq_len = 2 * num_pairs + 5
    mem = memory_budget_for_model(model_name, cfg, sequence_length=seq_len)
    params = parameter_breakdown(model)
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    return {
        "model": model_name,
        "num_pairs": num_pairs,
        "sequence_length": seq_len,
        "steps": args.steps,
        "train_loss": last_loss,
        "test_answer_accuracy": test_acc,
        "params": params["total"],
        "memory_floats": mem["floats"],
        "recurrent_memory_dim": cfg.recurrent_memory_dim or "",
        "training_time_sec": elapsed,
        "peak_gpu_memory_mb": peak_gpu_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument("--models", nargs="+", default=["baseline", "per_token", "recurrent"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-examples", type=int, default=5000)
    parser.add_argument("--test-examples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-keys", type=int, default=128)
    parser.add_argument("--num-values", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv-path", default="logs/kv_sweep.csv")
    args = parser.parse_args()

    rows = []
    for num_pairs in args.pairs:
        for model_name in args.models:
            rows.append(train_one(model_name, num_pairs, args))
            Path(args.csv_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(args.csv_path).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    print(f"\nwrote {args.csv_path}")


if __name__ == "__main__":
    main()
