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
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from data.synthetic import ANSWER, KeyValueRetrievalDataset, collate_batch
from evaluation.efficiency import (
    matched_recurrent_dim_for_per_token,
    parameter_breakdown,
)
from models import TransformerConfig
from training.trainer import build_model, memory_budget_for_model


def parse_recurrent_shape(shape: str) -> tuple[int, int]:
    """Parse strings like '256x128' into (num_memory_tokens, memory_dim)."""

    try:
        tokens, dim = shape.lower().split("x", maxsplit=1)
        return int(tokens), int(dim)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("recurrent shapes must look like 256x128") from exc


@torch.no_grad()
def answer_token_accuracy(model, dataloader, device: torch.device) -> float:
    """Accuracy only at the position after <ANSWER>.

    This stays a retrieval metric even when training uses dense next-token loss.
    The target answer token is predicted from the hidden state at the <ANSWER>
    position.
    """

    model.eval()
    correct = 0
    total = 0
    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        logits = model(input_ids)["logits"]
        pred = logits.argmax(dim=-1)
        answer_positions = input_ids == ANSWER
        correct += int((pred[answer_positions] == targets[answer_positions]).sum().item())
        total += int(answer_positions.sum().item())
    model.train()
    return correct / max(1, total)


def make_config(
    model_name: str,
    num_pairs: int,
    vocab_size: int,
    recurrent_shape: tuple[int, int] | None = None,
) -> TransformerConfig:
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
        if recurrent_shape is not None:
            base.num_memory_tokens = recurrent_shape[0]
            base.recurrent_memory_dim = recurrent_shape[1]
        else:
            base.recurrent_memory_dim = matched_recurrent_dim_for_per_token(
                base,
                sequence_length=seq_len,
                num_memory_tokens=base.num_memory_tokens,
            )
    elif model_name == "assoc_recurrent":
        base.num_memory_tokens = recurrent_shape[0] if recurrent_shape is not None else 256
        base.recurrent_memory_dim = recurrent_shape[1] if recurrent_shape is not None else base.hidden_size
    return base


def weighted_next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor, targets: torch.Tensor, answer_weight: float):
    """Cross-entropy with optional extra weight on answer-token positions."""

    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(targets)
    valid = targets != -100
    weights = valid.float()
    if answer_weight != 1.0:
        weights = weights * torch.where(
            input_ids == ANSWER,
            torch.full_like(weights, float(answer_weight)),
            torch.ones_like(weights),
        )
    return (per_token_loss * weights).sum() / weights.sum().clamp_min(1.0)


def train_one(model_name: str, num_pairs: int, args, recurrent_shape: tuple[int, int] | None = None) -> dict:
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
        supervise_all_tokens=args.supervise_all_tokens,
        value_mode=args.value_mode,
    )
    test_ds = KeyValueRetrievalDataset(
        num_examples=args.test_examples,
        num_pairs=num_pairs,
        num_keys=args.num_keys,
        num_values=args.num_values,
        seed=seed + 1_000_000,
        supervise_all_tokens=args.supervise_all_tokens,
        value_mode=args.value_mode,
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

    cfg = make_config(model_name, num_pairs, train_val.vocab.size, recurrent_shape=recurrent_shape)
    shape_label = (
        f"{cfg.num_memory_tokens}x{cfg.recurrent_memory_dim}"
        if model_name in {"recurrent", "assoc_recurrent"}
        else ""
    )
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
            logits = model(input_ids)["logits"]
            loss = weighted_next_token_loss(logits, input_ids, targets, args.answer_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            last_loss = float(loss.item())
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                val_acc = answer_token_accuracy(model, val_loader, device)
                print(
                    f"pairs {num_pairs:03d} | {model_name:15s} | {shape_label:7s} | "
                    f"step {step:04d} | loss {last_loss:.4f} | val_acc {val_acc:.3f}",
                    flush=True,
                )
            if step >= args.steps:
                break

    test_acc = answer_token_accuracy(model, test_loader, device)
    diagnostics = collect_diagnostics(model, test_loader, device)
    elapsed = time.time() - start
    seq_len = 2 * num_pairs + 5
    mem = memory_budget_for_model(model_name, cfg, sequence_length=seq_len)
    params = parameter_breakdown(model)
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    return {
        "model": model_name,
        "recurrent_shape": shape_label,
        "num_pairs": num_pairs,
        "sequence_length": seq_len,
        "steps": args.steps,
        "supervise_all_tokens": args.supervise_all_tokens,
        "value_mode": args.value_mode,
        "answer_loss_weight": args.answer_loss_weight,
        "train_loss": last_loss,
        "test_answer_accuracy": test_acc,
        "params": params["total"],
        "memory_floats": mem["floats"],
        "num_memory_tokens": cfg.num_memory_tokens if model_name in {"recurrent", "assoc_recurrent"} else "",
        "recurrent_memory_dim": cfg.recurrent_memory_dim or "",
        "training_time_sec": elapsed,
        "peak_gpu_memory_mb": peak_gpu_mb,
        **diagnostics,
    }


@torch.no_grad()
def collect_diagnostics(model, dataloader, device: torch.device) -> dict:
    """Collect optional recurrent-memory diagnostics from one held-out batch."""

    model.eval()
    try:
        input_ids, _ = next(iter(dataloader))
    except StopIteration:
        model.train()
        return {}
    out = model(input_ids.to(device))
    model.train()
    diagnostics = out.get("diagnostics", {})
    return {
        "read_entropy": diagnostics.get("read_entropy", ""),
        "write_entropy": diagnostics.get("write_entropy", ""),
        "memory_delta_norm": diagnostics.get("memory_delta_norm", ""),
        "memory_value_norm": diagnostics.get("memory_value_norm", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument("--models", nargs="+", default=["baseline", "per_token", "recurrent"])
    parser.add_argument(
        "--recurrent-shapes",
        nargs="+",
        type=parse_recurrent_shape,
        default=None,
        help="Optional recurrent shapes like 256x128. Applies to recurrent and assoc_recurrent.",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-examples", type=int, default=5000)
    parser.add_argument("--test-examples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-keys", type=int, default=128)
    parser.add_argument("--num-values", type=int, default=100)
    parser.add_argument("--value-mode", choices=["random", "identity", "shifted"], default="random")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--answer-loss-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--supervise-all-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dense next-token loss while still reporting answer-token accuracy.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv-path", default="logs/kv_sweep.csv")
    args = parser.parse_args()

    rows = []
    for num_pairs in args.pairs:
        for model_name in args.models:
            recurrent_shapes = (
                args.recurrent_shapes
                if model_name in {"recurrent", "assoc_recurrent"} and args.recurrent_shapes
                else [None]
            )
            for recurrent_shape in recurrent_shapes:
                rows.append(train_one(model_name, num_pairs, args, recurrent_shape=recurrent_shape))
                Path(args.csv_path).parent.mkdir(parents=True, exist_ok=True)
                with Path(args.csv_path).open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

    print(f"\nwrote {args.csv_path}")


if __name__ == "__main__":
    main()
