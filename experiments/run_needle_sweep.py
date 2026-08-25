"""Run a synthetic needle-in-context retrieval sweep."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.synthetic import ANSWER, NeedleDataset, collate_batch
from evaluation.efficiency import matched_recurrent_dim_for_per_token, parameter_breakdown
from models import TransformerConfig
from training.trainer import build_model, memory_budget_for_model


@torch.no_grad()
def answer_accuracy(model, dataloader, device: torch.device) -> float:
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


def weighted_loss(logits, input_ids, targets, answer_weight: float):
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


def parse_recurrent_shape(shape: str) -> tuple[int, int]:
    """Parse strings like '256x128' into (num_memory_tokens, memory_dim)."""

    try:
        tokens, dim = shape.lower().split("x", maxsplit=1)
        return int(tokens), int(dim)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("recurrent shapes must look like 256x128") from exc


def make_config(
    model_name: str,
    sequence_length: int,
    vocab_size: int,
    recurrent_update_style: str = "cross_attention",
    recurrent_shape: tuple[int, int] | None = None,
) -> TransformerConfig:
    cfg = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        context_length=max(32, sequence_length),
        mlp_ratio=4.0,
        dropout=0.0,
        tie_embeddings=True,
        memory_dim=64,
        num_memory_tokens=8,
        recurrent_update_rank=4,
        recurrent_compressed_attention=True,
        recurrent_learned_initial=False,
        recurrent_update_style=recurrent_update_style,
        chunk_size=32,
    )
    if model_name == "recurrent":
        if recurrent_shape is not None:
            cfg.num_memory_tokens = recurrent_shape[0]
            cfg.recurrent_memory_dim = recurrent_shape[1]
        else:
            cfg.recurrent_memory_dim = matched_recurrent_dim_for_per_token(
                cfg,
                sequence_length=sequence_length,
                num_memory_tokens=cfg.num_memory_tokens,
            )
    elif model_name == "assoc_recurrent":
        cfg.num_memory_tokens = recurrent_shape[0] if recurrent_shape is not None else 256
        cfg.recurrent_memory_dim = recurrent_shape[1] if recurrent_shape is not None else cfg.hidden_size
    return cfg


def train_one(model_name: str, gap_length: int, args, recurrent_shape: tuple[int, int] | None = None) -> dict:
    seed = args.seed + gap_length * 100
    torch.manual_seed(seed)
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)

    train_val = NeedleDataset(
        num_examples=args.num_examples,
        prefix_length=args.prefix_length,
        gap_length=gap_length,
        vocab_tokens=args.vocab_tokens,
        num_values=args.num_values,
        seed=seed,
        supervise_all_tokens=args.supervise_all_tokens,
    )
    test_ds = NeedleDataset(
        num_examples=args.test_examples,
        prefix_length=args.prefix_length,
        gap_length=gap_length,
        vocab_tokens=args.vocab_tokens,
        num_values=args.num_values,
        seed=seed + 1_000_000,
        supervise_all_tokens=args.supervise_all_tokens,
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

    sequence_length = train_val[0][0].numel()
    cfg = make_config(
        model_name,
        sequence_length,
        train_val.vocab_size,
        recurrent_update_style=args.recurrent_update_style,
        recurrent_shape=recurrent_shape,
    )
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
            loss = weighted_loss(logits, input_ids, targets, args.answer_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            last_loss = float(loss.item())
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                val_acc = answer_accuracy(model, val_loader, device)
                print(
                    f"gap {gap_length:03d} | {model_name:15s} | {shape_label:7s} | "
                    f"step {step:04d} | loss {last_loss:.4f} | answer_acc {val_acc:.3f}",
                    flush=True,
                )
            if step >= args.steps:
                break

    test_acc = answer_accuracy(model, test_loader, device)
    diagnostics = collect_diagnostics(model, test_loader, device)
    elapsed = time.time() - start
    mem = memory_budget_for_model(model_name, cfg, sequence_length=sequence_length)
    params = parameter_breakdown(model)
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    return {
        "model": model_name,
        "recurrent_shape": shape_label,
        "recurrent_update_style": cfg.recurrent_update_style if model_name == "recurrent" else "",
        "gap_length": gap_length,
        "sequence_length": sequence_length,
        "steps": args.steps,
        "supervise_all_tokens": args.supervise_all_tokens,
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
    parser.add_argument("--gaps", nargs="+", type=int, default=[8, 16, 32, 64])
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
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--vocab-tokens", type=int, default=64)
    parser.add_argument("--num-values", type=int, default=64)
    parser.add_argument(
        "--recurrent-update-style",
        choices=["mean_gru", "cross_attention", "last_tokens"],
        default="cross_attention",
        help="Only applies to the naive recurrent model. assoc_recurrent uses explicit associative read/write.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--answer-loss-weight", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--supervise-all-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dense next-token loss while reporting answer-token accuracy.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv-path", default="logs/needle_sweep.csv")
    args = parser.parse_args()

    rows = []
    for gap_length in args.gaps:
        for model_name in args.models:
            recurrent_shapes = (
                args.recurrent_shapes
                if model_name in {"recurrent", "assoc_recurrent"} and args.recurrent_shapes
                else [None]
            )
            for recurrent_shape in recurrent_shapes:
                rows.append(train_one(model_name, gap_length, args, recurrent_shape=recurrent_shape))
                Path(args.csv_path).parent.mkdir(parents=True, exist_ok=True)
                with Path(args.csv_path).open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

    print(f"\nwrote {args.csv_path}")


if __name__ == "__main__":
    main()
