"""Evaluate whether salience-selected per-token memory slots can be dropped.

This is an oracle compression probe for the per-token memory model. It first
computes token salience by ablating each compressed K/V slot, then evaluates
accuracy after retaining only a fraction of the most salient slots. Random and
uniform-stride retention are reported as controls at the same keep fraction.
"""

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
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from data.synthetic import ANSWER, COPY, CopyDataset, collate_batch  # noqa: E402
from training.trainer import build_model, memory_budget_for_model  # noqa: E402
from evaluation.efficiency import parameter_breakdown  # noqa: E402
from run_token_salience import (  # noqa: E402
    build_dataset,
    build_test_dataset,
    load_configured_model,
    make_synthetic_config,
    requested_device,
    token_salience,
    train_synthetic_per_token,
)


@torch.no_grad()
def task_accuracy(task: str, model, input_ids: torch.Tensor, targets: torch.Tensor, memory_overrides=None) -> float:
    logits = model(input_ids, memory_overrides=memory_overrides)["logits"]
    pred = logits.argmax(dim=-1)
    if task == "copy":
        copy_marker = input_ids == COPY
        mask = (copy_marker.cumsum(dim=1) > 0) & (targets != -100)
    elif task in {"needle", "kv"}:
        answer_marker = input_ids == ANSWER
        answer_positions = torch.zeros_like(targets, dtype=torch.bool)
        marker_idx = answer_marker.float().argmax(dim=1)
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        answer_positions[rows, marker_idx] = targets[rows, marker_idx] != -100
        mask = answer_positions
    else:
        mask = targets != -100
    return float((pred[mask] == targets[mask]).float().mean().item()) if mask.any() else 0.0


@torch.no_grad()
def task_loss(model, input_ids: torch.Tensor, targets: torch.Tensor, memory_overrides=None) -> float:
    logits = model(input_ids, memory_overrides=memory_overrides)["logits"]
    return float(
        F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        ).item()
    )


def memory_with_keep_mask(caches, keep_mask: torch.Tensor, drop_fill: str):
    """Replace dropped token slots with mean or zero memory."""

    mask = keep_mask[:, None, :, None]
    overrides = []
    for key, value in caches:
        if drop_fill == "mean":
            fill_key = key.mean(dim=2, keepdim=True).expand_as(key)
            fill_value = value.mean(dim=2, keepdim=True).expand_as(value)
        elif drop_fill == "zero":
            fill_key = torch.zeros_like(key)
            fill_value = torch.zeros_like(value)
        else:
            raise ValueError(f"Unknown drop_fill={drop_fill!r}")
        key2 = torch.where(mask, key, fill_key)
        value2 = torch.where(mask, value, fill_value)
        overrides.append((key2, value2))
    return overrides


def topk_mask(scores: torch.Tensor, keep_frac: float) -> torch.Tensor:
    batch, seq_len = scores.shape
    k = max(0, min(seq_len, round(seq_len * keep_frac)))
    keep = torch.zeros_like(scores, dtype=torch.bool)
    if k == 0:
        return keep
    idx = torch.topk(scores, k=k, dim=1).indices
    keep.scatter_(1, idx, True)
    return keep


def random_mask(batch: int, seq_len: int, keep_frac: float, device: torch.device, generator: torch.Generator):
    k = max(0, min(seq_len, round(seq_len * keep_frac)))
    keep = torch.zeros((batch, seq_len), dtype=torch.bool, device=device)
    if k == 0:
        return keep
    for row in range(batch):
        idx = torch.randperm(seq_len, generator=generator, device=device)[:k]
        keep[row, idx] = True
    return keep


def stride_mask(batch: int, seq_len: int, keep_frac: float, device: torch.device):
    keep = torch.zeros((batch, seq_len), dtype=torch.bool, device=device)
    k = max(0, min(seq_len, round(seq_len * keep_frac)))
    if k == 0:
        return keep
    idx = torch.linspace(0, seq_len - 1, steps=k, device=device).round().long().unique()
    keep[:, idx] = True
    return keep


def evaluate_with_mask(task: str, model, input_ids, targets, keep_mask, drop_fill: str):
    base_out = model(input_ids)
    overrides = memory_with_keep_mask(base_out["cache"], keep_mask, drop_fill)
    return {
        "accuracy": task_accuracy(task, model, input_ids, targets, overrides),
        "loss": task_loss(model, input_ids, targets, overrides),
    }


def make_model_and_loader(args, device: torch.device):
    dataset, vocab_size, context_length = build_dataset(args.task, args)
    test_ds = build_test_dataset(args.task, args)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    if args.checkpoint:
        if not args.config:
            raise ValueError("--config is required with --checkpoint")
        model = load_configured_model(Path(args.config), Path(args.checkpoint), device)
    else:
        model = build_model("per_token", make_synthetic_config(vocab_size, context_length)).to(device)
        if args.train_steps <= 0:
            raise ValueError("Use --train-steps > 0 when no checkpoint is provided")
        val_size = max(1, int(0.1 * len(dataset)))
        train_ds, _ = random_split(
            dataset,
            [len(dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        train_synthetic_per_token(model, train_loader, args, device)

    return model, test_loader, context_length


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["copy", "needle", "kv"], default="copy")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--num-examples", type=int, default=10000)
    parser.add_argument("--test-examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--analyze-batches", type=int, default=4)
    parser.add_argument("--keep-fracs", nargs="+", type=float, default=[0.5, 0.25, 0.1, 0.05])
    parser.add_argument(
        "--drop-fill",
        choices=["mean", "zero"],
        default="mean",
        help="Replacement for dropped memory slots.",
    )
    parser.add_argument("--random-trials", type=int, default=3)
    parser.add_argument("--copy-length", type=int, default=32)
    parser.add_argument("--gap-length", type=int, default=64)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--num-pairs", type=int, default=16)
    parser.add_argument("--num-keys", type=int, default=32)
    parser.add_argument("--num-values", type=int, default=32)
    parser.add_argument("--value-mode", choices=["random", "identity", "shifted"], default="random")
    parser.add_argument("--vocab-tokens", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv-path", default="logs/token_retention.csv")
    args = parser.parse_args()

    if any(frac < 0 or frac > 1 for frac in args.keep_fracs):
        raise ValueError("--keep-fracs must be in [0, 1]")

    torch.manual_seed(args.seed)
    device = requested_device(args.device)
    model, test_loader, sequence_length = make_model_and_loader(args, device)
    params = parameter_breakdown(model)["total"]
    full_mem = memory_budget_for_model("per_token", model.config, sequence_length)["floats"]
    rng = torch.Generator(device=device).manual_seed(args.seed + 99_000)

    rows = []
    start = time.time()
    for batch_idx, (input_ids, targets) in enumerate(test_loader):
        if batch_idx >= args.analyze_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        full_acc = task_accuracy(args.task, model, input_ids, targets)
        full_loss = task_loss(model, input_ids, targets)
        rows.append(
            {
                "task": args.task,
                "batch": batch_idx,
                "method": "full",
                "keep_frac": 1.0,
                "kept_tokens": input_ids.size(1),
                "memory_floats": full_mem,
                "memory_fraction": 1.0,
                "params": params,
                "accuracy": full_acc,
                "loss": full_loss,
            }
        )

        salience = token_salience(model, input_ids, targets, args.task)
        for keep_frac in args.keep_fracs:
            kept_tokens = max(0, min(input_ids.size(1), round(input_ids.size(1) * keep_frac)))
            compressed_mem = round(full_mem * kept_tokens / input_ids.size(1))
            oracle = evaluate_with_mask(
                args.task,
                model,
                input_ids,
                targets,
                topk_mask(salience, keep_frac),
                args.drop_fill,
            )
            rows.append(
                {
                    "task": args.task,
                    "batch": batch_idx,
                    "method": "oracle_topk",
                    "keep_frac": keep_frac,
                    "kept_tokens": kept_tokens,
                    "memory_floats": compressed_mem,
                    "memory_fraction": compressed_mem / full_mem,
                    "params": params,
                    **oracle,
                }
            )

            stride = evaluate_with_mask(
                args.task,
                model,
                input_ids,
                targets,
                stride_mask(input_ids.size(0), input_ids.size(1), keep_frac, device),
                args.drop_fill,
            )
            rows.append(
                {
                    "task": args.task,
                    "batch": batch_idx,
                    "method": "stride_topk",
                    "keep_frac": keep_frac,
                    "kept_tokens": kept_tokens,
                    "memory_floats": compressed_mem,
                    "memory_fraction": compressed_mem / full_mem,
                    "params": params,
                    **stride,
                }
            )

            random_metrics = []
            for _ in range(args.random_trials):
                random_metrics.append(
                    evaluate_with_mask(
                        args.task,
                        model,
                        input_ids,
                        targets,
                        random_mask(input_ids.size(0), input_ids.size(1), keep_frac, device, rng),
                        args.drop_fill,
                    )
                )
            rows.append(
                {
                    "task": args.task,
                    "batch": batch_idx,
                    "method": "random_topk",
                    "keep_frac": keep_frac,
                    "kept_tokens": kept_tokens,
                    "memory_floats": compressed_mem,
                    "memory_fraction": compressed_mem / full_mem,
                    "params": params,
                    "accuracy": sum(m["accuracy"] for m in random_metrics) / len(random_metrics),
                    "loss": sum(m["loss"] for m in random_metrics) / len(random_metrics),
                }
            )

        print(f"analyzed batch {batch_idx + 1}", flush=True)

    if not rows:
        raise RuntimeError("No batches analyzed")
    write_rows(Path(args.csv_path), rows)

    by_method = {}
    for row in rows:
        key = (row["method"], row["keep_frac"])
        by_method.setdefault(key, []).append(row)
    print("\nRetention summary")
    for key in sorted(by_method, key=lambda item: (item[1], item[0])):
        vals = by_method[key]
        mean_acc = sum(float(v["accuracy"]) for v in vals) / len(vals)
        mean_loss = sum(float(v["loss"]) for v in vals) / len(vals)
        method, keep_frac = key
        print(f"{method:12s} keep={keep_frac:.2f} | acc {mean_acc:.3f} | loss {mean_loss:.4f}")
    print(f"\nwrote {args.csv_path}")
    print(f"elapsed {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
