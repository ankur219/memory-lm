"""Estimate token-level salience for per-token memory.

This script measures how load-bearing each token-indexed memory slot is by
ablating one position's compressed per-token K/V memory at every layer and
measuring downstream loss damage.

The fastest trustworthy use is synthetic validation:

    python3 experiments/run_token_salience.py --task copy --train-steps 1000
    python3 experiments/run_token_salience.py --task needle --train-steps 1000

For trained real-LM checkpoints, pass both a checkpoint and its YAML config.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.synthetic import (  # noqa: E402
    ANSWER,
    COPY,
    NEEDLE,
    CopyDataset,
    KeyValueRetrievalDataset,
    NeedleDataset,
    collate_batch,
)
from training.trainer import build_model  # noqa: E402
from models import TransformerConfig  # noqa: E402


def requested_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def make_synthetic_config(vocab_size: int, context_length: int) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        context_length=max(32, context_length),
        mlp_ratio=4.0,
        dropout=0.0,
        tie_embeddings=True,
        memory_dim=64,
        chunk_size=32,
    )


def load_configured_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    if raw_cfg["model_name"] != "per_token":
        raise ValueError("Token salience currently supports model_name=per_token only")
    model = build_model("per_token", TransformerConfig(**raw_cfg["model"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model.to(device)


def build_dataset(task: str, args):
    if task == "copy":
        dataset = CopyDataset(
            num_examples=args.num_examples,
            copy_length=args.copy_length,
            vocab_tokens=args.vocab_tokens,
            seed=args.seed,
            supervise_all_tokens=True,
        )
        return dataset, dataset.vocab_size, 2 * args.copy_length + 3
    if task == "needle":
        dataset = NeedleDataset(
            num_examples=args.num_examples,
            prefix_length=args.prefix_length,
            gap_length=args.gap_length,
            vocab_tokens=args.vocab_tokens,
            num_values=args.num_values,
            seed=args.seed,
            supervise_all_tokens=True,
        )
        return dataset, dataset.vocab_size, args.prefix_length + args.gap_length + 6
    if task == "kv":
        dataset = KeyValueRetrievalDataset(
            num_examples=args.num_examples,
            num_pairs=args.num_pairs,
            num_keys=args.num_keys,
            num_values=args.num_values,
            seed=args.seed,
            supervise_all_tokens=True,
            value_mode=args.value_mode,
        )
        return dataset, dataset.vocab.size, 2 * args.num_pairs + 5
    raise ValueError(task)


def build_test_dataset(task: str, args):
    test_args = argparse.Namespace(**vars(args))
    test_args.num_examples = args.test_examples
    test_args.seed = args.seed + 1_000_000
    return build_dataset(task, test_args)[0]


def train_synthetic_per_token(model, train_loader, args, device: torch.device) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    step = 0
    start = time.time()
    while step < args.train_steps:
        for input_ids, targets in train_loader:
            step += 1
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            loss = model(input_ids, targets=targets)["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.train_steps:
                print(f"step {step:04d} | train_loss {loss.item():.4f}", flush=True)
            if step >= args.train_steps:
                break
    print(f"trained synthetic per_token for {step} steps in {time.time() - start:.1f}s")


def clone_with_position_ablated(caches, position: int):
    ablated = []
    for key, value in caches:
        key2 = key.clone()
        value2 = value.clone()
        mean_key = key2.mean(dim=2, keepdim=True)
        mean_value = value2.mean(dim=2, keepdim=True)
        key2[:, :, position : position + 1, :] = mean_key
        value2[:, :, position : position + 1, :] = mean_value
        ablated.append((key2, value2))
    return ablated


def per_position_labels(input_ids: torch.Tensor, task: str) -> list[str]:
    ids = input_ids[0].detach().cpu().tolist()
    labels = []
    for i, token in enumerate(ids):
        label = "other"
        if token == COPY:
            label = "copy_marker"
        elif token == NEEDLE:
            label = "needle_marker"
        elif token == ANSWER:
            label = "answer_marker"
        elif task == "copy" and COPY in ids:
            copy_idx = ids.index(COPY)
            if 0 < i < copy_idx:
                label = "copy_source"
            elif i > copy_idx:
                label = "copy_output"
        elif task == "needle" and NEEDLE in ids:
            needle_idx = ids.index(NEEDLE)
            if i == needle_idx + 1:
                label = "needle_value"
            elif i > needle_idx + 1:
                label = "post_needle"
        labels.append(label)
    return labels


@torch.no_grad()
def token_salience(model, input_ids: torch.Tensor, targets: torch.Tensor, task: str) -> torch.Tensor:
    """Return downstream loss damage for ablating each token's memory slot."""

    model.eval()
    base_out = model(input_ids)
    base_logits = base_out["logits"]
    caches = base_out["cache"]
    base_loss = F.cross_entropy(
        base_logits.reshape(-1, base_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(targets)

    batch, seq_len = input_ids.shape
    salience = torch.zeros(batch, seq_len, device=input_ids.device)
    for pos in range(seq_len):
        ablated = clone_with_position_ablated(caches, pos)
        logits = model(input_ids, memory_overrides=ablated)["logits"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(targets)
        delta = loss - base_loss
        downstream = torch.zeros_like(delta, dtype=torch.bool)
        downstream[:, pos + 1 :] = targets[:, pos + 1 :] != -100
        salience[:, pos] = delta.masked_fill(~downstream, 0.0).sum(dim=1)

    return salience.clamp_min(0.0)


def summarize_salience(salience: torch.Tensor, thresholds: Iterable[float]) -> dict:
    flat = salience.detach().cpu().flatten().numpy()
    total = float(flat.sum())
    if total <= 0:
        return {
            "tokens_analyzed": len(flat),
            "total_salience": 0.0,
            "top_10pct_mass": 0.0,
            **{f"frac_for_{int(t * 100)}pct": 1.0 for t in thresholds},
        }
    sorted_flat = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_flat) / total
    out = {
        "tokens_analyzed": len(flat),
        "total_salience": total,
        "top_10pct_mass": float(cumsum[max(0, math.ceil(0.1 * len(flat)) - 1)]),
    }
    for threshold in thresholds:
        n = int(np.searchsorted(cumsum, threshold) + 1)
        out[f"frac_for_{int(threshold * 100)}pct"] = n / len(flat)
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    parser.add_argument("--copy-length", type=int, default=32)
    parser.add_argument("--gap-length", type=int, default=32)
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
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.8, 0.9, 0.95])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv-path", default="logs/token_salience.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = requested_device(args.device)

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

    all_salience = []
    label_mass: dict[str, float] = {}
    rows = []
    for batch_idx, (input_ids, targets) in enumerate(test_loader):
        if batch_idx >= args.analyze_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        salience = token_salience(model, input_ids, targets, args.task)
        all_salience.append(salience.cpu())
        labels = per_position_labels(input_ids, args.task)
        mean_salience = salience.mean(dim=0).detach().cpu().tolist()
        for pos, value in enumerate(mean_salience):
            label_mass[labels[pos]] = label_mass.get(labels[pos], 0.0) + float(value)
            rows.append(
                {
                    "task": args.task,
                    "batch": batch_idx,
                    "position": pos,
                    "token_id": int(input_ids[0, pos].item()),
                    "label": labels[pos],
                    "mean_salience": float(value),
                }
            )

    if not all_salience:
        raise RuntimeError("No batches analyzed")
    salience_tensor = torch.cat(all_salience, dim=0)
    summary = summarize_salience(salience_tensor, args.thresholds)

    total_label_mass = sum(label_mass.values()) or 1.0
    print("\nSalience summary")
    for key, value in summary.items():
        if key.startswith("frac_for_"):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nMass by token label")
    for label, mass in sorted(label_mass.items(), key=lambda item: item[1], reverse=True):
        print(f"{label}: {mass / total_label_mass:.3f}")

    summary_row = {"task": args.task, "batch": "summary", "position": "", "token_id": "", "label": "summary"}
    summary_row.update(summary)
    rows.append(summary_row)
    write_rows(Path(args.csv_path), rows)
    print(f"\nwrote {args.csv_path}")


if __name__ == "__main__":
    main()
