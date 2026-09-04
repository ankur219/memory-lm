"""Train KVM on synthetic tasks and measure compressed-slot usage.

This diagnostic asks whether KVM uses its fixed compressed K/V slots broadly or
collapses onto a small number of slots. It trains the same KVM configurations
used in the synthetic baseline (`128x64`, 32,768 memory floats), then aggregates
the final per-slot counts (`state_vlen`) returned by each KVM layer.

The output is a CSV with one row per task/setting/layer/head plus an aggregate
row per layer. High entropy/effective_slots means writes are spread across many
slots; high top1/top5 concentration means a few slots dominate.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.synthetic import CopyDataset, KeyValueRetrievalDataset, NeedleDataset, collate_batch
from evaluation.efficiency import parameter_breakdown
from experiments.run_copy_sweep import copy_accuracy, make_config as make_copy_config
from experiments.run_kv_sweep import answer_token_accuracy as kv_accuracy
from experiments.run_kv_sweep import make_config as make_kv_config
from experiments.run_kv_sweep import weighted_next_token_loss
from experiments.run_needle_sweep import answer_accuracy as needle_accuracy
from experiments.run_needle_sweep import make_config as make_needle_config
from experiments.run_needle_sweep import weighted_loss
from training.trainer import build_model, memory_budget_for_model


KVM_SHAPE = (128, 64)


@dataclass
class TaskSpec:
    task: str
    setting: int
    steps: int


def parse_task(text: str) -> TaskSpec:
    try:
        task, setting = text.split(":", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tasks must look like copy:128, needle:512, or kv:16") from exc
    if task not in {"copy", "needle", "kv"}:
        raise argparse.ArgumentTypeError("task must be copy, needle, or kv")
    return TaskSpec(task=task, setting=int(setting), steps=0)


def build_task(spec: TaskSpec, args: argparse.Namespace):
    seed = args.seed + spec.setting * 100
    if spec.task == "copy":
        train_val = CopyDataset(
            num_examples=args.num_examples,
            copy_length=spec.setting,
            vocab_tokens=args.vocab_tokens,
            seed=seed,
            supervise_all_tokens=True,
        )
        test_ds = CopyDataset(
            num_examples=args.test_examples,
            copy_length=spec.setting,
            vocab_tokens=args.vocab_tokens,
            seed=seed + 1_000_000,
            supervise_all_tokens=True,
        )
        cfg = make_copy_config("kvm", spec.setting, train_val.vocab_size, recurrent_shape=KVM_SHAPE)
        loss_fn = lambda logits, input_ids, targets: F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        acc_fn = copy_accuracy
        steps = args.copy_steps
    elif spec.task == "needle":
        train_val = NeedleDataset(
            num_examples=args.num_examples,
            prefix_length=args.prefix_length,
            gap_length=spec.setting,
            vocab_tokens=args.vocab_tokens,
            num_values=args.num_values,
            seed=seed,
            supervise_all_tokens=True,
        )
        test_ds = NeedleDataset(
            num_examples=args.test_examples,
            prefix_length=args.prefix_length,
            gap_length=spec.setting,
            vocab_tokens=args.vocab_tokens,
            num_values=args.num_values,
            seed=seed + 1_000_000,
            supervise_all_tokens=True,
        )
        sequence_length = train_val[0][0].numel()
        cfg = make_needle_config("kvm", sequence_length, train_val.vocab_size, recurrent_shape=KVM_SHAPE)
        loss_fn = lambda logits, input_ids, targets: weighted_loss(
            logits, input_ids, targets, args.answer_loss_weight
        )
        acc_fn = needle_accuracy
        steps = args.needle_steps
    else:
        train_val = KeyValueRetrievalDataset(
            num_examples=args.num_examples,
            num_pairs=spec.setting,
            num_keys=args.num_keys,
            num_values=args.num_values,
            seed=seed,
            supervise_all_tokens=True,
            value_mode=args.value_mode,
        )
        test_ds = KeyValueRetrievalDataset(
            num_examples=args.test_examples,
            num_pairs=spec.setting,
            num_keys=args.num_keys,
            num_values=args.num_values,
            seed=seed + 1_000_000,
            supervise_all_tokens=True,
            value_mode=args.value_mode,
        )
        cfg = make_kv_config("kvm", spec.setting, train_val.vocab.size, recurrent_shape=KVM_SHAPE)
        loss_fn = lambda logits, input_ids, targets: weighted_next_token_loss(
            logits, input_ids, targets, args.kv_answer_loss_weight
        )
        acc_fn = kv_accuracy
        steps = args.kv_steps

    val_size = max(1, int(0.1 * len(train_val)))
    train_size = len(train_val) - val_size
    train_ds, val_ds = random_split(
        train_val,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return cfg, train_ds, val_ds, test_ds, loss_fn, acc_fn, steps


def train_model(model, train_loader, val_loader, acc_fn, loss_fn, steps, args, device, label):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step = 0
    last_loss = None
    start = time.time()
    while step < steps:
        for input_ids, targets in train_loader:
            step += 1
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            logits = model(input_ids)["logits"]
            loss = loss_fn(logits, input_ids, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            last_loss = float(loss.item())
            if step == 1 or step % args.log_every == 0 or step == steps:
                val_acc = acc_fn(model, val_loader, device)
                print(f"{label} | step {step:04d} | loss {last_loss:.4f} | val_acc {val_acc:.3f}", flush=True)
            if step >= steps:
                break
    return last_loss, time.time() - start


@torch.no_grad()
def collect_slot_usage(model, dataloader, device, max_batches: int):
    model.eval()
    layer_loads = None
    batches = 0
    for input_ids, _ in dataloader:
        out = model(input_ids.to(device))
        caches = out["cache"]
        batch_loads = [cache[2].detach().float().cpu().squeeze(-1).sum(dim=0) for cache in caches]
        if layer_loads is None:
            layer_loads = [load.clone() for load in batch_loads]
        else:
            for dst, src in zip(layer_loads, batch_loads):
                dst += src
        batches += 1
        if batches >= max_batches:
            break
    model.train()
    return layer_loads, batches


def usage_rows(task: str, setting: int, layer_loads, batches: int, test_acc: float, train_loss: float, cfg, model):
    params = parameter_breakdown(model)["total"]
    mem = memory_budget_for_model("kvm", cfg, cfg.context_length)["floats"]
    rows = []
    for layer_idx, loads in enumerate(layer_loads):
        rows.extend(
            summarize_loads(
                task,
                setting,
                layer_idx,
                "all",
                loads.sum(dim=0),
                batches,
                test_acc,
                train_loss,
                params,
                mem,
                cfg.num_memory_tokens,
            )
        )
        for head_idx in range(loads.size(0)):
            rows.extend(
                summarize_loads(
                    task,
                    setting,
                    layer_idx,
                    str(head_idx),
                    loads[head_idx],
                    batches,
                    test_acc,
                    train_loss,
                    params,
                    mem,
                    cfg.num_memory_tokens,
                )
            )
    return rows


def summarize_loads(task, setting, layer_idx, head, loads, batches, test_acc, train_loss, params, mem, configured_slots):
    total = float(loads.sum().item())
    active = int((loads > 0).sum().item())
    if total <= 0:
        entropy = 0.0
        effective = 0.0
        top1_frac = 0.0
        top5_frac = 0.0
        max_load = 0.0
    else:
        probs = loads / total
        nz = probs[probs > 0]
        entropy = float(-(nz * nz.log()).sum().item())
        effective = float(torch.exp(torch.tensor(entropy)).item())
        sorted_loads = torch.sort(loads, descending=True).values
        top1_frac = float((sorted_loads[:1].sum() / total).item())
        top5_frac = float((sorted_loads[:5].sum() / total).item())
        max_load = float(sorted_loads[0].item())
    return [
        {
            "task": task,
            "setting": setting,
            "layer": layer_idx,
            "head": head,
            "batches": batches,
            "test_accuracy": test_acc,
            "train_loss": train_loss,
            "params": params,
            "memory_floats": mem,
            "configured_slots": configured_slots,
            "slots": int(loads.numel()),
            "active_slots": active,
            "entropy": entropy,
            "effective_slots": effective,
            "top1_frac": top1_frac,
            "top5_frac": top5_frac,
            "max_load": max_load,
            "mean_load": total / max(1, loads.numel()),
            "total_load": total,
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=parse_task,
        default=[TaskSpec("copy", 128, 0), TaskSpec("needle", 128, 0), TaskSpec("kv", 16, 0)],
        help="Task specs like copy:128 needle:1024 kv:16.",
    )
    parser.add_argument("--copy-steps", type=int, default=3000)
    parser.add_argument("--needle-steps", type=int, default=3000)
    parser.add_argument("--kv-steps", type=int, default=5000)
    parser.add_argument("--num-examples", type=int, default=30000)
    parser.add_argument("--test-examples", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--diagnostic-batches", type=int, default=8)
    parser.add_argument("--vocab-tokens", type=int, default=64)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--num-values", type=int, default=100)
    parser.add_argument("--num-keys", type=int, default=16)
    parser.add_argument("--value-mode", choices=["random", "identity", "shifted"], default="random")
    parser.add_argument("--answer-loss-weight", type=float, default=10.0)
    parser.add_argument("--kv-answer-loss-weight", type=float, default=20.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv-path", default="logs/kvm_slot_diagnostics.csv")
    args = parser.parse_args()

    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)

    rows = []
    for spec in args.tasks:
        seed = args.seed + spec.setting * 100
        torch.manual_seed(seed)
        cfg, train_ds, val_ds, test_ds, loss_fn, acc_fn, steps = build_task(spec, args)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        model = build_model("kvm", cfg).to(device)
        label = f"{spec.task}:{spec.setting}"
        train_loss, elapsed = train_model(model, train_loader, val_loader, acc_fn, loss_fn, steps, args, device, label)
        test_acc = acc_fn(model, test_loader, device)
        layer_loads, batches = collect_slot_usage(model, test_loader, device, args.diagnostic_batches)
        task_rows = usage_rows(spec.task, spec.setting, layer_loads, batches, test_acc, train_loss, cfg, model)
        for row in task_rows:
            row["training_time_sec"] = elapsed
        rows.extend(task_rows)

        Path(args.csv_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.csv_path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv_path}", flush=True)


if __name__ == "__main__":
    main()
