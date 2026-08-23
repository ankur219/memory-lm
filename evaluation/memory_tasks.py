"""Evaluation helpers for synthetic memory tasks."""

from __future__ import annotations

import torch


@torch.no_grad()
def token_accuracy(model, dataloader, device: torch.device, ignore_index: int = -100) -> float:
    model.eval()
    correct = 0
    total = 0
    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        logits = model(input_ids)["logits"]
        pred = logits.argmax(dim=-1)
        mask = targets != ignore_index
        correct += int((pred[mask] == targets[mask]).sum().item())
        total += int(mask.sum().item())
    model.train()
    return correct / max(1, total)

