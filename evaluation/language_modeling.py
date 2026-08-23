"""Basic language-modeling metrics."""

from __future__ import annotations

import math
from typing import Iterable

import torch


@torch.no_grad()
def evaluate_loss(model, dataloader: Iterable, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        out = model(input_ids, targets=targets)
        total_loss += float(out["loss"].item())
        total_batches += 1
    model.train()
    return total_loss / max(1, total_batches)


def perplexity(loss: float) -> float:
    return math.exp(min(20.0, loss))

