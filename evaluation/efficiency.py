"""Parameter and persistent-memory accounting utilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import torch.nn as nn

from models.layers import TransformerConfig


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """Group parameters into the buckets used in experiment logs."""

    buckets = {"embedding": 0, "attention": 0, "mlp": 0, "norm": 0, "other": 0}
    seen_tied = set()
    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        if ptr in seen_tied:
            continue
        seen_tied.add(ptr)
        n = param.numel()
        if "embed_tokens" in name or "lm_head" in name:
            buckets["embedding"] += n
        elif ".attn." in name:
            buckets["attention"] += n
        elif ".mlp." in name:
            buckets["mlp"] += n
        elif "norm" in name:
            buckets["norm"] += n
        else:
            buckets["other"] += n
    buckets["total"] = sum(buckets.values())
    return buckets


def floats_to_size(num_floats: int, bytes_per_float: int = 2) -> Dict[str, float]:
    """Report memory assuming fp16/bf16 by default."""

    bytes_total = num_floats * bytes_per_float
    return {
        "floats": int(num_floats),
        "bytes_per_float": bytes_per_float,
        "kb": bytes_total / 1024,
        "mb": bytes_total / (1024**2),
    }


def baseline_kv_memory_budget(config: TransformerConfig, sequence_length: int, bytes_per_float: int = 2):
    """Full Transformer KV cache: tokens x layers x 2(K,V) x hidden_size."""

    floats = sequence_length * config.num_layers * 2 * config.hidden_size
    out = floats_to_size(floats, bytes_per_float)
    out.update({"kind": "baseline_kv", "sequence_length": sequence_length, "config": asdict(config)})
    return out


def per_token_memory_budget(config: TransformerConfig, sequence_length: int, bytes_per_float: int = 2):
    """Per-token compressed memory: tokens x layers x 2(K,V) x memory_dim."""

    floats = sequence_length * config.num_layers * 2 * config.memory_dim
    out = floats_to_size(floats, bytes_per_float)
    out.update({"kind": "per_token", "sequence_length": sequence_length, "config": asdict(config)})
    return out


def recurrent_memory_budget(config: TransformerConfig, bytes_per_float: int = 2, per_layer_memory: bool = False):
    """Recurrent memory: memory_tokens x memory_dim x relevant_layers.

    In this implementation memory is stored once between chunks at hidden_size,
    not once per layer. The optional per_layer_memory flag is included because
    later RMT variants may keep separate layerwise memories.
    """

    layers = config.num_layers if per_layer_memory else 1
    recurrent_dim = config.recurrent_memory_dim or config.hidden_size
    floats = config.num_memory_tokens * recurrent_dim * layers
    out = floats_to_size(floats, bytes_per_float)
    out.update(
        {
            "kind": "recurrent",
            "num_memory_tokens": config.num_memory_tokens,
            "recurrent_memory_dim": recurrent_dim,
            "per_layer_memory": per_layer_memory,
            "config": asdict(config),
        }
    )
    return out


def kvm_memory_budget(config: TransformerConfig, bytes_per_float: int = 2):
    """KVM compressed state: layers x slots x 2(K,V) x state_dim."""

    state_dim = config.recurrent_memory_dim or config.memory_dim
    floats = config.num_layers * config.num_memory_tokens * 2 * state_dim
    out = floats_to_size(floats, bytes_per_float)
    out.update(
        {
            "kind": "kvm",
            "num_memory_tokens": config.num_memory_tokens,
            "recurrent_memory_dim": state_dim,
            "per_layer_memory": True,
            "config": asdict(config),
        }
    )
    return out


def matched_recurrent_tokens_for_per_token(
    config: TransformerConfig,
    sequence_length: int,
    per_layer_memory: bool = False,
) -> int:
    """Choose recurrent memory tokens with approximately equal float count."""

    per_token_floats = sequence_length * config.num_layers * 2 * config.memory_dim
    layers = config.num_layers if per_layer_memory else 1
    recurrent_dim = config.recurrent_memory_dim or config.hidden_size
    denom = recurrent_dim * layers
    return max(1, round(per_token_floats / denom))


def matched_recurrent_dim_for_per_token(
    config: TransformerConfig,
    sequence_length: int,
    num_memory_tokens: int,
    per_layer_memory: bool = False,
) -> int:
    """Choose rich recurrent slot width for a fixed number of memory tokens."""

    per_token_floats = sequence_length * config.num_layers * 2 * config.memory_dim
    layers = config.num_layers if per_layer_memory else 1
    denom = num_memory_tokens * layers
    return max(1, round(per_token_floats / denom))


def matched_kvm_tokens_for_per_token(config: TransformerConfig, sequence_length: int) -> int:
    """Choose KVM state slots with equal float count to per-token compressed KV."""

    per_token_floats = sequence_length * config.num_layers * 2 * config.memory_dim
    state_dim = config.recurrent_memory_dim or config.memory_dim
    denom = config.num_layers * 2 * state_dim
    return max(1, round(per_token_floats / denom))


def print_parameter_breakdown(model: nn.Module) -> None:
    breakdown = parameter_breakdown(model)
    print(f"total parameters:     {breakdown['total']:,}")
    print(f"attention parameters: {breakdown['attention']:,}")
    print(f"MLP parameters:       {breakdown['mlp']:,}")
    print(f"embedding parameters: {breakdown['embedding']:,}")
