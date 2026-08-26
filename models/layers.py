"""Shared Transformer building blocks.

The code in this file intentionally stays close to the equations. The project
is about comparing memory mechanisms, so the baseline layers should be easy to
audit rather than aggressively optimized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int = 128
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    context_length: int = 512
    mlp_ratio: float = 4.0
    rope_base: float = 10_000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    memory_dim: int = 128
    num_memory_tokens: int = 8
    recurrent_memory_dim: Optional[int] = None
    recurrent_update_rank: int = 4
    recurrent_compressed_attention: bool = True
    recurrent_learned_initial: bool = False
    per_layer_memory: bool = False
    recurrent_update_style: str = "mean_gru"
    assoc_memory_norm: bool = False
    assoc_memory_clip: Optional[float] = None
    param_padding: int = 0
    chunk_size: int = 128

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self.hidden_size // self.num_heads


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by LLaMA-style decoders."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


def precompute_rope_frequencies(
    head_dim: int,
    max_position: int,
    base: float = 10_000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cos/sin tables for rotary position embeddings."""

    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_position, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply RoPE to a tensor shaped [batch, heads, seq, head_dim]."""

    if positions is None:
        cos = cos[: x.size(-2)]
        sin = sin[: x.size(-2)]
    else:
        cos = cos.index_select(0, positions.reshape(-1)).view(*positions.shape, -1)
        sin = sin.index_select(0, positions.reshape(-1)).view(*positions.shape, -1)

    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    The gate and value projections both expand from hidden_size to mlp_hidden.
    Their elementwise product is projected back to hidden_size.
    """

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.gate_proj = nn.Linear(hidden_size, mlp_hidden, bias=False)
        self.up_proj = nn.Linear(hidden_size, mlp_hidden, bias=False)
        self.down_proj = nn.Linear(mlp_hidden, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(x))


def causal_mask(query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
    """Mask where True means the query is allowed to read the key.

    This version supports cached prefixes: when key_len > query_len, the extra
    keys are treated as positions before the current query block.
    """

    q_pos = torch.arange(query_len, device=device) + (key_len - query_len)
    k_pos = torch.arange(key_len, device=device)
    return k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention with full-width K/V storage."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        need_weights: bool = False,
    ):
        batch, seq_len, hidden = x.shape
        heads = self.config.num_heads
        head_dim = self.config.head_dim

        # Q, K, and V are all produced from the current hidden states. In the
        # baseline, keys and values are stored at full attention width:
        # [batch, heads, tokens, head_dim].
        q = self.q_proj(x).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = causal_mask(seq_len, k.size(2), x.device)
        scores = scores.masked_fill(~mask.view(1, 1, seq_len, k.size(2)), torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        y = self.out_proj(y)
        new_kv = (k.detach(), v.detach())
        if need_weights:
            return y, new_kv, attn
        return y, new_kv


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, attention_cls=CausalSelfAttention):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = attention_cls(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config.hidden_size, config.mlp_ratio, config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, **attn_kwargs):
        attn_out = self.attn(self.attn_norm(x), cos, sin, **attn_kwargs)
        if isinstance(attn_out, tuple) and len(attn_out) == 3:
            y, cache, weights = attn_out
            x = x + y
            x = x + self.mlp(self.mlp_norm(x))
            return x, cache, weights
        y, cache = attn_out
        x = x + y
        x = x + self.mlp(self.mlp_norm(x))
        return x, cache
