"""Transformer with learned per-token compressed K/V memory.

This is not a post-training cache compression wrapper. The attention layer is
trained from scratch to write each token into a low-dimensional K/V memory and
then read from that low-dimensional memory during attention.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    RMSNorm,
    SwiGLU,
    TransformerConfig,
    apply_rope,
    causal_mask,
    precompute_rope_frequencies,
)


class CompressedPerTokenAttention(nn.Module):
    """Causal attention where persistent keys/values are memory_dim wide.

    Hidden states stay at hidden_size. Queries are projected from hidden_size
    into memory_dim so the dot product is performed against compressed keys.
    Values are also memory_dim wide while stored, then expanded back to
    hidden_size after attention.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        if config.memory_dim % config.num_heads != 0:
            raise ValueError("memory_dim must be divisible by num_heads")
        self.config = config
        self.memory_dim = config.memory_dim
        self.memory_head_dim = config.memory_dim // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.memory_dim, bias=False)
        self.k_mem_proj = nn.Linear(config.hidden_size, config.memory_dim, bias=False)
        self.v_mem_proj = nn.Linear(config.hidden_size, config.memory_dim, bias=False)
        self.out_proj = nn.Linear(config.memory_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        need_weights: bool = False,
    ):
        batch, seq_len, _ = x.shape
        heads = self.config.num_heads
        mem_head = self.memory_head_dim

        # The model computes a narrow query, key, and value for attention.
        # These K/V tensors are the persistent per-token memory:
        # [batch, heads, tokens, memory_dim / heads].
        q = self.q_proj(x).view(batch, seq_len, heads, mem_head).transpose(1, 2)
        k = self.k_mem_proj(x).view(batch, seq_len, heads, mem_head).transpose(1, 2)
        v = self.v_mem_proj(x).view(batch, seq_len, heads, mem_head).transpose(1, 2)

        # RoPE is applied in compressed head space. When memory_dim is smaller,
        # each head has fewer rotary dimensions to represent relative position.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(mem_head)
        mask = causal_mask(seq_len, k.size(2), x.device)
        scores = scores.masked_fill(~mask.view(1, 1, seq_len, k.size(2)), torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.memory_dim)
        y = self.out_proj(y)
        new_kv = (k.detach(), v.detach())
        if need_weights:
            return y, new_kv, attn
        return y, new_kv


class PerTokenBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = CompressedPerTokenAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config.hidden_size, config.mlp_ratio, config.dropout)

    def forward(self, x, cos, sin, **attn_kwargs):
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


class PerTokenMemoryTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([PerTokenBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.param_padding = (
            nn.Parameter(torch.zeros(config.param_padding)) if config.param_padding > 0 else None
        )
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

        mem_head = config.memory_dim // config.num_heads
        cos, sin = precompute_rope_frequencies(mem_head, config.context_length, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.embed_tokens(input_ids)
        caches = []
        for layer in self.layers:
            x, cache = layer(x, self.rope_cos, self.rope_sin)
            caches.append(cache)
        logits = self.lm_head(self.norm(x))
        if self.param_padding is not None:
            # Parameter-count matching knob. It participates as a tiny global
            # logit scale so the matched parameters are trainable, while the
            # memory mechanism itself remains unchanged. This is intentionally
            # after RMSNorm; before RMSNorm the scalar would mostly cancel out.
            logits = logits * (1.0 + self.param_padding.mean())
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss, "cache": caches}
