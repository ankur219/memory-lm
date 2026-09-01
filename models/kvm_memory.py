"""Key-Value Means style compressed recurrent KV baseline.

This module adapts the eager KVM mechanism from:

    https://github.com/recursal/KVM-paper

The public implementation is a full RWKV/GPT training stack with optional
Triton kernels. For this repo we keep the core mechanism and plug it into the
same small Transformer backbone used by the other controlled baselines:

* each attention layer keeps a fixed-size compressed K/V state
* each chunk attends to compressed state plus a short block-sliding window
* old window tokens overflow into the compressed state
* novel overflow tokens append while space remains; the rest merge into the
  most similar existing key slot

The state is layerwise, so memory accounting counts keys and values for every
layer: layers x slots x 2(K,V) x state_dim.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, SwiGLU, TransformerConfig, apply_rope, precompute_rope_frequencies


def _causal_state_window_mask(
    q_positions: torch.Tensor,
    window_positions: torch.Tensor,
    state_len: int,
) -> torch.Tensor:
    state_allowed = torch.ones(
        q_positions.numel(),
        state_len,
        device=q_positions.device,
        dtype=torch.bool,
    )
    window_allowed = window_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
    return torch.cat([state_allowed, window_allowed], dim=1)


class KVMAttention(nn.Module):
    """Compressed attention with KVM-style fixed-state merging."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.state_dim = config.recurrent_memory_dim or config.memory_dim
        if self.state_dim % config.num_heads != 0:
            raise ValueError("KVM state dimension must be divisible by num_heads")
        self.state_head_dim = self.state_dim // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, self.state_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.state_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.state_dim, bias=False)
        self.out_proj = nn.Linear(self.state_dim, config.hidden_size, bias=False)
        self.state_key_norm = RMSNorm(self.state_head_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.merge_gate = (
            nn.Linear(config.hidden_size, config.num_heads, bias=False)
            if config.kvm_use_merge_gate
            else None
        )

    def _project_qkv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        batch, seq_len, _ = x.shape
        heads = self.config.num_heads
        head_dim = self.state_head_dim
        q = self.q_proj(x).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.merge_gate is None:
            gate = torch.ones(batch, heads, seq_len, 1, device=x.device, dtype=x.dtype)
        else:
            gate = 1.0 + F.elu(self.merge_gate(x).transpose(1, 2).unsqueeze(-1))
        return q, k, v, gate

    def _read_state_and_window(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        state_vlen: torch.Tensor,
        start: int,
        end: int,
        window_begin: int,
    ) -> torch.Tensor:
        q_chunk = q[:, :, start:end]
        k_window = k[:, :, window_begin:end]
        v_window = v[:, :, window_begin:end]

        if state_k.size(2) > 0:
            k_state = self.state_key_norm(state_k)
            v_state = state_v / state_vlen.clamp_min(1.0)
            k_all = torch.cat([k_state, k_window], dim=2)
            v_all = torch.cat([v_state, v_window], dim=2)
            state_len = state_k.size(2)
        else:
            k_all = k_window
            v_all = v_window
            state_len = 0

        q_pos = torch.arange(start, end, device=q.device)
        w_pos = torch.arange(window_begin, end, device=q.device)
        mask = _causal_state_window_mask(q_pos, w_pos, state_len)
        scores = torch.matmul(q_chunk, k_all.transpose(-2, -1)) / math.sqrt(self.state_head_dim)
        scores = scores.masked_fill(~mask.view(1, 1, end - start, k_all.size(2)), torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v_all)

    def _append_to_state(self, state_k, state_v, state_vlen, k_new, v_new):
        if k_new.size(2) == 0:
            return state_k, state_v, state_vlen
        vlen_new = torch.ones_like(v_new[..., :1])
        return (
            torch.cat([state_k, k_new], dim=2),
            torch.cat([state_v, v_new], dim=2),
            torch.cat([state_vlen, vlen_new], dim=2),
        )

    def _merge_into_state(self, state_k, state_v, state_vlen, k_merge, v_merge):
        if k_merge.size(2) == 0:
            return state_k, state_v, state_vlen
        if state_k.size(2) == 0:
            keep = min(self.config.num_memory_tokens, k_merge.size(2))
            return self._append_to_state(state_k, state_v, state_vlen, k_merge[:, :, :keep], v_merge[:, :, :keep])

        state_norm = self.state_key_norm(state_k)
        incoming_norm = self.state_key_norm(k_merge)
        logits = torch.matmul(incoming_norm, state_norm.transpose(-1, -2))
        best = logits.argmax(dim=-1)
        scores = F.one_hot(best, num_classes=state_k.size(2)).to(k_merge.dtype)
        state_k = state_k + torch.matmul(scores.transpose(-1, -2), k_merge)
        state_v = state_v + torch.matmul(scores.transpose(-1, -2), v_merge)
        state_vlen = state_vlen + scores.sum(dim=-2, keepdim=True).transpose(-1, -2)
        return state_k, state_v, state_vlen

    def _update_state(self, state_k, state_v, state_vlen, overflow_k, overflow_v, overflow_gate):
        if overflow_k.size(2) == 0:
            return state_k, state_v, state_vlen

        overflow_k = self.state_key_norm(overflow_k) * overflow_gate
        overflow_v = overflow_v * overflow_gate
        free = max(self.config.num_memory_tokens - state_k.size(2), 0)
        if free > 0:
            n_append = min(free, overflow_k.size(2))
            state_k, state_v, state_vlen = self._append_to_state(
                state_k,
                state_v,
                state_vlen,
                overflow_k[:, :, :n_append],
                overflow_v[:, :, :n_append],
            )
            overflow_k = overflow_k[:, :, n_append:]
            overflow_v = overflow_v[:, :, n_append:]

        return self._merge_into_state(state_k, state_v, state_vlen, overflow_k, overflow_v)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        batch, seq_len, _ = x.shape
        heads = self.config.num_heads
        q, k, v, gate = self._project_qkv(x, cos, sin)
        state_k = k.new_zeros(batch, heads, 0, self.state_head_dim)
        state_v = v.new_zeros(batch, heads, 0, self.state_head_dim)
        state_vlen = v.new_zeros(batch, heads, 0, 1)

        outs = []
        state_coverage = 0
        chunk = self.config.chunk_size
        window_len = max(1, self.config.kvm_window_chunks) * chunk
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            window_begin = max(0, start - window_len)
            if window_begin > state_coverage:
                state_k, state_v, state_vlen = self._update_state(
                    state_k,
                    state_v,
                    state_vlen,
                    k[:, :, state_coverage:window_begin],
                    v[:, :, state_coverage:window_begin],
                    gate[:, :, state_coverage:window_begin],
                )
                state_coverage = window_begin
            outs.append(
                self._read_state_and_window(
                    q, k, v, state_k, state_v, state_vlen, start, end, window_begin
                )
            )

        y = torch.cat(outs, dim=2).transpose(1, 2).contiguous().view(batch, seq_len, self.state_dim)
        y = self.out_proj(y)
        diagnostics = {
            "memory_delta_norm": float(state_k.norm(dim=-1).mean().detach().cpu()) if state_k.size(2) else 0.0,
            "memory_value_norm": float((state_v / state_vlen.clamp_min(1.0)).norm(dim=-1).mean().detach().cpu())
            if state_v.size(2)
            else 0.0,
            "kvm_state_tokens": float(state_k.size(2)),
        }
        cache = (state_k.detach(), state_v.detach(), state_vlen.detach())
        return y, cache, diagnostics


class KVMBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = KVMAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config.hidden_size, config.mlp_ratio, config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        y, cache, diagnostics = self.attn(self.attn_norm(x), cos, sin)
        x = x + y
        x = x + self.mlp(self.mlp_norm(x))
        return x, cache, diagnostics


class KVMMemoryTransformer(nn.Module):
    """KVM-style Transformer using compressed fixed-state K/V attention."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.state_dim = config.recurrent_memory_dim or config.memory_dim
        if self.state_dim % config.num_heads != 0:
            raise ValueError("KVM state dimension must be divisible by num_heads")
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([KVMBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.param_padding = (
            nn.Parameter(torch.zeros(config.param_padding)) if config.param_padding > 0 else None
        )
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

        head_dim = self.state_dim // config.num_heads
        cos, sin = precompute_rope_frequencies(head_dim, config.context_length, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.embed_tokens(input_ids)
        caches = []
        diagnostic_rows = []
        for layer in self.layers:
            x, cache, diagnostics = layer(x, self.rope_cos, self.rope_sin)
            caches.append(cache)
            diagnostic_rows.append(diagnostics)

        logits = self.lm_head(self.norm(x))
        if self.param_padding is not None:
            logits = logits * (1.0 + self.param_padding.mean())
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        diagnostics = {}
        if diagnostic_rows:
            for key in diagnostic_rows[0]:
                vals = [row[key] for row in diagnostic_rows]
                diagnostics[key] = sum(vals) / len(vals)
        return {"logits": logits, "loss": loss, "cache": caches, "diagnostics": diagnostics}
