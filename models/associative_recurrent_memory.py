"""Recurrent Transformer with explicit associative memory reads and writes.

This variant tests a more structured few-rich memory than the naive recurrent
model. The per-sequence state is a set of memory values. Fixed learned read
keys and write queries are model parameters, not per-sequence memory state.

For each chunk:

1. Token states explicitly read from memory:
   token query x learned memory read keys -> slot weights -> memory values.
2. The retrieved memory vector is projected back into the token stream.
3. After local Transformer processing, learned write queries attend to token
   keys and produce candidate memory values.
4. A gated residual update writes those candidates into the recurrent values.

The important distinction from prefix-memory recurrent models is that the
read path directly queries memory keys and retrieves memory values instead of
turning memory into ordinary prefix tokens.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, TransformerConfig, precompute_rope_frequencies
from .per_token_memory import PerTokenBlock


def _attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Mean entropy over the last attention dimension."""

    probs = attn.clamp_min(1e-9)
    return -(probs * probs.log()).sum(dim=-1).mean()


class AssociativeRecurrentMemoryTransformer(nn.Module):
    """Few-rich recurrent memory with explicit key-value associative access."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.memory_dim = config.recurrent_memory_dim or config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # These are fixed learned addresses. They are parameters, not
        # per-sequence memory state, so they should be counted in Params rather
        # than persistent Mem Floats.
        self.memory_read_keys = nn.Parameter(torch.empty(config.num_memory_tokens, self.memory_dim))
        self.memory_write_queries = nn.Parameter(torch.empty(config.num_memory_tokens, self.memory_dim))

        initial_memory = torch.zeros(config.num_memory_tokens, self.memory_dim)
        if config.recurrent_learned_initial:
            self.initial_memory = nn.Parameter(initial_memory)
        else:
            self.register_buffer("initial_memory", initial_memory, persistent=True)

        # Explicit associative read projections.
        self.read_query = nn.Linear(config.hidden_size, self.memory_dim, bias=False)
        self.read_out = nn.Linear(self.memory_dim, config.hidden_size, bias=False)

        # Explicit associative write projections.
        self.write_key = nn.Linear(config.hidden_size, self.memory_dim, bias=False)
        self.write_value = nn.Linear(config.hidden_size, self.memory_dim, bias=False)
        self.write_gate = nn.Linear(self.memory_dim * 2, self.memory_dim, bias=False)

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
        cos, sin = precompute_rope_frequencies(mem_head, config.chunk_size, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        nn.init.normal_(self.memory_read_keys, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_write_queries, mean=0.0, std=0.02)
        if isinstance(self.initial_memory, nn.Parameter):
            nn.init.normal_(self.initial_memory, mean=0.0, std=0.02)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _read_memory(self, x: torch.Tensor, memory_values: torch.Tensor):
        # x: [batch, chunk_len, hidden_size]
        # memory_values: [batch, slots, memory_dim]
        q = self.read_query(x)
        scores = torch.matmul(q, self.memory_read_keys.t()) / math.sqrt(self.memory_dim)
        weights = F.softmax(scores, dim=-1)
        retrieved = torch.matmul(weights, memory_values)
        x = x + self.read_out(retrieved)
        return x, weights

    def _write_memory(self, token_out: torch.Tensor, memory_values: torch.Tensor):
        # Learned write queries are stable slot addresses. They decide which
        # token states each slot should summarize, while only memory_values are
        # carried as per-sequence recurrent state.
        batch = token_out.size(0)
        token_keys = self.write_key(token_out)
        token_values = self.write_value(token_out)
        write_queries = self.memory_write_queries.unsqueeze(0).expand(batch, -1, -1)
        scores = torch.matmul(write_queries, token_keys.transpose(-2, -1)) / math.sqrt(self.memory_dim)
        weights = F.softmax(scores, dim=-1)
        candidate = torch.matmul(weights, token_values)
        gate = torch.sigmoid(self.write_gate(torch.cat([memory_values, candidate], dim=-1)))
        new_memory = memory_values * (1.0 - gate) + candidate * gate
        return new_memory, weights

    def _run_chunk(self, token_embeddings: torch.Tensor, memory_values: torch.Tensor):
        x, read_weights = self._read_memory(token_embeddings, memory_values)
        caches = []
        for layer in self.layers:
            x, cache = layer(x, self.rope_cos, self.rope_sin)
            caches.append(cache)
        new_memory, write_weights = self._write_memory(x, memory_values)
        diagnostics = {
            "read_entropy": float(_attention_entropy(read_weights).detach().cpu()),
            "write_entropy": float(_attention_entropy(write_weights).detach().cpu()),
            "memory_delta_norm": float((new_memory - memory_values).norm(dim=-1).mean().detach().cpu()),
            "memory_value_norm": float(new_memory.norm(dim=-1).mean().detach().cpu()),
        }
        return x, new_memory, caches, diagnostics

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        batch, seq_len = input_ids.shape
        memory_values = self.initial_memory.unsqueeze(0).expand(batch, -1, -1)
        outputs = []
        all_caches = []
        diagnostic_rows = []

        for start in range(0, seq_len, self.config.chunk_size):
            end = min(start + self.config.chunk_size, seq_len)
            token_embeddings = self.embed_tokens(input_ids[:, start:end])
            token_out, memory_values, caches, diagnostics = self._run_chunk(token_embeddings, memory_values)
            outputs.append(token_out)
            all_caches.append(caches)
            diagnostic_rows.append(diagnostics)

        x = torch.cat(outputs, dim=1)
        logits = self.lm_head(self.norm(x))
        if self.param_padding is not None:
            logits = logits * (1.0 + self.param_padding.mean())
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        diagnostics = {}
        if diagnostic_rows:
            for key in diagnostic_rows[0]:
                diagnostics[key] = sum(row[key] for row in diagnostic_rows) / len(diagnostic_rows)
        return {
            "logits": logits,
            "loss": loss,
            "memory": memory_values,
            "cache": all_caches,
            "diagnostics": diagnostics,
        }
