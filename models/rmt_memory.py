"""RMT-style recurrent memory baseline.

This is a compact reproduction of the core Recurrent Memory Transformer idea:
carry a fixed set of hidden-size memory tokens from chunk to chunk and let the
Transformer itself update them.

For causal language modeling we use two memory blocks per chunk:

1. prefix memory tokens: previous recurrent state, visible to current text
2. suffix write tokens: hidden-size memory tokens placed after the text

The suffix tokens can attend to prefix memory and all current text, so their
outputs become the next recurrent state. Text logits are taken only from the
text positions, so no predicted token can read future text or suffix memory.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, TransformerBlock, TransformerConfig, precompute_rope_frequencies
from .per_token_memory import PerTokenBlock


class RMTMemoryTransformer(nn.Module):
    """Segment-recurrent memory-token baseline inspired by RMT."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.memory_dim = config.recurrent_memory_dim or config.hidden_size
        if self.memory_dim != config.hidden_size:
            raise ValueError(
                "RMTMemoryTransformer stores hidden-size memory tokens; "
                "set recurrent_memory_dim to hidden_size or leave it unset."
            )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        initial_memory = torch.zeros(config.num_memory_tokens, config.hidden_size)
        if config.recurrent_learned_initial:
            self.initial_memory = nn.Parameter(initial_memory)
        else:
            self.register_buffer("initial_memory", initial_memory, persistent=True)

        # Learned write-token bias. The current memory is added to this before
        # placing suffix write tokens after the text, giving the update a stable
        # slot identity while still conditioning on the previous memory value.
        self.write_memory_bias = nn.Parameter(torch.zeros(config.num_memory_tokens, config.hidden_size))

        block_cls = PerTokenBlock if config.recurrent_compressed_attention else TransformerBlock
        self.layers = nn.ModuleList([block_cls(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.param_padding = (
            nn.Parameter(torch.zeros(config.param_padding)) if config.param_padding > 0 else None
        )
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

        max_len = config.chunk_size + 2 * config.num_memory_tokens
        rope_dim = (config.memory_dim // config.num_heads) if config.recurrent_compressed_attention else config.head_dim
        cos, sin = precompute_rope_frequencies(rope_dim, max_len, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        if isinstance(self.initial_memory, nn.Parameter):
            nn.init.normal_(self.initial_memory, mean=0.0, std=0.02)
        nn.init.normal_(self.write_memory_bias, mean=0.0, std=0.02)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _run_chunk(self, token_embeddings: torch.Tensor, memory: torch.Tensor):
        batch = token_embeddings.size(0)
        prefix_memory = memory
        suffix_memory = memory + self.write_memory_bias.unsqueeze(0).expand(batch, -1, -1)

        # Sequence layout:
        # [previous memory][current text][write memory]
        # Causal attention gives current text access to previous memory, while
        # write memory can see both previous memory and current text.
        x = torch.cat([prefix_memory, token_embeddings, suffix_memory], dim=1)
        caches = []
        for layer in self.layers:
            x, cache = layer(x, self.rope_cos, self.rope_sin)
            caches.append(cache)

        n_mem = self.config.num_memory_tokens
        text_len = token_embeddings.size(1)
        token_out = x[:, n_mem : n_mem + text_len, :]
        memory_out = x[:, -n_mem:, :]
        diagnostics = {
            "memory_delta_norm": float((memory_out - memory).norm(dim=-1).mean().detach().cpu()),
            "memory_value_norm": float(memory_out.norm(dim=-1).mean().detach().cpu()),
        }
        return token_out, memory_out, caches, diagnostics

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        batch, seq_len = input_ids.shape
        memory = self.initial_memory.unsqueeze(0).expand(batch, -1, -1)
        outputs = []
        all_caches = []
        diagnostic_rows = []

        for start in range(0, seq_len, self.config.chunk_size):
            end = min(start + self.config.chunk_size, seq_len)
            token_embeddings = self.embed_tokens(input_ids[:, start:end])
            token_out, memory, caches, diagnostics = self._run_chunk(token_embeddings, memory)
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
                values = [row[key] for row in diagnostic_rows]
                diagnostics[key] = sum(values) / len(values)
        return {"logits": logits, "loss": loss, "memory": memory, "cache": all_caches, "diagnostics": diagnostics}
