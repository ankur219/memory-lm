"""Transformer with recurrent summarized memory tokens.

Text is processed in chunks. Each chunk is prepended with a fixed number of
learned memory vectors carrying information from previous chunks. After the
chunk is processed, the output states at those memory positions become the
memory for the next chunk.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, TransformerBlock, TransformerConfig, precompute_rope_frequencies
from .per_token_memory import PerTokenBlock


class RecurrentMemoryTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.recurrent_memory_dim = config.recurrent_memory_dim or config.hidden_size
        self.update_rank = config.recurrent_update_rank
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        initial_memory = torch.zeros(config.num_memory_tokens, self.recurrent_memory_dim)
        if config.recurrent_learned_initial:
            self.initial_memory = nn.Parameter(initial_memory)
        else:
            self.register_buffer("initial_memory", initial_memory, persistent=True)

        # Low-rank rich-memory bridge. The persistent slots can be very wide
        # for memory-budget matching, but the learned updater stays small enough
        # to keep total parameters close to the many-small model.
        self.memory_down = nn.Linear(self.recurrent_memory_dim, self.update_rank, bias=False)
        self.memory_up = nn.Linear(self.update_rank, config.hidden_size, bias=False)
        self.summary_down = nn.Linear(config.hidden_size, self.update_rank, bias=False)
        self.candidate_up = nn.Linear(self.update_rank, self.recurrent_memory_dim, bias=False)
        self.gate_up = nn.Linear(self.update_rank, self.recurrent_memory_dim, bias=False)

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

        max_len = config.chunk_size + config.num_memory_tokens
        rope_dim = (config.memory_dim // config.num_heads) if config.recurrent_compressed_attention else config.head_dim
        cos, sin = precompute_rope_frequencies(rope_dim, max_len, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        if isinstance(self.initial_memory, nn.Parameter):
            nn.init.normal_(self.initial_memory, mean=0.0, std=0.02)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _run_chunk(self, token_embeddings: torch.Tensor, memory: torch.Tensor):
        """Run one chunk plus memory through all layers.

        Persistent memory is stored as [batch, memory_tokens, recurrent_memory_dim].
        Before attention, each rich memory slot is projected down to a normal
        hidden_size prefix state through a low-rank bridge and prepended to the
        chunk. Text tokens can read those prefix states plus previous tokens in
        the chunk.

        A subtle causal point: with a normal causal mask, the prepended memory
        positions cannot read the later text positions, so their Transformer
        outputs are not a valid chunk summary. Instead, the memory is read-only
        inside the chunk, then updated afterward from a pooled summary of the
        token outputs. This keeps token logits causal while still giving the
        recurrent state a path to absorb the just-processed chunk.

        We intentionally do not detach memory_out here. Future chunk losses must
        be able to train the memory update parameters. Detaching would make the
        recurrent updater nearly decorative.
        """

        memory_prefix = self.memory_up(self.memory_down(memory))
        x = torch.cat([memory_prefix, token_embeddings], dim=1)
        caches = []
        for layer in self.layers:
            x, cache = layer(x, self.rope_cos, self.rope_sin)
            caches.append(cache)
        token_out = x[:, self.config.num_memory_tokens :, :]
        chunk_summary = token_out.mean(dim=1)

        # Update rich persistent slots in recurrent_memory_dim space through the
        # same low-rank bottleneck. Parameter count is controlled by
        # recurrent_update_rank, while stored memory width remains fixed by the
        # memory-budget equation.
        summary = self.summary_down(chunk_summary)
        candidate = torch.tanh(self.candidate_up(summary))[:, None, :]
        gate = torch.sigmoid(self.gate_up(summary))[:, None, :]
        memory_out = memory * (1.0 - gate) + candidate.expand_as(memory) * gate
        return token_out, memory_out, caches

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        batch, seq_len = input_ids.shape
        memory = self.initial_memory.unsqueeze(0).expand(batch, -1, -1)
        outputs = []
        all_caches = []

        for start in range(0, seq_len, self.config.chunk_size):
            end = min(start + self.config.chunk_size, seq_len)
            token_embeddings = self.embed_tokens(input_ids[:, start:end])
            token_out, memory, caches = self._run_chunk(token_embeddings, memory)
            outputs.append(token_out)
            all_caches.append(caches)

        x = torch.cat(outputs, dim=1)
        logits = self.lm_head(self.norm(x))
        if self.param_padding is not None:
            logits = logits * (1.0 + self.param_padding.mean())
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss, "memory": memory, "cache": all_caches}
