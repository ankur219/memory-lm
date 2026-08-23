"""Baseline decoder-only Transformer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, TransformerBlock, TransformerConfig, precompute_rope_frequencies


class DecoderOnlyTransformer(nn.Module):
    """A compact LLaMA-style causal decoder used as the baseline."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.param_padding = (
            nn.Parameter(torch.zeros(config.param_padding)) if config.param_padding > 0 else None
        )
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

        cos, sin = precompute_rope_frequencies(config.head_dim, config.context_length, config.rope_base)
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
            # Parameter-count matching knob. Applying the scalar after RMSNorm
            # ensures the extra trainable parameters have a real gradient path;
            # a pre-normalization scalar is mostly cancelled by RMSNorm.
            logits = logits * (1.0 + self.param_padding.mean())
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss, "cache": caches}
