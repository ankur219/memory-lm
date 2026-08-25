"""Transformer with recurrent summarized memory tokens.

Text is processed in chunks. Each chunk is prepended with a fixed number of
learned memory vectors carrying information from previous chunks. After the
chunk is processed, the output states at those memory positions become the
memory for the next chunk.
"""

from __future__ import annotations

import math
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
        
        if config.per_layer_memory:
            initial_memory = torch.zeros(config.num_layers, config.num_memory_tokens, self.recurrent_memory_dim)
        else:
            initial_memory = torch.zeros(config.num_memory_tokens, self.recurrent_memory_dim)
            
        if config.recurrent_learned_initial:
            self.initial_memory = nn.Parameter(initial_memory)
        else:
            self.register_buffer("initial_memory", initial_memory, persistent=True)

        # Low-rank rich-memory bridge and learned updater modules.
        if config.per_layer_memory:
            self.memory_down = nn.ModuleList([
                nn.Linear(self.recurrent_memory_dim, self.update_rank, bias=False)
                for _ in range(config.num_layers)
            ])
            self.memory_up = nn.ModuleList([
                nn.Linear(self.update_rank, config.hidden_size, bias=False)
                for _ in range(config.num_layers)
            ])
            if config.recurrent_update_style == "cross_attention":
                self.key_proj = nn.ModuleList([
                    nn.Linear(config.hidden_size, self.update_rank, bias=False)
                    for _ in range(config.num_layers)
                ])
                self.value_proj = nn.ModuleList([
                    nn.Linear(config.hidden_size, self.update_rank, bias=False)
                    for _ in range(config.num_layers)
                ])
            else:
                self.summary_down = nn.ModuleList([
                    nn.Linear(config.hidden_size, self.update_rank, bias=False)
                    for _ in range(config.num_layers)
                ])
            self.candidate_up = nn.ModuleList([
                nn.Linear(self.update_rank, self.recurrent_memory_dim, bias=False)
                for _ in range(config.num_layers)
            ])
            self.gate_up = nn.ModuleList([
                nn.Linear(self.update_rank, self.recurrent_memory_dim, bias=False)
                for _ in range(config.num_layers)
            ])
        else:
            self.memory_down = nn.Linear(self.recurrent_memory_dim, self.update_rank, bias=False)
            self.memory_up = nn.Linear(self.update_rank, config.hidden_size, bias=False)
            if config.recurrent_update_style == "cross_attention":
                self.key_proj = nn.Linear(config.hidden_size, self.update_rank, bias=False)
                self.value_proj = nn.Linear(config.hidden_size, self.update_rank, bias=False)
            else:
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

    def _update_memory_state(self, mem_state, token_out, layer_idx=None):
        # mem_state: [batch, num_memory_tokens, recurrent_memory_dim]
        # token_out: [batch, chunk_len, hidden_size]
        if self.config.per_layer_memory:
            assert layer_idx is not None
            mem_down = self.memory_down[layer_idx]
            candidate_up = self.candidate_up[layer_idx]
            gate_up = self.gate_up[layer_idx]
            if self.config.recurrent_update_style == "cross_attention":
                key_proj = self.key_proj[layer_idx]
                value_proj = self.value_proj[layer_idx]
            else:
                summary_down = self.summary_down[layer_idx]
        else:
            mem_down = self.memory_down
            candidate_up = self.candidate_up
            gate_up = self.gate_up
            if self.config.recurrent_update_style == "cross_attention":
                key_proj = self.key_proj
                value_proj = self.value_proj
            else:
                summary_down = self.summary_down

        if self.config.recurrent_update_style == "cross_attention":
            # Slot-specific attention
            q = mem_down(mem_state)  # [batch, num_memory_tokens, update_rank]
            k = key_proj(token_out)  # [batch, chunk_len, update_rank]
            v = value_proj(token_out)  # [batch, chunk_len, update_rank]
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.update_rank)  # [batch, num_memory_tokens, chunk_len]
            attn = F.softmax(scores, dim=-1)
            summary = torch.matmul(attn, v)  # [batch, num_memory_tokens, update_rank]
            
            candidate = torch.tanh(candidate_up(summary))  # [batch, num_memory_tokens, recurrent_memory_dim]
            gate = torch.sigmoid(gate_up(summary))  # [batch, num_memory_tokens, recurrent_memory_dim]
            memory_out = mem_state * (1.0 - gate) + candidate * gate
            write_entropy = -(attn.clamp_min(1e-9) * attn.clamp_min(1e-9).log()).sum(dim=-1).mean()
        elif self.config.recurrent_update_style == "last_tokens":
            # Ordered recent-token update. Instead of compressing the whole
            # chunk into one mean vector, each memory slot receives one of the
            # last num_memory_tokens token states. This tests whether recurrent
            # memory fails because the update destroys order/detail too early.
            n = self.config.num_memory_tokens
            if token_out.size(1) >= n:
                selected = token_out[:, -n:, :]
            else:
                pad = token_out.new_zeros(token_out.size(0), n - token_out.size(1), token_out.size(2))
                selected = torch.cat([pad, token_out], dim=1)
            summary = summary_down(selected)  # [batch, num_memory_tokens, update_rank]
            candidate = torch.tanh(candidate_up(summary))
            gate = torch.sigmoid(gate_up(summary))
            memory_out = mem_state * (1.0 - gate) + candidate * gate
            write_entropy = token_out.new_tensor(float("nan"))
        else:
            # Classic mean pooled update (original mean_gru)
            chunk_summary = token_out.mean(dim=1)  # [batch, hidden_size]
            summary = summary_down(chunk_summary)  # [batch, update_rank]
            candidate = torch.tanh(candidate_up(summary))[:, None, :]  # [batch, 1, recurrent_memory_dim]
            gate = torch.sigmoid(gate_up(summary))[:, None, :]  # [batch, 1, recurrent_memory_dim]
            memory_out = mem_state * (1.0 - gate) + candidate.expand_as(mem_state) * gate
            write_entropy = token_out.new_tensor(float("nan"))
            
        diagnostics = {
            "write_entropy": float(write_entropy.detach().cpu()),
            "memory_delta_norm": float((memory_out - mem_state).norm(dim=-1).mean().detach().cpu()),
            "memory_value_norm": float(memory_out.norm(dim=-1).mean().detach().cpu()),
        }
        return memory_out, diagnostics

    def _run_chunk(self, token_embeddings: torch.Tensor, memory: torch.Tensor):
        caches = []
        if self.config.per_layer_memory:
            x = token_embeddings
            token_outs_by_layer = []
            for l, layer in enumerate(self.layers):
                mem_l = memory[:, l]  # [batch, num_memory_tokens, recurrent_memory_dim]
                memory_prefix_l = self.memory_up[l](self.memory_down[l](mem_l))  # [batch, num_memory_tokens, hidden_size]
                x = torch.cat([memory_prefix_l, x], dim=1)
                x, cache = layer(x, self.rope_cos, self.rope_sin)
                
                # Slice out memory prefix output and text output
                x = x[:, self.config.num_memory_tokens:, :]
                token_outs_by_layer.append(x)
                caches.append(cache)
                
            token_out = x
            updated_memories = []
            diagnostic_rows = []
            for l in range(self.config.num_layers):
                updated_mem_l, diagnostics_l = self._update_memory_state(memory[:, l], token_outs_by_layer[l], layer_idx=l)
                updated_memories.append(updated_mem_l)
                diagnostic_rows.append(diagnostics_l)
            memory_out = torch.stack(updated_memories, dim=1)
            diagnostics = {
                key: sum(row[key] for row in diagnostic_rows) / len(diagnostic_rows)
                for key in diagnostic_rows[0]
            }
        else:
            memory_prefix = self.memory_up(self.memory_down(memory))
            x = torch.cat([memory_prefix, token_embeddings], dim=1)
            for layer in self.layers:
                x, cache = layer(x, self.rope_cos, self.rope_sin)
                caches.append(cache)
            token_out = x[:, self.config.num_memory_tokens:, :]
            memory_out, diagnostics = self._update_memory_state(memory, token_out)

        return token_out, memory_out, caches, diagnostics

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        batch, seq_len = input_ids.shape
        if self.config.per_layer_memory:
            memory = self.initial_memory.unsqueeze(0).expand(batch, -1, -1, -1)
        else:
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
                values = [row[key] for row in diagnostic_rows if row[key] == row[key]]
                diagnostics[key] = sum(values) / len(values) if values else ""
        return {"logits": logits, "loss": loss, "memory": memory, "cache": all_caches, "diagnostics": diagnostics}
