# Many Small Memories or Few Rich Memories?

A matched-budget study of memory allocation in small language models.

## Abstract

Long-context language models can spend their inference-time memory budget in
different ways. A model can store many token-indexed states, as in a compressed
KV cache, or compress context into a smaller set of recurrent memory states. We
study this allocation choice under matched persistent-memory budgets. Across
TinyStories and WikiText-103 language modeling at 7M and 35M scale, compressed
per-token memory gives lower validation loss than a custom recurrent
few-rich-memory model with the same parameter count and persistent-memory
budget. On synthetic exact-recall tasks, per-token memory is the most reliable
strategy tested: it solves dense copy and needle retrieval across seeds, and it
beats an RMT-style memory-token baseline in direct synthetic comparisons. At the
same time, RMT-style memory tokens are much stronger than the custom recurrent
baseline on long-gap single-fact needle retrieval, showing that few-rich memory
is not uniformly weak and that the recurrence mechanism matters. These results
support a more specific claim: under the tested budgets, many-small memory is
the best default for exact recall and language modeling, while few-rich memory
can be useful for sparse salient retrieval when the update/read mechanism is
well matched to the task.

## 1. Introduction

Memory is one of the central bottlenecks in long-context language modeling. A
standard Transformer stores a key and value for every previous token in every
layer. This is accurate but expensive: the memory grows linearly with sequence
length, layer count, and hidden size. A common response is to compress memory.
However, compression is not one design choice. The same persistent-memory budget
can be allocated across many small token-indexed states or concentrated into a
fixed set of richer recurrent states.

This paper asks a deliberately narrow question:

```text
Given the same persistent inference-time memory budget, is it better to keep
many small token-indexed memories, or fewer rich recurrent memories?
```

We study this question in a small but controlled setting. The main comparison is
between compressed per-token KV memory and recurrent memory under matched
persistent-memory floats. We also include a baseline Transformer with full KV
memory, structured associative recurrent variants, and an RMT-style
memory-token baseline adapted from public RMT-style implementations.

The answer is not a simple "recurrent memory is bad." Per-token memory is the
strongest strategy tested for language modeling and exact synthetic recall, but
RMT-style memory tokens substantially improve long-gap single-fact retrieval
over the custom recurrent updater. The resulting picture is task-dependent:
dense exact recall favors many token-indexed states, while sparse salient recall
can benefit from recurrent memory tokens.

## 2. Memory-Budget Framing

We report two quantities separately:

- **Parameters:** learned weights shared across examples.
- **Persistent memory floats:** per-sequence inference-time state carried
  forward to represent prior context.

This distinction matters. Learned memory tokens or read/write projections are
parameters. KV states, compressed token memories, and recurrent states that must
be carried for each active sequence are persistent memory.

For a standard KV cache, the persistent memory is:

```text
sequence_length x num_layers x 2 x hidden_size
```

The factor of two counts keys and values. For compressed per-token memory, the
hidden state remains wide but each token stores compressed keys and values:

```text
sequence_length x num_layers x 2 x memory_dim
```

For recurrent memory, the persistent state is a fixed table of memory slots:

```text
num_memory_tokens x recurrent_memory_dim
```

These formulas let us compare models with different memory layouts at the same
persistent-memory budget.

## 3. Models

### Baseline Transformer

The baseline is a decoder-only Transformer with RMSNorm, RoPE, SwiGLU MLPs,
causal self-attention, and tied embeddings. It uses the full KV cache and
therefore has the largest persistent-memory budget in the main tables.

### Per-Token Compressed Memory

The per-token model keeps memory indexed by token position, but compresses the
key/value path. Each token stores a smaller key and value per layer. Attention
is performed in compressed memory space and projected back to the model hidden
size. This is the "many-small" strategy.

### Custom Recurrent Memory

The custom recurrent model processes text in chunks and carries a fixed set of
memory vectors between chunks. After each chunk, token outputs are summarized and
used to update the recurrent state through a gated update. This is the initial
"few-rich" strategy.

### Associative Recurrent Diagnostics

The associative variant separates fixed learned read keys from per-sequence
memory values and uses explicit read/write attention. It was designed to test
whether more structured addressing could rescue the recurrent-memory failure
mode. In the current experiments, it did not produce robust improvements after
seed checks.

### RMT-Style Memory Tokens

The RMT-style baseline wraps each chunk with memory tokens. Memory tokens are
used as prefix tokens for reading and suffix tokens for writing, with a causal
mask that prevents predicted text tokens from reading future write memory. This
published-style recurrent mechanism is much stronger than the custom recurrent
updater on long-gap needle retrieval, but it does not beat per-token memory in
direct synthetic comparisons.

## 4. Experimental Setup

### Real-Data Language Modeling

We train on TinyStories and WikiText-103 using a GPT-2 `tiktoken` tokenizer.
Each model sees one full pass over the training tokens. Validation is full at
the final evaluation. We report validation loss, perplexity, tokens/sec, peak
VRAM, parameter count, and persistent memory floats.

The repo contains both 7M-scale and 35M-scale runs. The 35M tables are the main
real-data evidence.

### Synthetic Recall

We use three synthetic tasks:

- **Copy:** repeat a sequence of token identities after a delimiter. This tests
  dense exact recall of many simultaneous token identities.
- **Needle:** remember one salient value across filler tokens. This tests
  sparse single-fact retrieval.
- **Random KV:** remember random key-value associations and answer a queried
  key. This tests multi-pair associative recall.

Synthetic RMT and per-token comparisons are run across three seeds for the
highest-impact cells.

## 5. Results

### 5.1 Real-Data LM

At 35M scale, per-token memory beats the custom recurrent model on both
TinyStories and WikiText-103 under the matched compressed-memory budget.

| Dataset | Model | Val Loss | Perplexity | Params | Mem Floats |
|---|---|---:|---:|---:|---:|
| TinyStories | Baseline | 1.5499 | 4.71 | 38,179,584 | 786,432 |
| TinyStories | Per-token | 1.5802 | 4.86 | 35,431,680 | 262,144 |
| TinyStories | Recurrent, `8x32768` | 1.7068 | 5.51 | 35,431,680 | 262,144 |
| TinyStories | Recurrent, `128x2048` | 1.7029 | 5.49 | 35,450,112 | 262,144 |
| WikiText-103 | Baseline | 3.6202 | 37.35 | 38,179,584 | 786,432 |
| WikiText-103 | Per-token | 3.6831 | 39.77 | 35,431,680 | 262,144 |
| WikiText-103 | Recurrent, `8x32768` | 3.7973 | 44.58 | 35,431,680 | 262,144 |
| WikiText-103 | Recurrent, `128x2048` | 3.7861 | 44.08 | 35,450,112 | 262,144 |

The recurrent shape check is important. The original 35M recurrent shape used 8
very wide slots. A fairer many-slot `128x2048` shape improves recurrent loss
slightly, but it does not close the gap to per-token memory. This reduces the
concern that the main recurrent result is only a bad slot-shape artifact.

### 5.2 Synthetic Exact Recall

Per-token memory is the strongest exact-recall model tested. In direct
comparison with RMT-style memory, it solves copy and needle across the tested
seeds, and it also wins random KV-16.

| Task | Model | Seed 0 | Seed 1 | Seed 2 | Mean | Std |
|---|---|---:|---:|---:|---:|---:|
| Copy-32 | Per-token | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Copy-32 | RMT-style | 0.112 | 0.963 | 0.964 | 0.680 | 0.491 |
| Copy-64 | Per-token | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Copy-64 | RMT-style | 0.031 | 0.030 | 0.031 | 0.031 | 0.000 |
| Needle-32 | Per-token | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Needle-32 | RMT-style | 0.938 | 0.677 | 0.491 | 0.702 | 0.225 |
| Needle-64 | Per-token | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Needle-64 | RMT-style | 0.903 | 0.966 | 0.763 | 0.877 | 0.104 |
| Random KV-16 | Per-token | 0.163 | 0.167 | 0.173 | 0.168 | 0.005 |
| Random KV-16 | RMT-style | 0.067 | 0.159 | 0.067 | 0.098 | 0.053 |

The KV-16 separation is stronger than an average-only comparison: RMT's best
seed is below per-token's worst seed.

### 5.3 RMT Changes The Recurrent Story

The RMT-style baseline is much stronger than the custom recurrent updater on
long-gap needle retrieval.

| Task | Model | Seed 0 | Seed 1 | Seed 2 | Mean | Std |
|---|---|---:|---:|---:|---:|---:|
| Needle-64 | Custom recurrent | 0.015 | 0.016 | 0.013 | 0.015 | 0.002 |
| Needle-64 | RMT-style | 0.903 | 0.966 | 0.763 | 0.877 | 0.104 |

This result prevents the broad claim that all few-rich recurrent memory is weak.
Instead, the result is more precise: naive recurrent summarization is weak for
these tasks, while RMT-style memory tokens can support sparse salient retrieval.
However, RMT still does not beat per-token memory in the direct comparison.

### 5.4 Associative Recurrent Memory

The associative branch tested whether explicit read/write addressing and write
normalization could rescue recurrent memory. The answer is currently no. Some
single-seed runs looked promising, but seed checks did not support those as
robust findings:

- Write-normalized associative needle-32 was highly seed-sensitive.
- Raw associative 2x-budget needle-64 was a seed-0 outlier.
- Associative memory did not solve copy-64 or random KV.

This branch remains useful as diagnostic evidence: simple structured addressing
is not enough to match many-small per-token memory under these budgets.

## 6. Discussion

The results suggest a capacity-allocation effect. Dense exact recall requires
preserving many token identities at once. Per-token memory allocates the budget
directly across token-indexed states, so the model has one storage location per
recent token. Few-rich memory must compress many items into fewer recurrent
slots. Changing the update rule, adding associative reads/writes, or using RMT
tokens can change which sparse retrieval problems are solvable, but these
mechanisms do not remove the dense-recall advantage of token-indexed storage in
the current experiments.

The RMT result is the most important nuance. It shows that the failure of the
custom recurrent model is not enough to condemn recurrent memory in general.
RMT-style memory tokens recover long-gap needle retrieval, which is precisely
the kind of sparse salient task recurrent memory should be able to support. The
direct per-token comparison then clarifies the boundary: RMT is a much stronger
few-rich baseline, but many-small per-token memory remains stronger overall for
exact synthetic recall.

## 7. Limitations

The current main scale is 35M parameters. This is large enough to move beyond a
toy pilot, but it is not a 100M+ or frontier-scale result. The real-data tables
are still primarily seed-0 and need more seeds before submission-grade claims.
RMT-style memory has been evaluated on synthetic tasks in this repo, not yet on
the 35M real-data LM runs. Longer context lengths, such as 512, 1024, and 2048
tokens, are also not yet covered.

The paper should therefore present its result as a careful matched-budget study,
not as a universal law of memory architectures.

## 8. Conclusion

Under matched persistent-memory budgets, many-small per-token memory is the best
default tested here for language modeling and exact recall. Few-rich recurrent
memory is not uniformly bad: RMT-style memory tokens are strong for sparse
long-gap single-fact retrieval. But in direct comparisons across copy, needle,
and random KV, per-token memory remains the most reliable exact-recall strategy.

