# Memory-LM

Memory-LM is a small PyTorch research codebase for comparing two different ways a language model can remember previous text under a matched persistent-memory budget.

The research question is:

```text
Given the same total inference-time memory/storage budget, is it better to keep
small per-token memories, or a fixed set of recurrent summary memories?
```

The first milestone is intentionally modest. It proves that three model families can train under one clean framework:

- `baseline`: a standard decoder-only Transformer.
- `per_token`: a Transformer trained from scratch with compressed per-token K/V memory.
- `recurrent`: a Transformer that carries a fixed number of recurrent memory vectors across chunks.

This is not large-scale pretraining yet.

See `RESULTS.md` for the current TinyStories, WikiText, and synthetic memory
checkpoint. Paper drafting artifacts live under `paper/`.

## Current TODOs

### Active 1-6 Plan

1. **Paper outline.**
   First draft started around the central question: under a fixed
   persistent-memory budget, should capacity be allocated across many
   token-specific states or fewer compressed recurrent states? See
   `paper/DRAFT.md` and `paper/OUTLINE.md`.
2. **Core figures and tables.**
   Initial real-data, RMT-vs-per-token, and memory-layout figures are generated
   under `paper/figures/`. The memory-budget curve is still planned. See
   `paper/FIGURES.md`.
3. **Related work.**
   Position against Transformer-XL, Compressive Transformer, RMT, ARMT,
   TransformerFAM, Memorizing Transformers, Infini-Transformer, Key-Value Means,
   Melodi, and related recurrent-memory work. See `paper/RELATED_WORK.md`.
4. **RMT real-data scope decision.**
   Current plan: keep RMT as a synthetic published-baseline probe unless the
   paper needs a stronger external real-data baseline. Scaling RMT to 35M
   real-data runs is optional and compute-expensive.
5. **Multi-seed 35M real-data checks.**
   Run second/third seeds for the final central 35M TinyStories and WikiText
   tables before making strong submission claims.
6. **Memory-budget curve.**
   Build the main scaling figure over persistent-memory budgets, for example
   1x, 1/2x, 1/4x, and 1/8x of full KV memory. See
   `paper/EXPERIMENT_PLAN.md`.

### Parked Or Closed

- 100M-150M scale point: parked for now.
- Fast-weight or outer-product memory: deliberately not planned for now.
- Associative recurrent branch: treated as a negative/diagnostic result unless
  later evidence reopens it.
- RMT synthetic comparison: complete enough for the current paper story; see
  `RESULTS.md`.

Regenerate current paper figures with:

```bash
python3 paper/make_figures.py
```

### Venue Direction

- TMLR is a strong fit if the final paper emphasizes soundness, careful matched
  budgets, and negative results at the current 35M scale.
- ICLR/COLM 2027 workshops remain good lower-friction targets for an
  LLM-memory-focused audience.
- Main-track ICLR/NeurIPS is a stretch unless the project adds an external
  baseline such as RMT or ARMT reproduced inside this harness, longer-context
  evaluations, memory-budget curves, and likely a 100M-150M scale point.

## Repository Layout

```text
configs/                 YAML configs for tiny experiments
models/layers.py         RMSNorm, RoPE, SwiGLU, attention, Transformer block
models/transformer.py    Standard decoder-only baseline
models/per_token_memory.py
models/recurrent_memory.py
data/synthetic.py        Key-value retrieval synthetic dataset
training/trainer.py      Small training loop and JSON/CSV logging
evaluation/efficiency.py Parameter and memory-budget accounting
tests/                   Shape, accounting, and causal-behavior tests
train.py                 CLI entrypoint
```

## Model Architectures

### Baseline Transformer

The baseline is a decoder-only causal Transformer using:

- RMSNorm
- RoPE
- SwiGLU MLPs
- causal self-attention
- tied input/output embeddings by default

Standard K/V cache storage is counted as:

```text
sequence_length x num_layers x 2 x hidden_size
```

The factor of `2` is for keys and values.

### PerTokenMemoryTransformer

The main hidden representation stays wide, for example `hidden_size = 512`, but each token writes compressed keys and values:

```text
hidden state: 512d
query/key/value memory path: memory_dim
```

For `memory_dim = 128`, each token stores a 128d key and a 128d value per layer. Attention dot products happen in compressed memory space, and the attended value is projected back to `hidden_size`.

Persistent memory is counted as:

```text
sequence_length x num_layers x 2 x memory_dim
```

This is trained from scratch with compressed K/V memory. It is not post-training KV-cache compression.

### RecurrentMemoryTransformer

The recurrent model processes text in chunks. Each chunk receives a fixed set of memory vectors as prefix states:

```text
memory + chunk -> Transformer -> token outputs -> update memory
```

For the first implementation, prefix memory is read-only inside the chunk. After the chunk is processed, the model pools token outputs and updates each memory vector with a low-rank gated update. This keeps token logits causal while giving the recurrent state a path to summarize the chunk.

Persistent memory is counted as:

```text
num_memory_tokens x hidden_size
```

The accounting utility also supports a `per_layer_memory=True` option for later variants that store separate memory per layer:

```text
num_memory_tokens x hidden_size x num_layers
```

## Parameter Counting

`evaluation/efficiency.py` reports:

```text
total parameters
attention parameters
MLP parameters
embedding parameters
```

It avoids double-counting tied embeddings.

## Tiny Synthetic Experiment

The implemented synthetic dataset is key-value retrieval:

```text
<BOS> key value key value ... <QUERY> key <ANSWER> value <EOS>
```

The training loss is applied only to the answer token. This asks the model to store exact associations and retrieve one requested value.

Run tests:

```bash
cd memory-lm
python3 -m pytest
```

Run one tiny overfit experiment:

```bash
cd memory-lm
python3 train.py --config configs/per_token.yaml
```

Run one smoke comparison:

```bash
cd memory-lm
python3 experiments/run_memory_comparison.py
```

Logs are written as JSONL and CSV under `logs/`.

## Real-Text Language Modeling

There is also an end-to-end real-data path using TinyStories through Hugging
Face `datasets`, with a shared `tiktoken` tokenizer across all models. The
default encoding is `gpt2`, which keeps the vocabulary smaller than `cl100k_base`
and makes the 30M-50M parameter target easier to reason about.

Run one real-text model:

```bash
cd memory-lm
python3 train_real.py --config configs/real_per_token.yaml
```

Run the real-text comparison:

```bash
cd memory-lm
python3 experiments/run_real_comparison.py
```

Before launching a comparison, inspect the exact token/step/memory budgets:

```bash
cd memory-lm
python3 experiments/count_real_tokens.py
```

For an offline debug run that uses a tiny local text file:

```bash
cd memory-lm
python3 train_real.py --config configs/real_local_debug.yaml
```

The real-text configs train for one full pass over the TinyStories training
split and evaluate on the official TinyStories validation split. Token ids are
cached to disk as `uint16` memmap files so full-dataset runs do not keep the
entire token stream as a Python list in RAM:

```yaml
tokenizer:
  kind: tiktoken
  encoding: gpt2
  cache_dir: data/tiktoken_cache
data:
  source: tinystories
  split: train
  cache_dir: data/hf_cache
  cache_tokens: true
  token_cache_dir: data/token_cache
  max_examples:
  max_chars:
validation_data:
  source: tinystories
  split: validation
  cache_dir: data/hf_cache
  token_cache_dir: data/token_cache
  max_examples:
  max_chars:
```

The default real comparison uses `batch_size: 128` and is epoch-budgeted:

```text
planned_steps = ceil(full_train_blocks / batch_size)
planned_train_tokens ~= full TinyStories GPT-2-tokenized train split
```

Validation loss is computed every 500 steps on a capped validation subset
(`eval_max_batches: 100`) to keep training moving. The final validation pass is
full because `final_eval_max_batches` is unset. The CSV logs include
`validation_batches` and `validation_full` so capped and full validation rows
are explicit.

Qualitative samples are printed every 1000 steps. Each sample prints the fed
prompt separately from the predicted continuation.

## WikiText-103 Comparison

After TinyStories, run the same matched comparison on WikiText-103. This checks
whether the result is specific to simple story text or also appears on a more
general language-modeling corpus.

First inspect token counts, parameter counts, and memory budgets:

```bash
cd memory-lm
python3 experiments/count_real_tokens.py \
  configs/wikitext_baseline.yaml \
  configs/wikitext_per_token.yaml \
  configs/wikitext_recurrent.yaml
```

Then launch the full comparison:

```bash
cd memory-lm
nohup python3 experiments/run_wikitext_comparison.py > wikitext.out 2>&1 &
tail -f wikitext.out
```

Summarize the final logs:

```bash
cd memory-lm
python3 experiments/summarize_wikitext_comparison.py
```

The first run downloads WikiText-103 from Hugging Face and writes token caches
under `data/token_cache/`. Later runs reuse those cached token files.

When `max_steps` is unset, `num_epochs: 1` means one pass over the available
training blocks.

## Larger TinyStories Comparison

The initial full TinyStories and WikiText runs use small 7M-parameter models.
To move closer to the research target, run the larger TinyStories configs:

```bash
cd memory-lm
python3 experiments/count_real_tokens.py \
  configs/large_tinystories_baseline.yaml \
  configs/large_tinystories_per_token.yaml \
  configs/large_tinystories_recurrent.yaml
```

Then launch:

```bash
cd memory-lm
nohup python3 experiments/run_large_tinystories_comparison.py > large_tinystories.out 2>&1 &
tail -f large_tinystories.out
```

Summarize after completion:

```bash
cd memory-lm
python3 experiments/summarize_large_tinystories_comparison.py
```

The original large recurrent config uses 8 very wide slots. To check whether
that shape is unfairly weak, run the matched-budget and parameter-matched
many-slot recurrent variant:

```bash
cd memory-lm
nohup python3 experiments/run_large_tinystories_recurrent_shape_check.py > large_tinystories_recurrent_512x512.out 2>&1 &
tail -f large_tinystories_recurrent_512x512.out
```

## Larger WikiText-103 Comparison

After the larger TinyStories run, use the larger WikiText configs to test
whether the 35M-scale result replicates on a second natural-language dataset.
These configs use the same 35M compressed-memory architecture and match
per-token vs recurrent on parameter count and persistent memory floats.

Inspect the setup:

```bash
cd memory-lm
python3 experiments/count_real_tokens.py \
  configs/large_wikitext_baseline.yaml \
  configs/large_wikitext_per_token.yaml \
  configs/large_wikitext_recurrent.yaml
```

Launch:

```bash
cd memory-lm
nohup python3 experiments/run_large_wikitext_comparison.py > large_wikitext.out 2>&1 &
tail -f large_wikitext.out
```

Summarize after completion:

```bash
cd memory-lm
python3 experiments/summarize_large_wikitext_comparison.py
```

The original large recurrent config uses 8 very wide slots. To check whether
that shape is unfairly weak, run the matched-budget and parameter-matched
many-slot recurrent variant:

```bash
cd memory-lm
nohup python3 experiments/run_large_wikitext_recurrent_shape_check.py > large_wikitext_recurrent_512x512.out 2>&1 &
tail -f large_wikitext_recurrent_512x512.out
```

## Associative Recurrent Synthetic Probe

The first structured recurrent variant is available as `assoc_recurrent`. It
uses explicit associative reads and writes with fixed learned memory keys,
separate learned write queries, and per-sequence memory values. For the small
synthetic probes, compare it against naive recurrent using `256x128`: 256
memory slots, 128 floats per slot, 32,768 persistent memory floats total. This
keeps memory width equal to the small synthetic model hidden size.

```bash
python3 experiments/run_copy_sweep.py \
  --lengths 16 32 64 \
  --models recurrent assoc_recurrent \
  --recurrent-shapes 256x128 \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 128 \
  --csv-path logs/copy_assoc_recurrent.csv
```

```bash
python3 experiments/run_needle_sweep.py \
  --gaps 16 32 64 \
  --models recurrent assoc_recurrent \
  --recurrent-shapes 256x128 \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 128 \
  --answer-loss-weight 10 \
  --csv-path logs/needle_assoc_recurrent.csv
```

```bash
python3 experiments/run_kv_sweep.py \
  --pairs 4 8 16 \
  --models recurrent assoc_recurrent \
  --recurrent-shapes 256x128 \
  --steps 5000 \
  --num-examples 50000 \
  --test-examples 5000 \
  --batch-size 128 \
  --num-keys 16 \
  --num-values 16 \
  --answer-loss-weight 20 \
  --value-mode random \
  --csv-path logs/kv_assoc_recurrent.csv
```

These CSV files include memory diagnostics where available:
`read_entropy`, `write_entropy`, `memory_delta_norm`, and
`memory_value_norm`. Associative runs also log upstream write diagnostics:
`token_out_norm`, `write_source_norm`, `write_value_norm`, `candidate_norm`,
and `raw_memory_norm`.

If associative memory values become unstable, test the stabilized variant:

```bash
python3 experiments/run_copy_sweep.py \
  --lengths 16 32 64 \
  --models assoc_recurrent \
  --recurrent-shapes 256x128 \
  --assoc-write-norm \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 128 \
  --csv-path logs/copy_assoc_recurrent_write_norm.csv
```

```bash
python3 experiments/run_needle_sweep.py \
  --gaps 16 32 64 \
  --models assoc_recurrent \
  --recurrent-shapes 256x128 \
  --assoc-write-norm \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 128 \
  --answer-loss-weight 10 \
  --csv-path logs/needle_assoc_recurrent_write_norm.csv
```

If write-side normalization still leaves unstable memory values, also test
post-write memory normalization/clipping:

```bash
python3 experiments/run_copy_sweep.py \
  --lengths 16 32 64 \
  --models assoc_recurrent \
  --recurrent-shapes 256x128 \
  --assoc-write-norm \
  --assoc-memory-norm \
  --assoc-memory-clip 16 \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 128 \
  --csv-path logs/copy_assoc_recurrent_stabilized.csv
```

```bash
python3 experiments/run_needle_sweep.py \
  --gaps 16 32 64 \
  --models assoc_recurrent \
  --recurrent-shapes 256x128 \
  --assoc-write-norm \
  --assoc-memory-norm \
  --assoc-memory-clip 16 \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 128 \
  --answer-loss-weight 10 \
  --csv-path logs/needle_assoc_recurrent_stabilized.csv
```

## Published Recurrent Baseline Probe

The first published-style recurrent baseline is available as `rmt`. It follows
the language-modeling mechanics from the public RMT implementation
(`booydar/recurrent-memory-transformer`, commit
`9d0ebe1778687995697fe68e886bc1dcf0e45e1c`) inside this repo's small decoder
backbone: each chunk receives previous memory tokens as a prefix, the same
memory tokens are placed after the chunk as write positions, and the final
hidden states at those write positions become the next recurrent state. Text
logits are taken only from text positions, so suffix write tokens do not break
causal language modeling.

This is still an adapted in-harness implementation, not a full reproduction of
the upstream RMT training stack. The paper should call it an "RMT-style baseline
adapted from the public implementation" unless we later run the upstream code
directly.

Start with synthetic tasks before any real-data RMT run. RMT-style memory adds
memory tokens to the attention sequence, so it is much more compute-heavy than
the custom low-rank recurrent updater.

```bash
python3 experiments/run_rmt_synthetic_baseline.py
```

This writes:

```text
logs/copy_rmt_baseline.csv
logs/needle_rmt_baseline.csv
logs/kv_rmt_baseline.csv
```

Use this baseline to answer the reviewer question: whether per-token memory
only beats this repo's custom recurrent updater, or whether it also beats an
RMT-style memory-token recurrence under the same synthetic memory budget.

Real-data runs also save checkpoints every 5000 steps and at the end:

```text
checkpoints/real_baseline/
checkpoints/real_per_token/
checkpoints/real_recurrent/
```

## 30M-50M Starting Scale

The default YAML configs are laptop-friendly synthetic configs. The requested research starting scale is represented by:

```yaml
hidden_size: 512
num_layers: 8
num_heads: 8
context_length: 512
```

See `configs/research_30m_example.yaml`. With SwiGLU MLP ratio 4, this is roughly in the 30M-50M range depending on vocabulary size and memory variant.

## Assumptions To Revisit Before Paper-Scale Runs

- The recurrent model currently stores one shared memory state between chunks, not separate layerwise memories.
- The recurrent update uses mean pooling plus a low-rank gated update. This is intentionally simple and may be too weak for final experiments.
- Per-token compressed attention applies RoPE in compressed head space. Very small `memory_dim` values may lose positional capacity.
- Training FLOPs are not yet exactly matched; persistent memory accounting is implemented first.
- The real-text path now supports `tiktoken`; tokenizer choice must remain fixed across compared runs.
