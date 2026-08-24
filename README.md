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

See `RESULTS.md` for the current TinyStories and synthetic KV retrieval
checkpoint.

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
prompt separately from the predicted continuation, so it is clear what the
model received and what it generated. When `max_steps` is unset, `num_epochs: 1`
means one pass over the available training blocks.

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
