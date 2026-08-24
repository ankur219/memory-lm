# Current Results

This file records the current research checkpoint. These are one-seed results,
not final paper claims.

## TinyStories Full Validation

Setup:

- Dataset: full TinyStories train split.
- Validation: full official TinyStories validation split.
- Tokenizer: GPT-2 `tiktoken`.
- Train tokens per model: 473,992,192.
- Validation tokens: 4,765,824.
- Context length: 128.
- Batch size: 128.
- Memory-compressed variants use 32,768 persistent memory floats.

| Model | Val Loss | Perplexity | Tokens/sec | Peak VRAM | Params | Mem Floats |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 2.0459 | 7.74 | 194.0k | 13,184 MB | 6,957,824 | 65,536 |
| Per-token many-small | 2.0850 | 8.04 | 143.0k | 16,286 MB | 6,942,976 | 32,768 |
| Recurrent few-rich, mean-GRU | 2.1771 | 8.82 | 188.6k | 13,239 MB | 6,942,464 | 32,768 |
| Recurrent few-rich, cross-attn | 2.1788 | 8.84 | 187.9k | 13,345 MB | 6,942,976 | 32,768 |

Interpretation:

- Baseline has the best loss, but also uses the full KV memory budget.
- Among equal persistent-memory variants, per-token many-small has better loss.
- Recurrent few-rich is faster and uses much less peak VRAM than per-token, but has worse loss.
- Cross-attention recurrent did not improve over the simpler mean-GRU update in this run.

## Synthetic KV Retrieval

The synthetic task tests exact retrieval:

```text
K7 V12 K3 V4 ... QUERY K7 ANSWER V12
```

The metric is answer-token accuracy.

### Identity Values

Command:

```bash
python experiments/run_kv_sweep.py \
  --pairs 4 8 16 32 \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 128 \
  --num-keys 32 \
  --num-values 32 \
  --answer-loss-weight 10 \
  --value-mode identity \
  --csv-path logs/kv_sweep_identity.csv
```

Result:

| Pairs | Baseline | Per-token | Recurrent |
|---:|---:|---:|---:|
| 4 | 1.000 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 | 1.000 |
| 16 | 1.000 | 1.000 | 1.000 |
| 32 | 1.000 | 1.000 | 1.000 |

Interpretation: all models can learn the retrieval format when the value rule is simple.

### Random Values, Easier Setting

Command:

```bash
python experiments/run_kv_sweep.py \
  --pairs 4 8 16 \
  --steps 5000 \
  --num-examples 50000 \
  --test-examples 5000 \
  --batch-size 128 \
  --num-keys 16 \
  --num-values 16 \
  --answer-loss-weight 20 \
  --value-mode random \
  --csv-path logs/kv_sweep_random_easy.csv
```

Result:

| Pairs | Baseline | Per-token | Recurrent |
|---:|---:|---:|---:|
| 4 | 0.316 | 0.332 | 0.329 |
| 8 | 0.211 | 0.227 | 0.216 |
| 16 | 0.157 | 0.155 | 0.116 |

Interpretation: arbitrary in-context key-value binding is much harder. Per-token is generally strongest among the memory-compressed variants, while recurrent lags at 16 pairs.

## Current Takeaway

The current evidence supports a cautious statement:

> Under the tested budget and implementation, many-small per-token memory gives better language-modeling loss and generally stronger random key-value retrieval than few-rich recurrent memory, while few-rich recurrent memory is more efficient.

Next useful experiments:

1. Add a middle-difficulty `shifted` value mode.
2. Sweep memory allocation: recurrent `8 x 4096`, `16 x 2048`, `32 x 1024`, `64 x 512`.
3. Add copy and needle-in-context probes.
