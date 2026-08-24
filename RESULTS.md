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

### Shifted Values

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
  --value-mode shifted \
  --csv-path logs/kv_sweep_shifted.csv
```

Result:

| Pairs | Baseline | Per-token | Recurrent |
|---:|---:|---:|---:|
| 4 | 0.251 | 0.251 | 0.247 |
| 8 | 0.124 | 0.119 | 0.114 |
| 16 | 0.062 | 0.056 | 0.043 |
| 32 | 0.035 | 0.031 | 0.028 |

Interpretation: shifted values were not the hoped-for middle-difficulty task.
The models are near chance by 16-32 pairs, so this task is mostly too hard at
the current scale.

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

## Synthetic Copy

The copy task tests exact token preservation:

```text
<BOS> x1 x2 ... xn <COPY> x1 x2 ... xn <EOS>
```

The metric is copy-span token accuracy.

Command:

```bash
python experiments/run_copy_sweep.py \
  --lengths 8 16 32 64 \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 128 \
  --csv-path logs/copy_sweep.csv
```

Result:

| Copy Length | Baseline | Per-token | Recurrent |
|---:|---:|---:|---:|
| 8 | 1.000 | 1.000 | 1.000 |
| 16 | 1.000 | 1.000 | 0.942 |
| 32 | 1.000 | 1.000 | 0.045 |
| 64 | 1.000 | 1.000 | 0.030 |

Interpretation: baseline and per-token copy almost perfectly up to length 64.
The recurrent model collapses after length 16, suggesting the few-rich summary
state loses exact token details under this update policy.

### Recurrent Shape Sweep

This sweep keeps recurrent persistent memory fixed at 32,768 floats while
changing the number and width of memory slots.

Command:

```bash
python experiments/run_copy_sweep.py \
  --lengths 16 32 64 \
  --models recurrent \
  --recurrent-shapes 8x4096 16x2048 32x1024 64x512 \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 128 \
  --csv-path logs/copy_recurrent_shape_sweep.csv
```

Result:

| Recurrent Shape | Length 16 | Length 32 | Length 64 |
|---|---:|---:|---:|
| 8x4096 | 0.942 | 0.045 | 0.030 |
| 16x2048 | 0.942 | 0.057 | 0.031 |
| 32x1024 | 0.942 | 0.046 | 0.030 |
| 64x512 | 0.942 | 0.069 | 0.031 |

Interpretation: changing recurrent slot allocation does not solve the copy
collapse. More slots help slightly at length 32, but all shapes remain near
chance at length 64. This suggests the current recurrent update loses exact
sequence detail, not merely that it has too few slots.

## Current Takeaway

The current evidence supports a cautious statement:

> Under the tested budget and implementation, many-small per-token memory gives better language-modeling loss and generally stronger random key-value retrieval than few-rich recurrent memory, while few-rich recurrent memory is more efficient.

The copy result is the cleanest synthetic evidence so far: per-token memory
preserves exact details far better than the current recurrent memory design. The
KV probes also show that task design matters: identity retrieval is solved, but
shifted and random binding remain difficult for these small models.

Next useful experiments:

1. Add a needle-in-context probe.
2. Test stronger recurrent update policies.
