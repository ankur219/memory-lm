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

## Larger TinyStories Full Validation

Setup:

- Dataset: full TinyStories train split.
- Validation: full official TinyStories validation split.
- Tokenizer: GPT-2 `tiktoken`.
- Train tokens per model: 473,992,192.
- Validation tokens: 4,765,824.
- Context length: 128.
- Model scale: 35M-38M parameters.
- Memory-compressed variants use 262,144 persistent memory floats.
- Baseline ran with batch size 128; per-token and recurrent ran with batch size 96 due to per-token VRAM limits.

Command:

```bash
python3 experiments/run_large_tinystories_comparison.py
python3 experiments/summarize_large_tinystories_comparison.py
```

Result:

| Model | Train Loss | Val Loss | Perplexity | Tokens/sec | Peak VRAM | Params | Mem Floats |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1.5089 | 1.5499 | 4.71 | 41.4k | 18,791 MB | 38,179,584 | 786,432 |
| Per-token many-small | 1.5204 | 1.5802 | 4.86 | 43.1k | 16,056 MB | 35,431,680 | 262,144 |
| Recurrent few-rich | 1.7366 | 1.7068 | 5.51 | 43.0k | 14,907 MB | 35,431,680 | 262,144 |

Interpretation:

- The 7M TinyStories ordering holds at larger scale: baseline best, per-token close behind, recurrent worse.
- Among matched compressed-memory models, per-token many-small is substantially better than recurrent few-rich.
- Per-token and recurrent used the same training tokens, parameter count, and persistent-memory budget.
- Different batch sizes mean throughput should not be over-interpreted in this row.

## WikiText-103 Full Validation

Setup:

- Dataset: full WikiText-103 train split, `Salesforce/wikitext`, `wikitext-103-raw-v1`.
- Validation: full WikiText-103 validation split.
- Tokenizer: GPT-2 `tiktoken`.
- Train tokens per model: 119,085,056 processed from a 119,085,170-token cache.
- Validation tokens: 249,751-token cache, evaluated in 16 batches.
- Context length: 128.
- Batch size: 128.
- Memory-compressed variants use 32,768 persistent memory floats.

Command:

```bash
python3 experiments/run_wikitext_comparison.py
python3 experiments/summarize_wikitext_comparison.py
```

Result:

| Model | Train Loss | Val Loss | Perplexity | Tokens/sec | Peak VRAM | Params | Mem Floats |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 4.3248 | 4.3454 | 77.12 | 182.6k | 13,184 MB | 6,957,824 | 65,536 |
| Per-token many-small | 4.5074 | 4.3592 | 78.19 | 135.4k | 16,286 MB | 6,942,976 | 32,768 |
| Recurrent few-rich | 4.1705 | 4.4231 | 83.36 | 175.8k | 13,344 MB | 6,942,976 | 32,768 |

Interpretation:

- Baseline is best, as expected, because it uses twice the persistent memory.
- Per-token many-small is very close to baseline despite using half the persistent memory.
- Recurrent few-rich is worse than per-token on validation loss, but runs much faster than per-token.
- The ordering matches TinyStories: baseline best, per-token close behind, recurrent worse.

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

### Associative Recurrent Copy Probe

This sweep compares naive recurrent memory against explicit associative
read/write memory at the same persistent-memory budget. Both use `256x128`,
or 32,768 persistent memory floats. The associative model has more parameters
because it adds explicit read/write projections.

Command:

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

Result:

| Copy Length | Naive recurrent | Assoc recurrent |
|---:|---:|---:|
| 16 | 0.942 | 0.998 |
| 32 | 0.046 | 0.064 |
| 64 | 0.031 | 0.023 |

Diagnostics:

| Copy Length | Model | Params | Mem Floats | Write Entropy | Delta Norm | Value Norm |
|---:|---|---:|---:|---:|---:|---:|
| 16 | Naive recurrent | 471,936 | 32,768 | 1.757 | 4.822 | 8.928 |
| 16 | Assoc recurrent | 632,704 | 32,768 | 0.831 | 26.934 | 28.218 |
| 32 | Naive recurrent | 471,936 | 32,768 | 2.531 | 0.456 | 0.298 |
| 32 | Assoc recurrent | 632,704 | 32,768 | 2.198 | 5.027 | 4.897 |
| 64 | Naive recurrent | 471,936 | 32,768 | 2.764 | 1.645 | 1.641 |
| 64 | Assoc recurrent | 632,704 | 32,768 | 2.908 | 2141.628 | 2309.025 |

Interpretation: associative memory strongly improves short copy length and
slightly improves length 32, but it does not solve long exact copying. At
length 64 the associative memory values become unstable, with very large update
and value norms. This argues for adding memory normalization or clipping before
running associative memory on real text.

### Last-Token Recurrent Update

This variant updates recurrent memory from the last token states in each chunk
instead of mean pooling or cross-attention over the full chunk.

Copy command:

```bash
python experiments/run_copy_sweep.py \
  --lengths 16 32 64 \
  --models recurrent \
  --recurrent-update-style last_tokens \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 128 \
  --csv-path logs/copy_recurrent_last_tokens.csv
```

Needle command:

```bash
python experiments/run_needle_sweep.py \
  --gaps 16 32 64 \
  --models recurrent \
  --recurrent-update-style last_tokens \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 128 \
  --answer-loss-weight 10 \
  --csv-path logs/needle_recurrent_last_tokens.csv
```

Copy result:

| Copy Length | Cross-attn Recurrent | Last-token Recurrent |
|---:|---:|---:|
| 16 | 0.942 | 0.942 |
| 32 | 0.045 | 0.046 |
| 64 | 0.030 | 0.031 |

Needle result:

| Gap Length | Cross-attn Recurrent | Last-token Recurrent |
|---:|---:|---:|
| 16 | 1.000 | 1.000 |
| 32 | 0.016 | 0.016 |
| 64 | 0.015 | 0.018 |

Interpretation: the last-token update does not fix recurrent collapse. This
suggests the failure is not only due to mean pooling or cross-attention update
details.

## Synthetic Needle

The needle task tests recalling one exact value after filler:

```text
<BOS> filler <NEEDLE> value filler <QUERY> <ANSWER> value <EOS>
```

The metric is answer-token accuracy.

Command:

```bash
python experiments/run_needle_sweep.py \
  --gaps 8 16 32 64 \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 128 \
  --answer-loss-weight 10 \
  --csv-path logs/needle_sweep.csv
```

Result:

| Gap Length | Baseline | Per-token | Recurrent |
|---:|---:|---:|---:|
| 8 | 1.000 | 1.000 | 1.000 |
| 16 | 1.000 | 1.000 | 1.000 |
| 32 | 1.000 | 1.000 | 0.016 |
| 64 | 1.000 | 1.000 | 0.015 |

Interpretation: baseline and per-token recall the needle perfectly up to gap
64. The recurrent model collapses after gap 16, matching the copy-task failure
mode.

### Associative Recurrent Needle Probe

This sweep compares naive recurrent memory against associative recurrent memory
using the same `256x128` persistent-memory shape, or 32,768 memory floats.

Command:

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

Result:

| Gap Length | Naive recurrent | Assoc recurrent |
|---:|---:|---:|
| 16 | 1.000 | 1.000 |
| 32 | 0.014 | 0.014 |
| 64 | 0.013 | 0.015 |

Diagnostics:

| Gap Length | Model | Params | Mem Floats | Write Entropy | Delta Norm | Value Norm |
|---:|---|---:|---:|---:|---:|---:|
| 16 | Naive recurrent | 480,128 | 32,768 | 3.401 | 0.161 | 0.161 |
| 16 | Assoc recurrent | 640,896 | 32,768 | 3.401 | 3.294 | 3.294 |
| 32 | Naive recurrent | 480,128 | 32,768 | 2.161 | 3.446 | 2.998 |
| 32 | Assoc recurrent | 640,896 | 32,768 | 2.438 | 3.448 | 3.447 |
| 64 | Naive recurrent | 480,128 | 32,768 | 1.157 | 2.878 | 3.108 |
| 64 | Assoc recurrent | 640,896 | 32,768 | 1.155 | 21.351 | 18.855 |

Interpretation: associative recurrent memory does not fix needle recall. Both
models solve gap 16 and collapse at gaps 32-64. The associative model again
shows larger memory value/update norms at the longest setting.

## Current Takeaway

The current evidence supports a cautious statement:

> Under the tested budget and implementation, many-small per-token memory gives better language-modeling loss and generally stronger random key-value retrieval than few-rich recurrent memory, while few-rich recurrent memory is more efficient.

Copy and needle are the cleanest synthetic evidence so far: per-token memory
preserves exact details far better than the recurrent memory designs tested
here. Changing recurrent slot allocation, adding a last-token update, and adding
explicit associative read/write do not solve the long-range exact-recall
collapse. Associative memory helps short copy and slightly helps copy length 32,
but not needle or long copy. The KV probes also show that task design matters:
identity retrieval is solved, but shifted and random binding remain difficult
for these small models.

Next useful experiments:

1. Finish associative KV retrieval and shape sweeps.
2. Add memory normalization/clipping or a true fast-weight/outer-product recurrent variant.
3. Run larger WikiText configs to replicate the scaled TinyStories result on a second dataset.
