# Current Results

This file records the current research checkpoint. The main real-data tables
are still being seed-checked; focused synthetic comparisons include multi-seed
checks where noted.

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

Seed 0:

| Model | Train Loss | Val Loss | Perplexity | Tokens/sec | Peak VRAM | Params | Mem Floats |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1.5089 | 1.5499 | 4.71 | 41.4k | 18,791 MB | 38,179,584 | 786,432 |
| Per-token many-small | 1.5204 | 1.5802 | 4.86 | 43.1k | 16,056 MB | 35,431,680 | 262,144 |
| Recurrent few-rich | 1.7366 | 1.7068 | 5.51 | 43.0k | 14,907 MB | 35,431,680 | 262,144 |

Seed 1 completed rows:

| Model | Train Loss | Val Loss | Perplexity | Tokens/sec | Peak VRAM | Params | Mem Floats | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 1.5004 | 1.5433 | 4.68 | 23.6k | 9,628 MB | 38,179,584 | 786,432 | completed |
| Per-token many-small | 1.5514 | 1.5812 | 4.86 | 29.4k | 10,849 MB | 35,431,680 | 262,144 | completed |
| Recurrent fair shape `128x2048` | 1.7136 | 1.7029 | 5.49 | 15.2k | 16,892 MB | 35,450,112 | 262,144 | completed; main recurrent seed-1 row |

Interpretation:

- The 7M TinyStories ordering holds at larger scale: baseline best, per-token close behind, recurrent worse.
- Among matched compressed-memory models, per-token many-small is substantially better than recurrent few-rich.
- Per-token and recurrent used the same training tokens, parameter count, and persistent-memory budget.
- Different batch sizes mean throughput should not be over-interpreted in this row.
- Caveat: this recurrent row uses `8x32768` memory, an extreme few-giant-slots
  shape.
- Seed 1 uses the fairer `128x2048` recurrent shape as the main recurrent row,
  rather than rerunning the original extreme `8x32768` shape.

### 35M TinyStories Recurrent Shape Check

The initial proposed fairness check was `512x512`, but that shape was too
activation-heavy on a 24GB RTX 3090. We instead ran a full-token-budget
`128x2048` recurrent shape:

```text
128 memory tokens x 2048 dim = 262,144 memory floats
```

This keeps the same persistent-memory budget as the `8x32768` recurrent run
while using many more, narrower memory slots.

| Recurrent Shape | Train Loss | Val Loss | Perplexity | Params | Mem Floats | Train Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `8x32768` | 1.7366 | 1.7068 | 5.51 | 35,431,680 | 262,144 | 473,992,192 |
| `128x2048` | 1.7136 | 1.7029 | 5.49 | 35,450,112 | 262,144 | 473,992,192 |
| `128x2048`, seed 1 | 1.7136 | 1.7029 | 5.49 | 35,450,112 | 262,144 | 473,992,192 |

Interpretation: changing the 35M recurrent model from extreme few-giant-slots
memory to a more balanced `128x2048` shape slightly improves TinyStories
validation loss, but the improvement is small (`1.7068 -> 1.7029`) and does not
close the gap to per-token many-small memory (`1.5802`). This weakens the
concern that the main recurrent result is only an artifact of the original
`8x32768` shape, though WikiText has not yet been rerun with this shape.

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

## Larger WikiText-103 Full Validation

Setup:

- Dataset: full WikiText-103 train split, `Salesforce/wikitext`, `wikitext-103-raw-v1`.
- Validation: full WikiText-103 validation split.
- Tokenizer: GPT-2 `tiktoken`.
- Train tokens per model: 119,085,056 processed from a 119,085,170-token cache.
- Validation tokens: 249,751-token cache, evaluated in 21 batches.
- Context length: 128.
- Batch size: 96.
- Model scale: 35M-38M parameters.
- Memory-compressed variants use 262,144 persistent memory floats.

Command:

```bash
python3 experiments/run_large_wikitext_comparison.py
python3 experiments/summarize_large_wikitext_comparison.py
```

Result:

Seed 0:

| Model | Val Loss | Perplexity | Tokens/sec | Peak VRAM | Params | Mem Floats |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 3.6202 | 37.35 | 33.6k | 14,205 MB | 38,179,584 | 786,432 |
| Per-token many-small | 3.6831 | 39.77 | 34.1k | 16,052 MB | 35,431,680 | 262,144 |
| Recurrent few-rich | 3.7973 | 44.58 | 34.8k | 14,890 MB | 35,431,680 | 262,144 |

Seed 1 completed rows:

| Model | Train Loss | Val Loss | Perplexity | Tokens/sec | Peak VRAM | Params | Mem Floats | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 3.6439 | 3.6223 | 37.42 | 30.8k | 14,205 MB | 38,179,584 | 786,432 | completed |
| Per-token many-small | pending | pending | pending | pending | pending | 35,431,680 | 262,144 | seed-1 run still needed |
| Recurrent fair shape `128x2048` | 3.8147 | 3.7861 | 44.08 | 15.8k | 16,892 MB | 35,450,112 | 262,144 | completed; main recurrent seed-1 row |

Interpretation:

- The larger WikiText run matches the larger TinyStories ordering: baseline
  best, per-token many-small next, recurrent few-rich worst.
- Among matched compressed-memory models, per-token many-small again has better
  validation loss than recurrent few-rich with the same parameter count,
  training tokens, context length, tokenizer, optimizer settings, and persistent
  memory budget.
- Caveat: this recurrent row uses `8x32768` memory, an extreme few-giant-slots
  shape.
- Seed 1 currently has the fairer recurrent shape and baseline completed.
  WikiText per-token seed 1 is still pending, so the WikiText seed-1 table is
  not yet complete.

### 35M WikiText-103 Recurrent Shape Check

As with TinyStories, the full `512x512` recurrent shape was too
activation-heavy on a 24GB RTX 3090. We instead ran a full-token-budget
`128x2048` recurrent shape:

```text
128 memory tokens x 2048 dim = 262,144 memory floats
```

This keeps the same persistent-memory budget as the `8x32768` recurrent run
while using many more, narrower memory slots.

| Recurrent Shape | Train Loss | Val Loss | Perplexity | Params | Mem Floats | Train Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `8x32768` | 3.7435 | 3.7973 | 44.58 | 35,431,680 | 262,144 | 119,085,056 |
| `128x2048` | 3.8147 | 3.7861 | 44.08 | 35,450,112 | 262,144 | 119,085,056 |
| `128x2048`, seed 1 | 3.8147 | 3.7861 | 44.08 | 35,450,112 | 262,144 | 119,085,056 |

Interpretation: the fairer `128x2048` recurrent shape again improves
validation loss slightly (`3.7973 -> 3.7861`), but the improvement is small and
does not close the gap to per-token many-small memory (`3.6831`). Together with
the TinyStories shape check, this weakens the concern that the recurrent
underperformance is mainly caused by the original `8x32768` shape.

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

### Associative Recurrent Random KV

This sweep compares naive recurrent memory against associative recurrent memory
on the easier random key-value setting, using the same `256x128` persistent
memory shape.

Command:

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

Result:

| Pairs | Naive recurrent | Assoc recurrent |
|---:|---:|---:|
| 4 | 0.328 | 0.314 |
| 8 | 0.234 | 0.222 |
| 16 | 0.123 | 0.121 |

Diagnostics:

| Pairs | Model | Params | Mem Floats | Write Entropy | Delta Norm | Value Norm |
|---:|---|---:|---:|---:|---:|---:|
| 4 | Naive recurrent | 467,840 | 32,768 | 2.565 | 0.469 | 0.469 |
| 4 | Assoc recurrent | 628,608 | 32,768 | 2.560 | 13.642 | 13.642 |
| 8 | Naive recurrent | 467,840 | 32,768 | 3.045 | 0.868 | 0.868 |
| 8 | Assoc recurrent | 628,608 | 32,768 | 3.020 | 60.818 | 60.818 |
| 16 | Naive recurrent | 467,840 | 32,768 | 2.536 | 0.494 | 0.813 |
| 16 | Assoc recurrent | 628,608 | 32,768 | 0.623 | 0.024 | 0.024 |

Interpretation: associative recurrent memory does not improve random key-value
retrieval. It is slightly worse than naive recurrent at all tested pair counts,
despite using the same persistent-memory budget and more parameters.

### Associative Recurrent Random KV With Write Normalization

Status: single-seed probe; not a final claim.

This run tests whether RMS-normalizing the token states before associative
write-key/write-value projections improves random key-value retrieval. The
memory budget and `256x128` shape are unchanged.

Result:

| Pairs | Assoc raw | Assoc write-norm |
|---:|---:|---:|
| 4 | 0.314 | 0.314 |
| 8 | 0.222 | 0.222 |
| 16 | 0.121 | 0.107 |

Diagnostics:

| Pairs | Params | Mem Floats | Read Entropy | Write Entropy | Token Out Norm | Write Source Norm | Candidate Norm | Value Norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 628,736 | 32,768 | 5.545 | 2.565 | 213.186 | 11.314 | 0.791 | 0.395 |
| 8 | 628,736 | 32,768 | 5.545 | 3.045 | 546.101 | 11.314 | 0.963 | 0.482 |
| 16 | 628,736 | 32,768 | 5.531 | 2.198 | 177.866 | 9.523 | 5.253 | 1.562 |

Interpretation: write-side normalization controls the write-source magnitude,
but it does not improve random key-value retrieval, and it is measurably worse
at 16 pairs. The normalization benefit seen later on needle gap 32 is therefore
task-specific, not a general KV retrieval improvement.

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

### Associative Copy With Write Normalization

Status: single-seed probe; copy-64 evaluation is high-resolution because it
scores 195,000 copied-token predictions.

This run adds RMSNorm before the associative write-key/write-value projections.
The model is trained from scratch with normalization enabled.

Result:

| Copy Length | Assoc raw | Assoc write-norm |
|---:|---:|---:|
| 16 | 0.998 | 0.941 |
| 32 | 0.064 | 0.053 |
| 64 | 0.023 | 0.023 |

Diagnostics:

| Copy Length | Params | Mem Floats | Write Entropy | Token Out Norm | Write Source Norm | Candidate Norm | Value Norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 632,832 | 32,768 | 1.920 | 115.108 | 11.039 | 8.392 | 7.727 |
| 32 | 632,832 | 32,768 | 1.875 | 31.749 | 10.221 | 7.582 | 3.302 |
| 64 | 632,832 | 32,768 | 2.911 | 732.752 | 11.304 | 17.753 | 24.025 |

Interpretation: write-normalization reduces the magnitude problem, but does
not improve dense exact copying. This suggests the copy failure is primarily a
multi-item capacity/allocation problem rather than only a write-scale problem.

### Copy-64 With 2x Few-Rich Budget

Status: single-seed probe; parameter confound explicitly measured.

This run doubles the few-rich persistent memory from `256x128` to `512x128`
while keeping the memory dimension equal to the hidden size. The associative
variant is raw associative memory, without write normalization or clipping, so
the isolated variable is the number of memory slots.

Result:

| Model | Shape | Params | Mem Floats | Copy-64 Accuracy |
|---|---|---:|---:|---:|
| Naive recurrent | 256x128 | 471,936 | 32,768 | 0.031 |
| Assoc recurrent | 256x128 | 632,704 | 32,768 | 0.023 |
| Naive recurrent | 512x128 | 471,936 | 65,536 | 0.031 |
| Assoc recurrent | 512x128 | 698,240 | 65,536 | 0.023 |

Diagnostics:

| Model | Shape | Write Entropy | Delta Norm | Value Norm | Token Out Norm | Candidate Norm |
|---|---|---:|---:|---:|---:|---:|
| Naive recurrent | 512x128 | 0.693 | 10.380 | 8.919 | - | - |
| Assoc recurrent | 512x128 | 2.911 | 246.163 | 247.052 | 396.074 | 320.263 |

Interpretation: doubling few-rich memory does not improve dense exact copy for
either recurrent variant. Since copy accuracy is measured over many copied
tokens, this is not just a coarse-evaluation artifact. The naive recurrent
diagnostic gives a more specific mechanism: write entropy is `0.693`, about
`ln(2)`, so the model is effectively writing to roughly two slots out of the
512 available. In this run, extra few-rich slots did not help because the model
did not learn to use them. The associative model has more learned parameters
than naive recurrent, and the parameter gap widens from +160,768 at `256x128`
to +226,304 at `512x128`, because fixed associative read/write slot parameters
scale with the number of slots.

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

### Associative Needle With Write Normalization

Status: seed check completed; the positive gap-32 result is not reliable.

This run adds RMSNorm before the associative write-key/write-value projections.
The model is trained from scratch with normalization enabled.

Result:

| Gap Length | Assoc raw | Assoc write-norm |
|---:|---:|---:|
| 16 | 1.000 | 1.000 |
| 32 | 0.014 | 0.591 |
| 64 | 0.015 | 0.015 |

Seed check for gap 32:

| Seed | Needle-32 Accuracy |
|---:|---:|
| 0 | 0.591 |
| 1 | 0.029 |
| 2 | 0.019 |
| 3 | 0.999 |

Across seeds 0-3, gap-32 accuracy is highly unstable rather than consistently
improved.

Diagnostics:

| Gap Length | Params | Mem Floats | Write Entropy | Token Out Norm | Write Source Norm | Candidate Norm | Value Norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 641,024 | 32,768 | 3.401 | 40.541 | 11.314 | 1.698 | 0.845 |
| 32 | 641,024 | 32,768 | 2.959 | 98.935 | 10.192 | 6.832 | 2.433 |
| 64 | 641,024 | 32,768 | 2.586 | 164.866 | 10.027 | 3.976 | 2.115 |

Interpretation: write normalization can occasionally produce strong gap-32
needle recall, but the effect is not seed-robust in this setup. It should not
be treated as a reliable improvement without further stabilization or a better
training protocol.

### Needle-64 With 2x Few-Rich Budget

Status: seed check completed; the positive seed-0 result did not replicate.

This run doubles the few-rich persistent memory from `256x128` to `512x128`.
The associative variant is raw associative memory, without write normalization
or clipping.

Result:

| Model | Shape | Params | Mem Floats | Needle-64 Accuracy |
|---|---|---:|---:|---:|
| Naive recurrent | 512x128 | 480,128 | 65,536 | 0.013 |
| Assoc recurrent | 512x128 | 706,432 | 65,536 | 0.227 |

Seed check for `assoc_recurrent`, `512x128`:

| Seed | Needle-64 Accuracy |
|---:|---:|
| 0 | 0.227 |
| 1 | 0.018 |
| 2 | 0.017 |
| 3 | 0.015 |

The 2x associative improvement is therefore a seed-0 outlier, not a robust
effect.

Diagnostics:

| Model | Shape | Read Entropy | Write Entropy | Delta Norm | Value Norm | Token Out Norm | Candidate Norm |
|---|---|---:|---:|---:|---:|---:|---:|
| Naive recurrent | 512x128 | - | 3.190 | 0.044 | 0.075 | - | - |
| Assoc recurrent | 512x128 | 6.238 | 3.112 | 10.498 | 6.861 | 36.120 | 29.309 |

Interpretation: doubling few-rich memory and adding associative read/write does
not reliably solve needle-64. The seed-0 result remains useful diagnostically,
but it should not be used as a positive claim. It is also not a clean
equal-parameter result: the associative model has +226,304 parameters relative
to naive recurrent at this shape, including extra learned read/write slot
parameters. The seed-0 read entropy is `6.238`, close to `ln(512)`, so even that
outlier should not be interpreted as sharp content-addressed lookup of one slot.

## Long-Context Synthetic Results

Status: completed for the planned long-context copy and needle cells.
RMT-style rows were run separately at smaller batch sizes after earlier
long-context RMT attempts OOMed.

These runs test whether the short-context synthetic pattern persists beyond the
original 8-64 length/gap range.

Command:

```bash
nohup bash -c '
python3 experiments/run_copy_sweep.py \
  --lengths 128 256 512 \
  --models baseline per_token recurrent rmt \
  --recurrent-shapes 256x128 \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 64 \
  --csv-path logs/copy_long.csv

python3 experiments/run_needle_sweep.py \
  --gaps 128 256 512 1024 \
  --models baseline per_token recurrent rmt \
  --recurrent-shapes 256x128 \
  --steps 3000 \
  --num-examples 30000 \
  --test-examples 3000 \
  --batch-size 64 \
  --answer-loss-weight 10 \
  --csv-path logs/needle_long.csv
' > long_synthetic.out 2>&1 &
```

### Long Copy

| Copy Length | Baseline | Per-token | Recurrent | RMT-style |
|---:|---:|---:|---:|---:|
| 128 | 1.000 | 1.000 | 0.023 | 0.023 |
| 256 | 0.996 | 0.996 | 0.019 | 0.018 |
| 512 | 0.998 | 1.000 | 0.016 | 0.018 |

Diagnostics for completed copy rows:

| Copy Length | Model | Params | Mem Floats | Peak VRAM | Train Time | Copy Accuracy |
|---:|---|---:|---:|---:|---:|---:|
| 128 | Baseline | 534,400 | 132,608 | 652 MB | 201 sec | 1.000 |
| 128 | Per-token | 468,864 | 66,304 | 619 MB | 182 sec | 1.000 |
| 128 | Recurrent | 471,936 | 32,768 | 5,861 MB | 1,788 sec | 0.023 |
| 256 | Baseline | 534,400 | 263,680 | 1,855 MB | 207 sec | 0.996 |
| 256 | Per-token | 468,864 | 131,840 | 1,798 MB | 194 sec | 0.996 |
| 256 | Recurrent | 471,936 | 32,768 | 11,115 MB | 1,314 sec | 0.019 |
| 512 | Baseline | 534,400 | 525,824 | 6,246 MB | 661 sec | 0.998 |
| 512 | Per-token | 468,864 | 262,912 | 6,133 MB | 609 sec | 1.000 |
| 512 | Recurrent | 471,936 | 32,768 | 21,613 MB | 2,563 sec | 0.016 |
| 128 | RMT-style | 501,632 | 32,768 | 6,499 MB | 1,019 sec | 0.023 |
| 256 | RMT-style | 501,632 | 32,768 | 12,161 MB | 1,943 sec | 0.018 |
| 512 | RMT-style | 501,632 | 32,768 | 3,118 MB | 6,127 sec | 0.018 |

Interpretation: baseline and per-token still solve dense exact-recall
accessibility through copy length 512, while the custom recurrent model remains
near chance despite the same fixed 32,768-float recurrent budget. This extends
the copy-collapse pattern far beyond the earlier length-64 result. RMT-style
memory also remains near chance on long dense copy, matching custom recurrent
rather than per-token memory.

### Long Needle

| Gap Length | Baseline | Per-token | Recurrent | RMT-style |
|---:|---:|---:|---:|---:|
| 128 | 1.000 | 1.000 | 0.015 | 0.807 |
| 256 | 1.000 | 1.000 | 0.014 | 0.014 |
| 512 | 1.000 | 1.000 | 0.015 | 0.195 |
| 1024 | 1.000 | 1.000 | 0.018 | 0.014 |

Diagnostics for completed needle rows:

| Gap Length | Model | Params | Mem Floats | Peak VRAM | Train Time | Accuracy |
|---:|---|---:|---:|---:|---:|---:|
| 128 | Baseline | 542,592 | 72,704 | 343 MB | 121 sec | 1.000 |
| 128 | Per-token | 477,056 | 36,352 | 324 MB | 99 sec | 1.000 |
| 128 | Recurrent | 480,128 | 32,768 | 3,275 MB | 1,012 sec | 0.015 |
| 128 | RMT-style | 509,824 | 32,768 | 7,470 MB | 2,168 sec | 0.807 |
| 256 | Baseline | 542,592 | 138,240 | 693 MB | 207 sec | 1.000 |
| 256 | Per-token | 477,056 | 69,120 | 659 MB | 192 sec | 1.000 |
| 256 | Recurrent | 480,128 | 32,768 | 5,901 MB | 1,815 sec | 0.014 |
| 512 | Baseline | 542,592 | 269,312 | 1,938 MB | 218 sec | 1.000 |
| 512 | Per-token | 477,056 | 134,656 | 1,880 MB | 201 sec | 1.000 |
| 512 | Recurrent | 480,128 | 32,768 | 11,169 MB | 1,316 sec | 0.015 |
| 1024 | Baseline | 542,592 | 531,456 | 6,396 MB | 670 sec | 1.000 |
| 1024 | Per-token | 477,056 | 265,728 | 6,281 MB | 621 sec | 1.000 |
| 1024 | Recurrent | 480,128 | 32,768 | 21,686 MB | 2,580 sec | 0.018 |
| 256 | RMT-style | 509,824 | 32,768 | 3,310 MB | 838 sec | 0.014 |
| 512 | RMT-style | 509,824 | 32,768 | 6,169 MB | 1,584 sec | 0.195 |
| 1024 | RMT-style | 509,824 | 32,768 | 11,967 MB | 3,072 sec | 0.014 |

Interpretation: the available longer-gap needle results strengthen the
task-dependent story. Baseline and per-token solve gap 128 through 1024, while
custom recurrent remains near chance under the fixed 32,768-float budget.
RMT-style memory is strong at gap 128, collapses at gap 256, shows a partial
single-run spike at gap 512, and collapses again at gap 1024. Thus RMT's
long-gap advantage over custom recurrent is real at shorter needle settings,
but it is not monotonic or stable across the full 128-1024 gap range. Per-token
memory remains the only compressed-memory model here that solves every tested
long-needle gap.

## RMT-Style Published Baseline Probe

Status: corrected learned-initial-memory comparison completed with three seeds
for the highest-impact cells.

This compares the repo's custom recurrent baseline against `rmt`, an
RMT-style memory-token baseline adapted from the public RMT implementation. Both
use the same synthetic memory budget:

```text
256 memory tokens x 128 dim = 32,768 memory floats
```

Fairness correction: RMT uses learned initial memory tokens. The custom
recurrent baseline is now also run with learned initial memory, so the comparison
is closer in parameter count and learned memory capacity.

Command:

```bash
python3 experiments/run_rmt_synthetic_baseline.py
```

### Corrected Seed-0 Sweep

| Task | Setting | Custom Recurrent | RMT-style |
|---|---:|---:|---:|
| Copy | 16 | 0.764 | 0.922 |
| Copy | 32 | 0.045 | 0.266 |
| Copy | 64 | 0.031 | 0.065 |
| Needle | 16 | 1.000 | 1.000 |
| Needle | 32 | 0.012 | 0.938 |
| Needle | 64 | 0.015 | 0.903 |
| Random KV | 4 | 0.322 | 0.329 |
| Random KV | 8 | 0.222 | 0.222 |
| Random KV | 16 | 0.123 | 0.067 |

### Corrected Seed Checks

Focused seed checks were run for the cells most affected by the RMT comparison:
copy-32, needle-32, needle-64, and random KV-16. Values are answer/copy
accuracies.

| Task | Model | Seed 0 | Seed 1 | Seed 2 | Mean | Std |
|---|---|---:|---:|---:|---:|---:|
| Copy-32 | Custom recurrent | 0.045 | 0.045 | 0.057 | 0.049 | 0.007 |
| Copy-32 | RMT-style | 0.266 | 0.062 | 0.958 | 0.429 | 0.469 |
| Needle-32 | Custom recurrent | 0.012 | 0.015 | 0.019 | 0.016 | 0.003 |
| Needle-32 | RMT-style | 0.938 | 0.677 | 0.491 | 0.702 | 0.225 |
| Needle-64 | Custom recurrent | 0.015 | 0.016 | 0.013 | 0.015 | 0.002 |
| Needle-64 | RMT-style | 0.903 | 0.966 | 0.763 | 0.877 | 0.104 |
| Random KV-16 | Custom recurrent | 0.123 | 0.067 | 0.123 | 0.104 | 0.032 |
| Random KV-16 | RMT-style | 0.067 | 0.060 | 0.067 | 0.065 | 0.004 |

Interpretation:

- Copy-32 is not robust for RMT-style memory. It sometimes learns the task very
  well, but variance is extremely high.
- Needle-32 is seed-sensitive, but RMT-style memory remains far above the custom
  recurrent baseline in all three seeds.
- Needle-64 is the strongest corrected RMT result: all three RMT seeds are high,
  while all custom recurrent seeds remain near zero.
- Random KV-16 does not favor RMT-style memory after the learned-initial-memory
  correction.

Clarification: the RMT needle numbers are unchanged from the pre-correction seed
check because the fairness correction changed the custom recurrent baseline, not
the RMT model itself. RMT already used learned initial memory.

The random KV-16 result is an opposite-direction finding rather than a tie:
after the learned-initial-memory correction, custom recurrent is better on
average than RMT-style memory (`0.104` vs. `0.065`). This means the result is not
"RMT is better"; the memory mechanism interacts strongly with the task.

Overall RMT interpretation: the published-style memory-token baseline changes
the paper story. The correct claim is not "few-rich recurrent memory is bad."
Instead, the corrected evidence says naive recurrent summarization is weak,
RMT-style memory tokens are robustly strong for long-gap single-fact needle
retrieval, and many-small per-token memory remains the strongest tested strategy
for dense exact copy at longer length.

## Direct Per-Token vs RMT Synthetic Comparison

Status: three-seed direct comparison completed for the most important synthetic
cells.

This comparison asks whether RMT-style memory only beats the custom recurrent
baseline, or whether it also beats the many-small per-token model.

Command:

```bash
for SEED in 0 1 2; do
  python3 experiments/run_copy_sweep.py \
    --lengths 32 64 \
    --models per_token rmt \
    --recurrent-shapes 256x128 \
    --steps 3000 \
    --num-examples 30000 \
    --test-examples 3000 \
    --batch-size 128 \
    --seed $SEED \
    --csv-path logs/copy_rmt_vs_per_token_seed${SEED}.csv

  python3 experiments/run_needle_sweep.py \
    --gaps 32 64 \
    --models per_token rmt \
    --recurrent-shapes 256x128 \
    --steps 3000 \
    --num-examples 30000 \
    --test-examples 3000 \
    --batch-size 128 \
    --answer-loss-weight 10 \
    --seed $SEED \
    --csv-path logs/needle_rmt_vs_per_token_seed${SEED}.csv

  python3 experiments/run_kv_sweep.py \
    --pairs 16 \
    --models per_token rmt \
    --recurrent-shapes 256x128 \
    --steps 5000 \
    --num-examples 50000 \
    --test-examples 5000 \
    --batch-size 128 \
    --num-keys 16 \
    --num-values 16 \
    --answer-loss-weight 20 \
    --value-mode random \
    --seed $SEED \
    --csv-path logs/kv16_rmt_vs_per_token_seed${SEED}.csv
done
```

Results:

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

Interpretation: RMT-style memory is a much stronger recurrent baseline than the
custom recurrent updater for needle retrieval, but it does not beat per-token
memory in the direct synthetic comparison. Per-token is perfect on copy and
needle across these seeds and is better on random KV-16 with non-overlapping
seed ranges: RMT's best KV-16 seed remains below per-token's worst KV-16 seed.
The RMT needle numbers match the corrected recurrent-vs-RMT seed check because
the RMT configuration and seeds are unchanged; this is expected deterministic
reproduction, not a separate copied result. This supports the stronger version
of the many-small result: spreading memory across token-indexed states remains
the most reliable strategy tested here for exact synthetic recall, even when the
few-rich side uses an RMT-style memory-token mechanism.

## Token Salience and Retention Probe

Status: first-pass oracle retention probe completed for copy-32, needle-64, and
random KV-16.

This diagnostic asks whether all per-token memory slots are needed. For a
trained per-token model, each token's compressed K/V memory is ablated to
estimate salience. At evaluation time, only the top salience-ranked slots are
kept, and dropped slots are replaced with zero memory. Random and uniform-stride
retention are same-budget controls.

Command pattern:

```bash
python3 experiments/run_token_retention.py \
  --task copy \
  --copy-length 32 \
  --train-steps 1000 \
  --num-examples 10000 \
  --test-examples 128 \
  --batch-size 16 \
  --analyze-batches 4 \
  --keep-fracs 0.5 0.25 0.1 0.05 \
  --csv-path logs/token_retention_copy32.csv
```

Results:

| Task | Full Per-Token | Best Salience Retention | Random / Control | Conclusion |
|---|---:|---:|---:|---|
| Copy-32 accuracy | 1.000 | 0.983 at 50% memory | random 50%: 0.509; stride 50%: 0.492 | Strong: salience keeps near-full accuracy with half memory. |
| KV-16 loss | 2.843 | 2.850 at 50% memory | random 50%: 4.381; stride 50%: 4.781 | Useful loss result: salience preserves near-full loss, but accuracy is near floor. |
| Needle-64 accuracy/loss | acc 0.062; loss 3.931 | 50% memory: acc 0.078; loss 4.027 | zero memory: acc 0.016; loss 4.339 | Inconclusive: full model did not learn needle well enough. |

Detailed retention results:

| Task | Method | Memory Kept | Accuracy | Loss |
|---|---|---:|---:|---:|
| Copy-32 | full | 100% | 1.000 | 2.0188 |
| Copy-32 | oracle top-k | 50% | 0.983 | 2.0527 |
| Copy-32 | random top-k | 50% | 0.509 | 5.3514 |
| Copy-32 | stride top-k | 50% | 0.492 | 4.6388 |
| Copy-32 | oracle top-k | 25% | 0.463 | 4.4898 |
| Copy-32 | oracle top-k | 10% | 0.191 | 5.8972 |
| KV-16 | full | 100% | 0.047 | 2.8432 |
| KV-16 | oracle top-k | 50% | 0.047 | 2.8497 |
| KV-16 | random top-k | 50% | 0.047 | 4.3811 |
| KV-16 | stride top-k | 50% | 0.047 | 4.7806 |
| KV-16 | oracle top-k | 10% | 0.047 | 3.3526 |
| Needle-64 | full | 100% | 0.062 | 3.9312 |
| Needle-64 | oracle top-k | 50% | 0.078 | 4.0266 |
| Needle-64 | zero memory | 0% | 0.016 | 4.3391 |

Interpretation: token salience is meaningful, but currently strongest as a
diagnostic rather than a new model. Copy-32 gives the clean result:
oracle-selected token slots preserve almost full performance using 50% of
per-token memory, while random/stride retention collapse toward 50% accuracy.
KV-16 shows the same pattern in loss but not accuracy, because the full
per-token model's exact-match accuracy is near floor in this retention setup.
Needle-64 is not usable evidence yet because full-memory accuracy is also near
floor. The safe claim is that per-token memory contains useful redundancy, but
dense exact recall still degrades sharply under aggressive compression below
50% memory.

## Current Takeaway

The current evidence supports a cautious statement:

> Under the tested budget and implementation, many-small per-token memory gives better language-modeling loss and the strongest exact synthetic recall among the tested models. RMT-style memory tokens are robustly strong against the custom recurrent baseline on long-gap single-fact needle retrieval, but they do not beat per-token memory in direct synthetic comparisons.

Copy is the cleanest dense exact-recall evidence so far: per-token memory
preserves many simultaneous token identities better than the recurrent memory
designs tested here. Changing recurrent slot allocation, adding a last-token
update, adding explicit associative read/write, and adding RMT-style memory
tokens do not solve copy length 64 under the matched synthetic budget.
Associative memory helps short copy and slightly helps copy length 32, but it
does not improve copy length 64 or random KV retrieval. The apparent
write-normalized needle gap-32 improvement and the raw associative 2x needle-64
improvement do not survive seed checks.

The emerging distinction is:

| Task Type | Winner / Pattern | Meaning |
|---|---|---|
| Dense exact copy | Per-token | Many-small memory is best for storing many token identities. |
| Long-gap single fact | Per-token overall; RMT-style beats custom recurrent | Few-rich memory can work when the task is sparse/salient, but per-token still wins the direct comparison. |
| Random multi-pair KV | Per-token overall; custom recurrent > RMT-style at KV-16 | RMT is not universally better; task structure matters. |

Dense exact recall, such as copy, stresses many simultaneous token identities.
Spreading the same memory budget across token-indexed slots remains much
stronger than concentrating it into recurrent slots. Single-fact retrieval, such
as needle, is different: the custom and associative recurrent variants were weak
or unstable, while RMT-style memory tokens remain strong against custom
recurrent after the learned-initial-memory fairness correction and a focused
three-seed check. Direct comparison still favors per-token memory.

The salience-retention probe adds a mechanism hint: per-token memory has some
compressible redundancy, but the useful tokens are not interchangeable. On
copy-32, oracle-selected top-50% token memory nearly matches full per-token
accuracy, while random and stride retention at the same memory budget perform
much worse.

Next useful experiments:

1. Decide whether RMT-style should be scaled to real-data runs or kept as a
   synthetic published-baseline probe.
2. Begin the paper outline around the refined result: many-small per-token
   memory is stronger for dense exact recall, while RMT-style memory shows that
   few-rich memory can work well for sparse salient retrieval.
