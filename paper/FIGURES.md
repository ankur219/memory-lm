# Figure Plan

## Figure 1: Memory Allocation Diagram

Purpose: make the core comparison visible before the results.

Panels:

- Full KV cache: one key and value per token per layer.
- Per-token compressed memory: many token-indexed states, each smaller.
- Custom recurrent memory: fixed recurrent slots updated per chunk.
- RMT-style memory tokens: memory tokens prepended/appended around chunks.

Message: equal memory-float budgets can produce very different allocation
shapes.

## Figure 2: Real-Data LM Table

Use the 35M TinyStories and WikiText-103 results from `RESULTS.md`.

Columns:

- Dataset
- Model
- Validation loss
- Perplexity
- Parameters
- Persistent memory floats
- Peak VRAM

Main message: per-token memory beats custom recurrent memory at the same
compressed-memory budget on both real datasets.

## Figure 3: Synthetic Recall vs Length

Use copy and needle results from `RESULTS.md`.

Suggested layout:

- Left: copy accuracy vs length.
- Right: needle accuracy vs gap.
- Lines: baseline, per-token, custom recurrent, RMT-style where available.

Main message: per-token is stable for dense exact recall; RMT-style is strong
for sparse long-gap needle versus custom recurrent.

## Figure 4: RMT vs Per-Token Direct Comparison

Use the three-seed direct comparison table.

Suggested layout:

- Bar chart with mean accuracy and error bars.
- Tasks: copy-32, copy-64, needle-32, needle-64, random KV-16.
- Models: per-token, RMT-style.

Main message: RMT improves over custom recurrent on needle, but direct
comparison still favors per-token.

## Figure 5: Memory-Budget Curve

Status: planned.

Compare model quality as persistent-memory budget varies:

- 1x full KV budget.
- 1/2x.
- 1/4x.
- 1/8x.

Candidate y-axes:

- LM validation loss.
- Copy/needle/KV recall.
- Tokens/sec.

Main message: turn the paper from a single-budget comparison into a scaling
story.

