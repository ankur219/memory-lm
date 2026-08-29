# Experiment Plan

## Active Experiments

### 1. Multi-Seed 35M Real-Data Checks

Goal: make the main real-data table submission-grade.

Minimum:

- TinyStories 35M: baseline, per-token, recurrent fair shape.
- WikiText-103 35M: baseline, per-token, recurrent fair shape.
- Add at least one more seed before making strong claims.

Current status: seed 0 exists. Additional seeds are planned, not done.

### 2. Memory-Budget Curve

Goal: create the likely main figure.

Budgets:

- 1x full KV memory.
- 1/2x.
- 1/4x.
- 1/8x.

Models:

- Baseline full KV.
- Per-token compressed memory.
- Custom recurrent memory.
- Optional RMT-style memory on synthetic tasks only.

Metrics:

- Validation loss / perplexity for real data.
- Copy, needle, and KV accuracy for synthetic probes.
- Tokens/sec and peak VRAM.

Design rule: keep parameter counts explicit. If exact parameter matching is not
possible at every budget, report the parameter delta rather than hiding it.

### 3. RMT Real-Data Scope Decision

Current recommendation: do not run RMT at 35M real-data scale yet.

Reason:

- RMT is already valuable as a published-style synthetic baseline.
- Scaling it to real data is compute-expensive.
- Direct synthetic comparison shows per-token still wins exact recall.

Run RMT real-data only if the target venue needs a stronger external real-data
baseline.

## Deferred

- Longer contexts: 512, 1024, 2048.
- Additional recurrent shape point.
- 100M-150M scale point.
- Fast-weight / outer-product memory.

