# Experiment Plan

## Active Experiments

### 0. Runs In Progress

- `real_35m_seed1.out`: second seed for the 35M real-data tables.
- `long_synthetic.out`: longer-context synthetic copy/needle run.

When these finish, update `RESULTS.md`, `paper/DRAFT.md`, and `paper/TABLES.md`
before launching another large experiment.

### 1. Multi-Seed 35M Real-Data Checks

Goal: make the main real-data table submission-grade.

Minimum:

- TinyStories 35M: baseline, per-token, recurrent fair shape.
- WikiText-103 35M: baseline, per-token, recurrent fair shape.
- Add at least one more seed before making strong claims.

Current status: seed 0 exists; seed 1 is running.

### 2. Long-Context Synthetic Checks

Goal: address the "context length is too short" reviewer concern without moving
immediately to expensive long-context real-data language modeling.

Current run:

- Copy lengths: 128, 256, 512.
- Needle gaps: 128, 256, 512, 1024.
- Models: baseline, per-token, custom recurrent, RMT-style.

These should be treated as reviewer-defense evidence, not as a replacement for
future real-data long-context runs.

### 3. Memory-Budget Curve

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

### 4. RMT Real-Data Scope Decision

Current recommendation: do not run RMT at 35M real-data scale yet.

Reason:

- RMT is already valuable as a published-style synthetic baseline.
- Scaling it to real data is compute-expensive.
- Direct synthetic comparison shows per-token still wins exact recall.

Run RMT real-data only if the target venue needs a stronger external real-data
baseline.

## Deferred

- Real-data longer contexts: 512, 1024, 2048.
- Actual ARMT/KVM code reproduction.
- 70M-100M scale point.
- Additional recurrent shape point.
- 100M-150M scale point.
- Fast-weight / outer-product memory.
