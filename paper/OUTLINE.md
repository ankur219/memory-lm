# Paper Outline

Working title:

```text
Many Small Memories or Few Rich Memories?
A Matched-Budget Study of Memory Allocation in Small Language Models
```

## Core Claim

Under matched persistent-memory budgets, spreading memory across token-indexed
states is the strongest strategy tested for exact recall and language modeling.
Few-rich recurrent memory can work for sparse, salient retrieval when implemented
with RMT-style memory tokens, but it does not beat per-token memory in direct
synthetic comparisons.

## Abstract Shape

1. Long-context models need memory, but memory can be allocated in different
   ways.
2. Compare many-small per-token memory against few-rich recurrent memory under
   matched persistent-memory budgets.
3. Evaluate on real language modeling and synthetic recall probes.
4. Per-token memory gives stronger LM loss and exact recall.
5. RMT-style memory tokens rescue sparse long-gap needle retrieval versus a
   custom recurrent baseline, showing mechanism matters, but per-token remains
   stronger overall.

## Section Plan

1. **Introduction**
   - State the memory-allocation question.
   - Motivate matched persistent-memory budget rather than raw context length.
   - Preview the main result: allocation shape matters.

2. **Memory-Budget Framing**
   - Define persistent memory floats.
   - Separate parameters from per-sequence memory.
   - Explain full KV, compressed per-token KV, recurrent state, and RMT memory
     tokens in one common accounting frame.

3. **Models**
   - Baseline Transformer.
   - Per-token compressed memory.
   - Custom recurrent memory.
   - Associative recurrent diagnostic variant.
   - RMT-style published recurrent baseline.

4. **Experimental Setup**
   - Real-data LM: TinyStories and WikiText-103 at 7M and 35M scale.
   - Synthetic tasks: copy, needle, random KV.
   - Matched memory and parameter accounting.
   - Seeds and reporting policy.

5. **Results**
   - Real-data LM: per-token beats recurrent at both scales and datasets.
   - Synthetic exact recall: per-token dominates dense copy and needle.
   - RMT: strong sparse needle retrieval versus custom recurrent, but not
     stronger than per-token.
   - Associative branch: structured read/write and normalization did not produce
     robust improvements.
   - Shape check: fairer recurrent slot allocation improves only slightly.

6. **Discussion**
   - Dense exact recall needs many token-indexed states.
   - Sparse salient retrieval can benefit from RMT-style memory tokens.
   - Recurrent memory failures are not only implementation bugs, but published
     mechanisms show task-dependent strengths.
   - Parameter budget and memory budget must be reported separately.

7. **Limitations**
   - Current main scale is 35M, not 100M+.
   - Real-data results still need more seeds for submission-grade claims.
   - RMT is currently synthetic-only in this repo.
   - Context lengths beyond 128 remain future work.

8. **Conclusion**
   - Matched memory budgets reveal an allocation trade-off.
   - Many-small memory is the safest default for exact recall.
   - Few-rich memory remains useful for sparse retrieval depending on mechanism.

