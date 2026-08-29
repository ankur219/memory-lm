# Related Work Notes

This file is a drafting scaffold, not final prose.

## Segment Recurrence and Compressed Context

- Transformer-XL: segment-level recurrence, cached hidden states, long-range
  dependency modeling.
- Compressive Transformer: adds compressed memory for older activations.

Use for: historical framing that recurrence/compression are established ways to
extend context beyond full attention.

## Recurrent Memory Tokens

- Recurrent Memory Transformer (RMT): memory tokens carried between segments.
- ARMT / associative recurrent variants: more structured memory mechanisms.
- TransformerFAM: feedback-style recurrent memory in transformers.

Use for: external baselines and why custom recurrent memory alone is not enough
to support a broad claim about all few-rich memory.

## Retrieval and External Memory

- Memorizing Transformers: retrieval over stored key-value memories.
- Key-Value Means: interpolates between full KV-style memory and compressed
  recurrent summaries.
- Melodi / summary-token memory: memory represented through learned or selected
  summary tokens.

Use for: positioning the many-small vs few-rich allocation question as a memory
representation trade-off rather than just a model variant comparison.

## Infinite or Compressive Attention

- Infini-Transformer / Infini-attention: bounded memory and long-context
  extrapolation through compressive attention-style state.

Use for: newer long-context memory framing and limitations of fixed-size memory.

## How This Paper Should Differ

The paper should not claim that recurrent memory is universally bad. The current
evidence supports a narrower and stronger claim:

```text
Under matched persistent-memory budgets, token-indexed memory is the strongest
tested allocation for exact recall and LM loss, while RMT-style recurrent memory
shows that few-rich memory can still be effective for sparse salient retrieval.
```

The novelty is the matched-budget framing, the allocation comparison, and the
careful separation between per-sequence memory and learned parameters.

