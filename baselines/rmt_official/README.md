# Official RMT Baseline Notes

This directory records how the repo's `rmt` model is tied to the public RMT
implementation.

Source inspected:

```text
https://github.com/booydar/recurrent-memory-transformer
commit: 9d0ebe1778687995697fe68e886bc1dcf0e45e1c
```

Relevant official implementation:

```text
modeling_rmt/experimental.py
class RMTDecoderLMHeadMultiSeg
```

The important language-modeling mechanics are:

1. Create learned memory token embeddings.
2. Split the input into segments.
3. For each segment, build `[memory, token_embeddings, memory]`.
4. Mask labels on memory positions.
5. Run the base Transformer.
6. Replace memory with the final hidden states at the trailing memory positions.
7. Concatenate logits from text positions only.

Our in-harness `models/rmt_memory.py` implements those mechanics with this
repo's small decoder backbone rather than HuggingFace, so RMT can be compared
under the same tokenizer, dataset, optimizer, logging, parameter counting, and
memory-budget calculation as the baseline, per-token, and custom recurrent
models.

Paper wording should call this:

```text
RMT-style baseline adapted from the public RMT implementation
```

Do not call it an exact reproduction of the original published experiments
unless we later run the upstream repository's full training stack directly.
