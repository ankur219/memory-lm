# KVM Reference

The `kvm` model in this repo is adapted from the eager Key-Value Means mixer in:

<https://github.com/recursal/KVM-paper>

The upstream project uses a full RWKV/GPT training stack and optional Triton
kernels. This repo keeps the core KVM mechanism inside the local controlled
Transformer harness:

- fixed-size compressed key/value state per attention layer
- block sliding-window attention over recent tokens
- old window tokens overflow into compressed state
- overflow tokens append while slots remain, then merge into the nearest key
  slot by similarity
- state values are read as accumulated values divided by accumulated counts

Because the state is layerwise and stores both keys and values, memory budget is
counted as:

```text
num_layers * num_memory_tokens * 2(K,V) * recurrent_memory_dim
```

At the synthetic two-layer scale, `128x64` therefore matches the existing
32,768-float recurrent budget.
