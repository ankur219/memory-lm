import torch

from models import DecoderOnlyTransformer, PerTokenMemoryTransformer, RecurrentMemoryTransformer, TransformerConfig


def tiny_config(**overrides):
    cfg = dict(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        context_length=16,
        mlp_ratio=2.0,
        dropout=0.0,
        tie_embeddings=True,
        memory_dim=16,
        num_memory_tokens=4,
        chunk_size=8,
    )
    cfg.update(overrides)
    return TransformerConfig(**cfg)


def test_model_output_shapes():
    input_ids = torch.randint(0, 64, (3, 12))
    targets = torch.randint(0, 64, (3, 12))
    for cls in [DecoderOnlyTransformer, PerTokenMemoryTransformer, RecurrentMemoryTransformer]:
        model = cls(tiny_config())
        out = model(input_ids, targets=targets)
        assert out["logits"].shape == (3, 12, 64)
        assert out["loss"].ndim == 0


def assert_causal(model):
    model.eval()
    input_ids = torch.randint(0, 64, (1, 12))
    changed = input_ids.clone()
    changed[:, 9:] = torch.randint(0, 64, (1, 3))
    with torch.no_grad():
        a = model(input_ids)["logits"][:, :8]
        b = model(changed)["logits"][:, :8]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_causal_behavior_for_all_models():
    assert_causal(DecoderOnlyTransformer(tiny_config()))
    assert_causal(PerTokenMemoryTransformer(tiny_config()))
    assert_causal(RecurrentMemoryTransformer(tiny_config()))


def test_per_token_cache_uses_compressed_width():
    model = PerTokenMemoryTransformer(tiny_config(memory_dim=16))
    out = model(torch.randint(0, 64, (2, 10)))
    key, value = out["cache"][0]
    assert key.shape == (2, 4, 10, 4)
    assert value.shape == (2, 4, 10, 4)


def test_recurrent_memory_shape():
    cfg = tiny_config(num_memory_tokens=5, chunk_size=4)
    model = RecurrentMemoryTransformer(cfg)
    out = model(torch.randint(0, 64, (2, 11)))
    assert out["memory"].shape == (2, 5, 32)


def test_recurrent_rich_memory_shape():
    cfg = tiny_config(num_memory_tokens=5, recurrent_memory_dim=96, chunk_size=4)
    model = RecurrentMemoryTransformer(cfg)
    out = model(torch.randint(0, 64, (2, 11)))
    assert out["memory"].shape == (2, 5, 96)


def test_recurrent_memory_update_gets_gradients_across_chunks():
    cfg = tiny_config(num_memory_tokens=4, recurrent_memory_dim=48, chunk_size=4)
    model = RecurrentMemoryTransformer(cfg)
    input_ids = torch.randint(0, 64, (2, 12))
    targets = torch.randint(0, 64, (2, 12))
    loss = model(input_ids, targets=targets)["loss"]
    loss.backward()
    assert model.candidate_up.weight.grad is not None
    assert model.candidate_up.weight.grad.abs().sum().item() > 0
    assert model.gate_up.weight.grad is not None
    assert model.gate_up.weight.grad.abs().sum().item() > 0


def test_parameter_padding_gets_gradients_when_enabled():
    cfg = tiny_config(param_padding=17)
    model = PerTokenMemoryTransformer(cfg)
    input_ids = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    loss = model(input_ids, targets=targets)["loss"]
    loss.backward()
    assert model.param_padding.grad is not None
    assert model.param_padding.grad.abs().sum().item() > 0
