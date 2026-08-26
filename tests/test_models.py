import torch

from models import (
    AssociativeRecurrentMemoryTransformer,
    DecoderOnlyTransformer,
    PerTokenMemoryTransformer,
    RecurrentMemoryTransformer,
    TransformerConfig,
)


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
    for cls in [
        DecoderOnlyTransformer,
        PerTokenMemoryTransformer,
        RecurrentMemoryTransformer,
        AssociativeRecurrentMemoryTransformer,
    ]:
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
    assert_causal(AssociativeRecurrentMemoryTransformer(tiny_config()))


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


def test_associative_recurrent_memory_shape_and_diagnostics():
    cfg = tiny_config(num_memory_tokens=7, recurrent_memory_dim=32, chunk_size=4)
    model = AssociativeRecurrentMemoryTransformer(cfg)
    out = model(torch.randint(0, 64, (2, 11)))
    assert out["memory"].shape == (2, 7, 32)
    assert out["diagnostics"]["read_entropy"] > 0
    assert out["diagnostics"]["write_entropy"] > 0
    assert out["diagnostics"]["candidate_norm"] > 0
    assert out["diagnostics"]["raw_memory_norm"] > 0


def test_associative_recurrent_updates_values_not_keys_per_sequence():
    cfg = tiny_config(num_memory_tokens=4, recurrent_memory_dim=32, chunk_size=4)
    model = AssociativeRecurrentMemoryTransformer(cfg)
    keys_before = model.memory_read_keys.detach().clone()
    out = model(torch.randint(0, 64, (2, 12)))
    assert out["memory"].shape == (2, 4, 32)
    torch.testing.assert_close(model.memory_read_keys.detach(), keys_before)


def test_associative_memory_clip_bounds_slot_norms():
    cfg = tiny_config(num_memory_tokens=4, recurrent_memory_dim=32, chunk_size=4, assoc_memory_clip=1.5)
    model = AssociativeRecurrentMemoryTransformer(cfg)
    out = model(torch.randint(0, 64, (2, 12)))
    slot_norms = out["memory"].norm(dim=-1)
    assert slot_norms.max().item() <= 1.5001


def test_associative_memory_norm_sets_unit_rms():
    cfg = tiny_config(num_memory_tokens=4, recurrent_memory_dim=32, chunk_size=4, assoc_memory_norm=True)
    model = AssociativeRecurrentMemoryTransformer(cfg)
    out = model(torch.randint(0, 64, (2, 12)))
    rms = out["memory"].pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=1e-4)


def test_associative_write_norm_bounds_write_source():
    cfg = tiny_config(num_memory_tokens=4, recurrent_memory_dim=32, chunk_size=4, assoc_write_norm=True)
    model = AssociativeRecurrentMemoryTransformer(cfg)
    out = model(torch.randint(0, 64, (2, 12)))
    # hidden_size is 32, so RMS-normalized vectors should have L2 norm sqrt(32).
    assert abs(out["diagnostics"]["write_source_norm"] - (32 ** 0.5)) < 2e-2


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


def test_recurrent_layerwise_memory_shape():
    cfg = tiny_config(num_memory_tokens=5, per_layer_memory=True, recurrent_memory_dim=48)
    model = RecurrentMemoryTransformer(cfg)
    out = model(torch.randint(0, 64, (2, 11)))
    assert out["memory"].shape == (2, 2, 5, 48)


def test_recurrent_cross_attention_no_collapse():
    cfg = tiny_config(
        num_memory_tokens=4,
        recurrent_memory_dim=16,
        recurrent_update_style="cross_attention",
        chunk_size=4,
        recurrent_learned_initial=True,
    )
    model = RecurrentMemoryTransformer(cfg)
    input_ids = torch.randint(0, 64, (2, 12))
    out = model(input_ids)
    final_mem = out["memory"]
    first_batch_mem = final_mem[0]
    norm_mem = torch.nn.functional.normalize(first_batch_mem, dim=-1)
    cos_sim_matrix = torch.mm(norm_mem, norm_mem.T)
    diag = torch.eye(4, device=cos_sim_matrix.device)
    off_diag_sim = cos_sim_matrix * (1.0 - diag)
    assert (off_diag_sim < 0.99).any(), f"Memory slots collapsed: {cos_sim_matrix}"


def test_recurrent_last_tokens_update_gets_gradients():
    cfg = tiny_config(
        num_memory_tokens=4,
        recurrent_memory_dim=48,
        recurrent_update_style="last_tokens",
        chunk_size=4,
    )
    model = RecurrentMemoryTransformer(cfg)
    input_ids = torch.randint(0, 64, (2, 12))
    targets = torch.randint(0, 64, (2, 12))
    loss = model(input_ids, targets=targets)["loss"]
    loss.backward()
    assert model.candidate_up.weight.grad is not None
    assert model.candidate_up.weight.grad.abs().sum().item() > 0
