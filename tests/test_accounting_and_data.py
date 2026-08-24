import pytest

from data.synthetic import ANSWER, COPY, KEY_OFFSET, CopyDataset, KeyValueRetrievalDataset
from data.text import BYTE_VOCAB_SIZE, ByteTokenizer, TiktokenTokenizer, build_lm_datasets
from evaluation.efficiency import (
    matched_recurrent_dim_for_per_token,
    matched_recurrent_tokens_for_per_token,
    parameter_breakdown,
    per_token_memory_budget,
    recurrent_memory_budget,
)
from models import PerTokenMemoryTransformer, TransformerConfig
from training.trainer import generate_text_sample, maybe_print_generation_sample


def test_key_value_dataset_marks_only_answer_target():
    ds = KeyValueRetrievalDataset(num_examples=4, num_pairs=3, seed=123)
    input_ids, targets = ds[0]
    supervised = (targets != -100).nonzero().flatten()
    assert supervised.numel() == 1
    answer_prompt_pos = supervised.item()
    assert input_ids[answer_prompt_pos].item() == ANSWER


def test_shifted_key_value_dataset_uses_one_offset_per_example():
    ds = KeyValueRetrievalDataset(
        num_examples=4,
        num_pairs=5,
        num_keys=16,
        num_values=16,
        seed=123,
        value_mode="shifted",
    )
    input_ids, _ = ds[0]
    value_offset = ds.vocab.value_offset
    offsets = []
    # Tokens are <BOS> key value key value ...
    for pos in range(1, 1 + 2 * 5, 2):
        key = int(input_ids[pos].item()) - KEY_OFFSET
        value = int(input_ids[pos + 1].item()) - value_offset
        offsets.append((value - key) % ds.vocab.num_values)
    assert len(set(offsets)) == 1


def test_copy_dataset_marks_copy_span_when_sparse():
    ds = CopyDataset(num_examples=4, copy_length=6, seed=123, supervise_all_tokens=False)
    input_ids, targets = ds[0]
    marker_pos = (input_ids == COPY).nonzero().flatten().item()
    supervised = (targets != -100).nonzero().flatten()
    assert supervised.numel() == 6
    assert supervised[0].item() == marker_pos


def test_parameter_breakdown_has_requested_buckets():
    cfg = TransformerConfig(vocab_size=64, hidden_size=32, num_layers=1, num_heads=4, context_length=16, memory_dim=16)
    model = PerTokenMemoryTransformer(cfg)
    counts = parameter_breakdown(model)
    assert counts["total"] > 0
    assert counts["attention"] > 0
    assert counts["mlp"] > 0
    assert counts["embedding"] > 0


def test_memory_budget_formulas():
    cfg = TransformerConfig(hidden_size=128, num_layers=2, num_heads=4, memory_dim=32, num_memory_tokens=8)
    per_token = per_token_memory_budget(cfg, sequence_length=64)
    recurrent = recurrent_memory_budget(cfg)
    assert per_token["floats"] == 64 * 2 * 2 * 32
    assert recurrent["floats"] == 8 * 128
    assert matched_recurrent_tokens_for_per_token(cfg, 64) == 64
    assert matched_recurrent_dim_for_per_token(cfg, 64, num_memory_tokens=8) == 1024


def test_byte_text_dataset_builds_next_token_blocks():
    text = "A real text stream for language modeling. " * 20
    train_ds, val_ds, tokenizer = build_lm_datasets(
        text, block_size=16, val_fraction=0.2, tokenizer_config={"kind": "byte"}
    )
    input_ids, targets = train_ds[0]
    assert tokenizer.vocab_size == BYTE_VOCAB_SIZE
    assert input_ids.shape == (16,)
    assert targets.shape == (16,)
    assert targets[0].item() == input_ids[1].item()


def test_text_dataset_uses_non_overlapping_blocks_by_default():
    text = "A real text stream for language modeling. " * 20
    train_ds, _, _ = build_lm_datasets(
        text, block_size=16, val_fraction=0.2, tokenizer_config={"kind": "byte"}
    )
    first_ids, _ = train_ds[0]
    second_ids, _ = train_ds[1]
    assert train_ds.stride == 16
    assert first_ids[-1].item() != second_ids[0].item()


def test_tiktoken_tokenizer_round_trip():
    try:
        tokenizer = TiktokenTokenizer("gpt2")
    except Exception as exc:
        pytest.skip(f"tiktoken encoding cache is not available offline: {exc}")
    tokens = tokenizer.encode("hello research memory", add_eos=True)
    assert tokenizer.vocab_size == 50257
    assert tokens[-1] == tokenizer.eos_token
    assert "research" in tokenizer.decode(tokens[:-1])


def test_generation_sample_uses_tokenizer():
    cfg = {
        "model": {
            "context_length": 16,
        },
        "generation": {
            "max_new_tokens": 4,
            "temperature": 1.0,
            "top_k": 10,
        },
    }
    model_cfg = TransformerConfig(
        vocab_size=BYTE_VOCAB_SIZE,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        context_length=16,
        memory_dim=16,
    )
    model = PerTokenMemoryTransformer(model_cfg)
    sample = generate_text_sample(model, ByteTokenizer(), "Once", cfg, device=next(model.parameters()).device)
    assert isinstance(sample, str)
    assert sample.startswith("Once")


def test_generation_sample_respects_every_steps(capsys):
    cfg = {
        "model": {"context_length": 16},
        "generation": {
            "enabled": True,
            "prompt": "Once",
            "max_new_tokens": 2,
            "every_steps": 100,
            "print_at_step_one": False,
        },
    }
    model_cfg = TransformerConfig(
        vocab_size=BYTE_VOCAB_SIZE,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        context_length=16,
        memory_dim=16,
    )
    model = PerTokenMemoryTransformer(model_cfg)
    maybe_print_generation_sample(model, ByteTokenizer(), cfg, step=50, device=next(model.parameters()).device)
    assert capsys.readouterr().out == ""
    maybe_print_generation_sample(model, ByteTokenizer(), cfg, step=100, device=next(model.parameters()).device)
    out = capsys.readouterr().out
    assert "sample step 0100" in out
    assert "fed prompt:" in out
    assert "predicted continuation:" in out


def test_real_trainer_cycles_until_max_steps(tmp_path):
    from training.trainer import train_language_model

    cfg = {
        "seed": 0,
        "model_name": "per_token",
        "device": "cpu",
        "batch_size": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "max_steps": 5,
        "num_epochs": 1,
        "eval_every": 100,
        "val_fraction": 0.2,
        "block_stride": None,
        "log_path": str(tmp_path / "run.jsonl"),
        "csv_path": str(tmp_path / "run.csv"),
        "tokenizer": {"kind": "byte"},
        "data": {
            "source": "text_file",
            "path": "data/sample_real_text.txt",
            "max_chars": 1200,
        },
        "model": {
            "vocab_size": BYTE_VOCAB_SIZE,
            "hidden_size": 32,
            "num_layers": 1,
            "num_heads": 4,
            "context_length": 32,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "tie_embeddings": True,
            "memory_dim": 16,
            "num_memory_tokens": 4,
            "recurrent_memory_dim": None,
            "recurrent_update_rank": 4,
            "recurrent_compressed_attention": True,
            "recurrent_learned_initial": False,
            "param_padding": 0,
            "chunk_size": 16,
        },
    }
    summary = train_language_model(cfg)
    assert summary["final_metrics"]["step"] == 5
    assert summary["final_metrics"]["tokens_processed"] == 5 * 4 * 32
