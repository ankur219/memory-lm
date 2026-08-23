"""Real-text language-model datasets.

For real language modeling the preferred tokenizer is tiktoken with a fixed
encoding shared across all model variants. A byte tokenizer remains available
for offline debugging because it has no external package dependency.

The tokenizer object is part of the experiment config and logs, because changing
tokenization changes the task and the embedding parameter count.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset


BYTE_VOCAB_SIZE = 257
EOS_TOKEN = 256


class Tokenizer(Protocol):
    kind: str
    vocab_size: int
    eos_token: int

    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        ...

    def decode(self, tokens: Iterable[int]) -> str:
        ...


@dataclass
class ByteTokenizer:
    kind: str = "byte"
    eos_token: int = EOS_TOKEN
    vocab_size: int = BYTE_VOCAB_SIZE

    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        tokens = list(text.encode("utf-8", errors="replace"))
        if add_eos:
            tokens.append(self.eos_token)
        return tokens

    def decode(self, tokens: Iterable[int]) -> str:
        byte_values = [int(t) for t in tokens if 0 <= int(t) < 256]
        return bytes(byte_values).decode("utf-8", errors="replace")


class TiktokenTokenizer:
    """Thin wrapper around a fixed tiktoken encoding.

    We use encode_ordinary so raw text is tokenized without interpreting special
    strings inside the dataset. EOS is appended explicitly.
    """

    kind = "tiktoken"

    def __init__(self, encoding_name: str = "gpt2", cache_dir: str | Path = "data/tiktoken_cache"):
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError("Install `tiktoken` to use tokenizer.kind=tiktoken.") from exc

        self.encoding_name = encoding_name
        self.cache_dir = str(cache_dir)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", self.cache_dir)
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab
        self.eos_token = self.encoding.eot_token

    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        tokens = self.encoding.encode_ordinary(text)
        if add_eos:
            tokens.append(self.eos_token)
        return tokens

    def decode(self, tokens: Iterable[int]) -> str:
        return self.encoding.decode([int(t) for t in tokens])

    def metadata(self) -> dict:
        return {
            "kind": self.kind,
            "encoding_name": self.encoding_name,
            "cache_dir": self.cache_dir,
            "vocab_size": self.vocab_size,
            "eos_token": self.eos_token,
        }


class TextBlockDataset(Dataset):
    """Fixed-length next-token LM blocks from one long token stream.

    By default stride == block_size, so one epoch is approximately one pass over
    the token stream with non-overlapping training blocks. A smaller stride can
    be used for debugging or data augmentation, but then "one epoch" no longer
    means "one pass over unique tokens."
    """

    def __init__(self, tokens: List[int], block_size: int, stride: Optional[int] = None):
        if len(tokens) < block_size + 1:
            raise ValueError("Need at least block_size + 1 tokens")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = int(block_size)
        self.stride = int(stride or block_size)

    def __len__(self) -> int:
        return 1 + max(0, (self.tokens.numel() - self.block_size - 1) // self.stride)

    def __getitem__(self, idx: int):
        start = idx * self.stride
        chunk = self.tokens[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class MemmapTextBlockDataset(Dataset):
    """Fixed-length LM blocks backed by a token file on disk.

    Full TinyStories is large enough that storing token ids as a Python list is
    wasteful. This dataset reads only the requested block from a uint16 memmap
    and converts that small slice to a torch.long tensor for the model.
    """

    def __init__(self, token_path: str | Path, block_size: int, stride: Optional[int] = None):
        self.token_path = Path(token_path)
        self.tokens = np.memmap(self.token_path, dtype=np.uint16, mode="r")
        if self.tokens.shape[0] < block_size + 1:
            raise ValueError("Need at least block_size + 1 tokens")
        self.block_size = int(block_size)
        self.stride = int(stride or block_size)

    def __len__(self) -> int:
        return 1 + max(0, (self.tokens.shape[0] - self.block_size - 1) // self.stride)

    def __getitem__(self, idx: int):
        start = idx * self.stride
        chunk = np.asarray(self.tokens[start : start + self.block_size + 1], dtype=np.int64)
        chunk = torch.from_numpy(chunk)
        return chunk[:-1], chunk[1:]


def load_text_file(path: str | Path, max_chars: Optional[int] = None) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if max_chars is not None:
        text = text[:max_chars]
    return text


def load_tinystories_text(
    split: str = "train",
    max_examples: Optional[int] = 10_000,
    max_chars: Optional[int] = None,
    cache_dir: Optional[str | Path] = None,
    offline: bool = False,
) -> str:
    """Load TinyStories through Hugging Face datasets."""

    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install `datasets` to load TinyStories from Hugging Face.") from exc

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split=split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        download_mode="reuse_dataset_if_exists",
    )
    pieces = []
    total_chars = 0
    for i, row in enumerate(dataset):
        if max_examples is not None and i >= max_examples:
            break
        story = row.get("text", "")
        pieces.append(story)
        total_chars += len(story)
        if max_chars is not None and total_chars >= max_chars:
            break
    text = "\n\n".join(pieces)
    if max_chars is not None:
        text = text[:max_chars]
    return text


def _limit_label(value: Optional[int]) -> str:
    return "full" if value is None else str(int(value))


def prepare_tinystories_token_cache(
    split: str,
    tokenizer: Tokenizer,
    cache_dir: str | Path = "data/token_cache",
    max_examples: Optional[int] = None,
    max_chars: Optional[int] = None,
    hf_cache_dir: Optional[str | Path] = "data/hf_cache",
    offline: bool = False,
) -> tuple[Path, int]:
    """Tokenize TinyStories into a reusable uint16 binary token file.

    GPT-2/tiktoken ids fit in uint16 because the vocab size is 50,257. The byte
    tokenizer also fits. The file is written incrementally story by story, so
    full-dataset preparation does not need to hold every token in Python memory.
    """

    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("The memmap cache currently expects token ids to fit in uint16.")
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install `datasets` to load TinyStories from Hugging Face.") from exc

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    tokenizer_name = getattr(tokenizer, "encoding_name", tokenizer.kind)
    stem = (
        f"tinystories_{split}_{tokenizer.kind}_{tokenizer_name}_"
        f"examples-{_limit_label(max_examples)}_chars-{_limit_label(max_chars)}"
    )
    token_path = cache_root / f"{stem}.uint16.bin"
    meta_path = cache_root / f"{stem}.meta.json"

    if token_path.exists() and meta_path.exists():
        count = int(np.memmap(token_path, dtype=np.uint16, mode="r").shape[0])
        return token_path, count

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split=split,
        cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
        download_mode="reuse_dataset_if_exists",
    )
    total_chars = 0
    total_tokens = 0
    examples = 0
    tmp_path = token_path.with_suffix(token_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        for i, row in enumerate(dataset):
            if max_examples is not None and i >= max_examples:
                break
            story = row.get("text", "")
            if max_chars is not None:
                remaining = max_chars - total_chars
                if remaining <= 0:
                    break
                story = story[:remaining]
            ids = tokenizer.encode(story + "\n\n", add_eos=False)
            arr = np.asarray(ids, dtype=np.uint16)
            arr.tofile(f)
            total_chars += len(story)
            total_tokens += int(arr.shape[0])
            examples += 1
            if max_chars is not None and total_chars >= max_chars:
                break
        np.asarray([tokenizer.eos_token], dtype=np.uint16).tofile(f)
        total_tokens += 1

    tmp_path.replace(token_path)
    meta_path.write_text(
        (
            "{\n"
            f'  "source": "roneneldan/TinyStories",\n'
            f'  "split": "{split}",\n'
            f'  "tokenizer_kind": "{tokenizer.kind}",\n'
            f'  "tokenizer_name": "{tokenizer_name}",\n'
            f'  "max_examples": {max_examples if max_examples is not None else "null"},\n'
            f'  "max_chars": {max_chars if max_chars is not None else "null"},\n'
            f'  "examples": {examples},\n'
            f'  "chars": {total_chars},\n'
            f'  "tokens": {total_tokens}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    return token_path, total_tokens


def build_tokenizer(config: Optional[dict] = None) -> Tokenizer:
    config = config or {}
    kind = config.get("kind", "tiktoken")
    if kind == "byte":
        return ByteTokenizer()
    if kind == "tiktoken":
        return TiktokenTokenizer(config.get("encoding", "gpt2"), config.get("cache_dir", "data/tiktoken_cache"))
    raise ValueError("tokenizer.kind must be 'tiktoken' or 'byte'")


def tokenizer_metadata(tokenizer: Tokenizer) -> dict:
    if hasattr(tokenizer, "metadata"):
        return tokenizer.metadata()
    return {
        "kind": tokenizer.kind,
        "vocab_size": tokenizer.vocab_size,
        "eos_token": tokenizer.eos_token,
    }


def build_lm_datasets(
    text: str,
    block_size: int,
    val_fraction: float = 0.05,
    tokenizer_config: Optional[dict] = None,
    block_stride: Optional[int] = None,
):
    tokenizer = build_tokenizer(tokenizer_config)
    tokens = tokenizer.encode(text, add_eos=True)
    split = int(len(tokens) * (1.0 - val_fraction))
    split = max(block_size + 1, min(split, len(tokens) - block_size - 1))
    train_ds = TextBlockDataset(tokens[:split], block_size, stride=block_stride)
    val_ds = TextBlockDataset(tokens[split:], block_size, stride=block_stride)
    return train_ds, val_ds, tokenizer


def build_lm_datasets_from_texts(
    train_text: str,
    validation_text: str,
    block_size: int,
    tokenizer_config: Optional[dict] = None,
    block_stride: Optional[int] = None,
):
    tokenizer = build_tokenizer(tokenizer_config)
    train_tokens = tokenizer.encode(train_text, add_eos=True)
    validation_tokens = tokenizer.encode(validation_text, add_eos=True)
    train_ds = TextBlockDataset(train_tokens, block_size, stride=block_stride)
    val_ds = TextBlockDataset(validation_tokens, block_size, stride=block_stride)
    return train_ds, val_ds, tokenizer


def build_tinystories_memmap_datasets(config: dict, block_size: int, block_stride: Optional[int] = None):
    """Build train/validation datasets from disk-backed TinyStories token caches."""

    tokenizer = build_tokenizer(config.get("tokenizer"))
    data_cfg = config.get("data", {})
    val_cfg = config.get("validation_data", {})
    train_path, _ = prepare_tinystories_token_cache(
        split=data_cfg.get("split", "train"),
        tokenizer=tokenizer,
        cache_dir=data_cfg.get("token_cache_dir", "data/token_cache"),
        max_examples=data_cfg.get("max_examples"),
        max_chars=data_cfg.get("max_chars"),
        hf_cache_dir=data_cfg.get("cache_dir", "data/hf_cache"),
        offline=bool(data_cfg.get("offline", False)),
    )
    val_path, _ = prepare_tinystories_token_cache(
        split=val_cfg.get("split", "validation"),
        tokenizer=tokenizer,
        cache_dir=val_cfg.get("token_cache_dir", data_cfg.get("token_cache_dir", "data/token_cache")),
        max_examples=val_cfg.get("max_examples"),
        max_chars=val_cfg.get("max_chars"),
        hf_cache_dir=val_cfg.get("cache_dir", data_cfg.get("cache_dir", "data/hf_cache")),
        offline=bool(val_cfg.get("offline", data_cfg.get("offline", False))),
    )
    train_ds = MemmapTextBlockDataset(train_path, block_size, stride=block_stride)
    val_ds = MemmapTextBlockDataset(val_path, block_size, stride=block_stride)
    return train_ds, val_ds, tokenizer
