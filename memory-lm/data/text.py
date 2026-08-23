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
