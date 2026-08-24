"""Synthetic memory datasets.

The first milestone implements a compact key-value retrieval task. Sequences
look like:

    <BOS> key value key value ... <QUERY> key <ANSWER> value <EOS>

By default the loss is applied only to the answer value token. For easier
optimization, callers can supervise all next-token positions while still
evaluating answer-token accuracy separately.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


PAD = 0
BOS = 1
EOS = 2
QUERY = 3
ANSWER = 4
COPY = 5
NEEDLE = 6
KEY_OFFSET = 10


@dataclass
class SyntheticVocab:
    num_keys: int = 64
    num_values: int = 100

    @property
    def value_offset(self) -> int:
        return KEY_OFFSET + self.num_keys

    @property
    def size(self) -> int:
        return self.value_offset + self.num_values

    def key_token(self, key_id: int) -> int:
        return KEY_OFFSET + key_id

    def value_token(self, value_id: int) -> int:
        return self.value_offset + value_id


class KeyValueRetrievalDataset(Dataset):
    def __init__(
        self,
        num_examples: int = 1000,
        num_pairs: int = 12,
        num_keys: int = 64,
        num_values: int = 100,
        seed: int = 0,
        supervise_all_tokens: bool = False,
        value_mode: str = "random",
    ):
        super().__init__()
        if value_mode not in {"random", "identity", "shifted"}:
            raise ValueError("value_mode must be 'random', 'identity', or 'shifted'")
        self.supervise_all_tokens = supervise_all_tokens
        self.value_mode = value_mode
        self.vocab = SyntheticVocab(num_keys=num_keys, num_values=num_values)
        self.examples = [
            self._make_example(random.Random(seed + i), num_pairs) for i in range(num_examples)
        ]

    def _make_example(self, rng: random.Random, num_pairs: int) -> Tuple[List[int], int]:
        keys = rng.sample(range(self.vocab.num_keys), k=num_pairs)
        if self.value_mode == "identity":
            values = [key % self.vocab.num_values for key in keys]
        elif self.value_mode == "shifted":
            offset = rng.randrange(self.vocab.num_values)
            values = [(key + offset) % self.vocab.num_values for key in keys]
        else:
            values = [rng.randrange(self.vocab.num_values) for _ in range(num_pairs)]
        pairs: Dict[int, int] = dict(zip(keys, values))
        query_key = rng.choice(keys)

        tokens = [BOS]
        for key in keys:
            tokens.append(self.vocab.key_token(key))
            tokens.append(self.vocab.value_token(pairs[key]))
        tokens.extend([QUERY, self.vocab.key_token(query_key), ANSWER, self.vocab.value_token(pairs[query_key]), EOS])
        answer_index = len(tokens) - 2
        return tokens, answer_index

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        tokens, answer_index = self.examples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        if self.supervise_all_tokens:
            # Dense next-token supervision teaches the model the sequence
            # grammar. Answer-only accuracy is still computed with a separate
            # answer mask by the sweep script.
            targets = torch.tensor(tokens[1:], dtype=torch.long)
        else:
            targets = torch.full_like(input_ids, -100)
            # Standard next-token LM alignment: logits at position
            # answer_index - 1 must predict the answer token at answer_index.
            targets[answer_index - 1] = tokens[answer_index]
        return input_ids, targets


class CopyDataset(Dataset):
    """Exact copying task.

    Each example is:

        <BOS> random_tokens <COPY> random_tokens <EOS>

    Training can supervise either every next-token position or only the copied
    output span. Evaluation should measure only the copied span.
    """

    def __init__(
        self,
        num_examples: int = 1000,
        copy_length: int = 16,
        vocab_tokens: int = 64,
        seed: int = 0,
        supervise_all_tokens: bool = True,
    ):
        super().__init__()
        self.copy_length = int(copy_length)
        self.vocab_tokens = int(vocab_tokens)
        self.supervise_all_tokens = supervise_all_tokens
        self.vocab_size = KEY_OFFSET + self.vocab_tokens
        self.examples = [
            self._make_example(random.Random(seed + i)) for i in range(num_examples)
        ]

    def _make_example(self, rng: random.Random) -> List[int]:
        payload = [KEY_OFFSET + rng.randrange(self.vocab_tokens) for _ in range(self.copy_length)]
        return [BOS] + payload + [COPY] + payload + [EOS]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        tokens = self.examples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        if self.supervise_all_tokens:
            targets = torch.tensor(tokens[1:], dtype=torch.long)
        else:
            targets = torch.full_like(input_ids, -100)
            copy_start = 1 + self.copy_length
            targets[copy_start : copy_start + self.copy_length] = torch.tensor(
                tokens[copy_start + 1 : copy_start + 1 + self.copy_length],
                dtype=torch.long,
            )
        return input_ids, targets


class NeedleDataset(Dataset):
    """Needle-in-context retrieval.

    Each example places one value token after a <NEEDLE> marker, surrounds it
    with random filler, and asks for the value near the end:

        <BOS> filler <NEEDLE> value filler <QUERY> <ANSWER> value <EOS>

    The gap after the needle controls how long the model must preserve the
    exact value.
    """

    def __init__(
        self,
        num_examples: int = 1000,
        prefix_length: int = 8,
        gap_length: int = 32,
        vocab_tokens: int = 64,
        num_values: int = 64,
        seed: int = 0,
        supervise_all_tokens: bool = True,
    ):
        super().__init__()
        self.prefix_length = int(prefix_length)
        self.gap_length = int(gap_length)
        self.vocab_tokens = int(vocab_tokens)
        self.num_values = int(num_values)
        self.supervise_all_tokens = supervise_all_tokens
        self.value_offset = KEY_OFFSET + self.vocab_tokens
        self.vocab_size = self.value_offset + self.num_values
        self.examples = [
            self._make_example(random.Random(seed + i)) for i in range(num_examples)
        ]

    def _filler_token(self, rng: random.Random) -> int:
        return KEY_OFFSET + rng.randrange(self.vocab_tokens)

    def _value_token(self, value_id: int) -> int:
        return self.value_offset + value_id

    def _make_example(self, rng: random.Random) -> Tuple[List[int], int]:
        value_id = rng.randrange(self.num_values)
        value = self._value_token(value_id)
        prefix = [self._filler_token(rng) for _ in range(self.prefix_length)]
        gap = [self._filler_token(rng) for _ in range(self.gap_length)]
        tokens = [BOS] + prefix + [NEEDLE, value] + gap + [QUERY, ANSWER, value, EOS]
        answer_index = len(tokens) - 2
        return tokens, answer_index

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        tokens, answer_index = self.examples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        if self.supervise_all_tokens:
            targets = torch.tensor(tokens[1:], dtype=torch.long)
        else:
            targets = torch.full_like(input_ids, -100)
            targets[answer_index - 1] = tokens[answer_index]
        return input_ids, targets


def collate_batch(batch):
    max_len = max(x[0].numel() for x in batch)
    input_ids = torch.full((len(batch), max_len), PAD, dtype=torch.long)
    targets = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (ids, tgt) in enumerate(batch):
        input_ids[i, : ids.numel()] = ids
        targets[i, : tgt.numel()] = tgt
    return input_ids, targets
