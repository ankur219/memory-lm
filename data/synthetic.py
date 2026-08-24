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
    ):
        super().__init__()
        self.supervise_all_tokens = supervise_all_tokens
        self.vocab = SyntheticVocab(num_keys=num_keys, num_values=num_values)
        self.examples = [
            self._make_example(random.Random(seed + i), num_pairs) for i in range(num_examples)
        ]

    def _make_example(self, rng: random.Random, num_pairs: int) -> Tuple[List[int], int]:
        keys = rng.sample(range(self.vocab.num_keys), k=num_pairs)
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


def collate_batch(batch):
    max_len = max(x[0].numel() for x in batch)
    input_ids = torch.full((len(batch), max_len), PAD, dtype=torch.long)
    targets = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (ids, tgt) in enumerate(batch):
        input_ids[i, : ids.numel()] = ids
        targets[i, : tgt.numel()] = tgt
    return input_ids, targets
