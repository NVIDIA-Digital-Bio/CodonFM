# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from src.data.preprocess.mutation_pred import mlm_process_item, likelihood_process_item
from src.data.preprocess.codon_sequence import process_item as seq_process_item


class DummyTok:
    def __init__(self):
        self.cls_token_id = 0
        self.sep_token_id = 1
        self.pad_token_id = 2
        self.mask_token_id = 3
        self._enc = {'ATG': 10, 'CGT': 11, 'AAA': 12, 'CCC': 13}
    def tokenize(self, s):
        s = s.upper()
        # naive 3-mer split
        return [s[i:i+3] for i in range(0, len(s), 3) if len(s[i:i+3])==3]
    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._enc.get(tokens, 99)
        return [self._enc.get(t, 99) for t in tokens]


def test_mlm_process_item_and_bounds():
    tok = DummyTok()
    out = mlm_process_item('ATGCGT', codon_position=1, ref_codon='ATG', alt_codon='CGT', context_length=8, tokenizer=tok)
    assert out['input_ids'].shape == (8,)
    # Mutation position 1 -> index 2 (with CLS), masked
    assert out['input_ids'][2] == tok.mask_token_id
    assert out['attention_mask'][-1] == 0

    # Out of bounds raises
    with pytest.raises(ValueError):
        mlm_process_item('ATG', codon_position=10, ref_codon='ATG', alt_codon='CGT', context_length=6, tokenizer=tok)


def test_likelihood_process_item_uses_alt():
    tok = DummyTok()
    out = likelihood_process_item('ATGCGT', codon_position=1, ref_codon='ATG', alt_codon='CGT', context_length=8, tokenizer=tok)
    # index 2 should be alt codon id (11 for 'CGT')
    assert out['input_ids'][2] == 11


def test_codon_sequence_process_item_padding():
    tok = DummyTok()
    out = seq_process_item('ATG', context_length=6, tokenizer=tok)
    assert out['input_ids'].shape == (6,)
    # last positions are pad
    assert out['input_ids'][-1] == tok.pad_token_id

