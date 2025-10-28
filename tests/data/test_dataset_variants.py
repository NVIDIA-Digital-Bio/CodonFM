# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.data.codon_memmap_dataset import CodonMemmapDataset
from src.data.codon_bert_dataset import CodonBertDataset


def _write_memmap_dir(tmpdir):
    d = str(tmpdir)
    # Minimal metadata with one chunk and larger arrays to ensure non-empty splits
    seq = np.arange(300, dtype=np.int32)
    idx = np.array([
        [0, 50, 1],   # start, end, taxid
        [50, 100, 2],
        [100, 150, 3],
        [150, 200, 4],
        [200, 250, 5],
        [250, 300, 6],
    ], dtype=np.int32)
    os.makedirs(os.path.join(d, 'data'), exist_ok=True)
    seq_path = os.path.join('data', 'seq.memmap')
    idx_path = os.path.join('data', 'idx.memmap')

    # Create memmaps
    seq_mm = np.memmap(os.path.join(d, seq_path), dtype='int32', mode='w+', shape=seq.shape)
    seq_mm[:] = seq[:]
    del seq_mm
    idx_mm = np.memmap(os.path.join(d, idx_path), dtype='int32', mode='w+', shape=idx.shape)
    idx_mm[:] = idx[:]
    del idx_mm

    metadata = {
        'chunks': [{
            'sequences': {'path': seq_path, 'dtype': 'int32', 'shape': list(seq.shape)},
            'index': {'path': idx_path, 'dtype': 'int32', 'shape': list(idx.shape)},
        }],
        'file_metadata': [{'file_name': 'grp0.bin'}],
    }
    with open(os.path.join(d, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    return d


def _dummy_tokenizer():
    class T:
        special_tokens = ['<CLS>', '<SEP>', '<UNK>', '<PAD>', '<MASK>']
        @property
        def vocab_size(self):
            return 69
    return T()


def test_codon_memmap_dataset_smoke(tmpdir):
    data_dir = _write_memmap_dir(tmpdir)
    tok = _dummy_tokenizer()

    def process_item(tokenizer, sequence_tokens, context_length, codon_weights=None):
        # Return minimal structure the downstream expects
        return {'input_ids': sequence_tokens[: min(8, len(sequence_tokens))]}

    ds = CodonMemmapDataset(
        data_path=data_dir,
        tokenizer=tok,
        context_length=8,
        context_overlap=0,
        pretraining_task='mlm',
        # Ensure non-empty splits with tiny data (3 sequences => plenty of subseqs)
        train_val_test_ratio=[0.5, 0.25, 0.25],
        process_item=process_item,
        min_seq_length=1,
        max_seq_length=100,
        split_name_prefix='test',
        seed=123,
    )

    assert len(ds) > 0
    item = ds[0]
    assert 'input_ids' in item

    # Split copies
    train = ds.get_train(process_item)
    val = ds.get_validation(process_item)
    test = ds.get_test(process_item)
    assert len(train) + len(val) + len(test) == len(ds.train_indices) + len(ds.val_indices) + len(ds.test_indices)


def test_codon_bert_dataset_basic(tmpdir):
    df = pd.DataFrame({
        'id': ['a', 'b', 'c', 'd'],
        'value': [0.1, 0.2, 0.3, 0.4],
        'ref_seq': ['ATG', 'AUG', 'TTT', 'UUU'],
        'split': ['train', 'val', 'test', 'train'],
    })
    csv_path = os.path.join(str(tmpdir), 'data.csv')
    df.to_csv(csv_path, index=False)

    def process_item(sequence, tokenizer):
        return {'input_ids': np.array([1,2,3])}

    tok = _dummy_tokenizer()
    ds_all = CodonBertDataset(csv_path, tokenizer=tok, process_item=process_item, split_name='all')
    assert len(ds_all) == 4

    ds_train = ds_all.get_train()
    ds_val = ds_all.get_validation()
    ds_test = ds_all.get_test()
    assert len(ds_train) == 2 and len(ds_val) == 1 and len(ds_test) == 1
    item = ds_all[0]
    assert 'input_ids' in item and 'labels' in (k.lower() for k in item.keys())


