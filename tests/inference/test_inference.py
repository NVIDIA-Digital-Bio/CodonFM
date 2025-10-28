# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from src.inference.base import BaseInference
from src.inference.encodon import EncodonInference
from src.inference.task_types import TaskTypes
from src.inference.model_outputs import (
    MaskedLMOutput,
    MutationPredictionOutput,
    FitnessPredictionOutput,
    EmbeddingOutput,
    DownstreamPredictionOutput,
)
from src.data.metadata import MetadataFields


class DummyInference(BaseInference):
    def configure_model(self):
        self.model = MagicMock()
    def _predict_step(self, batch, batch_idx):
        return {"ok": True, "batch_idx": batch_idx}


def test_base_inference_predict_step_increments_counter():
    inf = DummyInference(model_path="/tmp/dummy.ckpt", task_type="dummy")
    before = inf.prediction_counter
    out = inf.predict_step({}, 0)
    assert out["ok"] is True
    assert inf.prediction_counter == before + 1


@pytest.fixture
def mock_ckpt(tmp_path):
    # Minimal lightning-like checkpoint content
    ckpt = {
        "hyper_parameters": {
            'vocab_size': 69,
            'hidden_size': 64,
            'num_hidden_layers': 1,
            'num_attention_heads': 8,
            'intermediate_size': 128,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.0,
            'attention_probs_dropout_prob': 0.0,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'pad_token_id': 3,
            'position_embedding_type': 'rotary',
            'classifier_dropout': 0.0,
            'rotary_theta': 1e4,
            'ignore_index': -100,
            'loss_type': 'regression',
            'lora': False,
            'lora_alpha': 32.0,
            'lora_r': 16,
            'lora_dropout': 0.0,
            'finetune_strategy': 'full',
            'num_classes': 2,
            'use_downstream_head': False,
            'cross_attention_hidden_dim': 32,
            'cross_attention_num_heads': 4,
            'max_position_embeddings': 256,
        },
        "state_dict": {},
    }
    p = tmp_path / "model.ckpt"
    torch.save(ckpt, p)
    return str(p)


def _make_lm_batch(bs=2, seqlen=8, vocab_size=69):
    return {
        MetadataFields.INPUT_IDS: torch.randint(0, vocab_size, (bs, seqlen)),
        MetadataFields.ATTENTION_MASK: torch.ones(bs, seqlen),
        MetadataFields.LABELS: torch.randint(0, vocab_size, (bs, seqlen)),
        MetadataFields.INPUT_MASK: torch.ones(bs, seqlen, dtype=torch.bool),
    }


@patch("src.inference.encodon.EncodonPL")
def test_encodon_inference_configure_and_mlm(mock_pl, mock_ckpt):
    # Make forward return an object with .logits
    class Out:
        def __init__(self, logits):
            self.logits = logits
    instance = MagicMock()
    instance.side_effect = lambda batch: Out(torch.randn(batch[MetadataFields.INPUT_IDS].shape[0], batch[MetadataFields.INPUT_IDS].shape[1], 69))
    mock_pl.return_value = instance

    inf = EncodonInference(model_path=mock_ckpt, task_type=TaskTypes.MASKED_LANGUAGE_MODELING)
    inf.configure_model()
    batch = _make_lm_batch()
    out = inf._predict_step(batch, 0)
    assert isinstance(out, MaskedLMOutput)
    assert out.preds.ndim == 2  # collapsed by mask


@patch("src.inference.encodon.EncodonPL")
def test_encodon_inference_mutation(mock_pl, mock_ckpt):
    from types import SimpleNamespace
    instance = MagicMock()
    # logits for batch=2, seq=5, vocab=69
    instance.return_value = SimpleNamespace(logits=torch.randn(2, 5, 69))
    mock_pl.return_value = instance

    inf = EncodonInference(model_path=mock_ckpt, task_type=TaskTypes.MUTATION_PREDICTION)
    inf.configure_model()
    batch = {
        MetadataFields.INPUT_IDS: torch.randint(0, 69, (2, 5)),
        MetadataFields.ATTENTION_MASK: torch.ones(2, 5),
        MetadataFields.REF_CODON_TOKS: torch.tensor([1, 2]),
        MetadataFields.ALT_CODON_TOKS: torch.tensor([3, 4]),
        MetadataFields.MUTATION_TOKEN_IDX: torch.tensor([1, 2]),
    }
    out = inf._predict_step(batch, 0)
    assert isinstance(out, MutationPredictionOutput)
    assert out.ref_likelihoods.shape == (2,)


@patch("src.inference.encodon.EncodonPL")
def test_encodon_inference_embeddings(mock_pl, mock_ckpt):
    from types import SimpleNamespace
    instance = MagicMock()
    instance.return_value = SimpleNamespace(
        all_hidden_states=[torch.randn(2, 5, 16), torch.randn(2, 5, 16)],
        last_hidden_state=torch.randn(2, 5, 16),
    )
    mock_pl.return_value = instance

    inf = EncodonInference(model_path=mock_ckpt, task_type=TaskTypes.EMBEDDING_PREDICTION)
    inf.configure_model()
    batch = {
        MetadataFields.INPUT_IDS: torch.randint(0, 69, (2, 5)),
        MetadataFields.ATTENTION_MASK: torch.ones(2, 5),
    }
    out = inf._predict_step(batch, 0)
    assert isinstance(out, EmbeddingOutput)
    assert out.embeddings.shape[0] == 2


@patch("src.inference.encodon.EncodonPL")
def test_encodon_inference_fitness(mock_pl, mock_ckpt):
    from types import SimpleNamespace
    instance = MagicMock()
    instance.return_value = SimpleNamespace(logits=torch.randn(2, 5, 69))
    mock_pl.return_value = instance

    inf = EncodonInference(model_path=mock_ckpt, task_type=TaskTypes.FITNESS_PREDICTION)
    inf.configure_model()
    batch = {
        MetadataFields.INPUT_IDS: torch.randint(0, 69, (2, 5)),
        MetadataFields.ATTENTION_MASK: torch.ones(2, 5),
    }
    out = inf._predict_step(batch, 0)
    assert isinstance(out, FitnessPredictionOutput)
    assert out.fitness.shape == (2,)


@patch("src.inference.encodon.EncodonPL")
def test_encodon_inference_downstream(mock_pl, mock_ckpt):
    from types import SimpleNamespace
    instance = MagicMock()
    # Simulate presence of downstream heads
    instance.model = MagicMock()
    # Make projection a callable identity
    instance.model.cross_attention_input_proj = (lambda x: x)
    instance.model.cross_attention_head = MagicMock(return_value=torch.randn(2, 1))
    instance.hparams = SimpleNamespace(loss_type='regression')
    instance.return_value = SimpleNamespace(
        last_hidden_state=torch.randn(2, 5, 16)
    )
    mock_pl.return_value = instance

    inf = EncodonInference(model_path=mock_ckpt, task_type=TaskTypes.DOWNSTREAM_PREDICTION)
    inf.configure_model()
    batch = {
        MetadataFields.INPUT_IDS: torch.randint(0, 69, (2, 5)),
        MetadataFields.ATTENTION_MASK: torch.ones(2, 5),
    }
    out = inf._predict_step(batch, 0)
    assert isinstance(out, DownstreamPredictionOutput)
    assert out.predictions.shape == (2,)