# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tasks import finetune as finetune_fn


def _make_min_config():
    """Create a minimal config dict structure expected by tasks.finetune."""
    logger = MagicMock()
    data = MagicMock()
    trainer_kwargs = {"max_steps": 1}
    model = MagicMock()
    callbacks = {"cb": MagicMock()}
    return {
        "log": logger,
        "data": data,
        "trainer": trainer_kwargs,
        "model": model,
        "callbacks": callbacks,
    }


@patch("src.tasks.load_file")
@patch("src.tasks.Trainer")
@patch("src.tasks.os.path.exists")
def test_finetune_with_safetensors_loads_and_uses_none_ckpt(mock_exists, MockTrainer, mock_load_file, tmp_path):
    config = _make_min_config()

    # Pretend pretrained safetensors exists and target ckpt does not
    pretrained = str(tmp_path / "pretrained.safetensors")
    ckpt_path = str(tmp_path / "last.ckpt")
    
    def _exists(p):
        if p == pretrained:
            return True
        if p == ckpt_path:
            return False
        return Path(p).exists()
    mock_exists.side_effect = _exists
    mock_load_file.return_value = {"some": "weights"}

    trainer = MagicMock()
    MockTrainer.return_value = trainer

    finetune_fn(
        config=config,
        pretrained_ckpt_path=pretrained,
        seed=123,
        resume_trainer_state=False,
        config_dict={"foo": "bar"},
        out_dir=str(tmp_path),
        ckpt_path=ckpt_path,
    )

    # configure_model called with safetensors state_dict
    config["model"].configure_model.assert_called_once()
    args, kwargs = config["model"].configure_model.call_args
    assert kwargs["state_dict"] == {"some": "weights"}

    # trainer.fit called with ckpt_path=None (first-time finetune)
    trainer.fit.assert_called_once()
    _, k = trainer.fit.call_args
    assert k.get("ckpt_path") is None


@patch("src.tasks.torch.load")
@patch("src.tasks.Trainer")
@patch("src.tasks.os.path.exists")
def test_finetune_with_ckpt_loads_and_maybe_resumes(mock_exists, MockTrainer, mock_torch_load, tmp_path):
    config = _make_min_config()
    pretrained = str(tmp_path / "pretrained.ckpt")
    ckpt_path = str(tmp_path / "last.ckpt")

    # Pretrained exists, target ckpt does not
    
    def _exists(p):
        if p == pretrained:
            return True
        if p == ckpt_path:
            return False
        return Path(p).exists()
    mock_exists.side_effect = _exists
    mock_torch_load.return_value = {"state_dict": {"model.layer": 1}}

    trainer = MagicMock()
    MockTrainer.return_value = trainer

    # Case 1: resume_trainer_state=False -> ckpt_path=None
    finetune_fn(
        config=config,
        pretrained_ckpt_path=pretrained,
        seed=123,
        resume_trainer_state=False,
        config_dict={},
        out_dir=str(tmp_path),
        ckpt_path=ckpt_path,
    )
    config["model"].configure_model.assert_called()
    _, k = trainer.fit.call_args
    assert k.get("ckpt_path") is None

    # Case 2: resume_trainer_state=True -> ckpt_path points to pretrained ckpt
    config2 = _make_min_config()
    trainer2 = MagicMock()
    MockTrainer.return_value = trainer2
    finetune_fn(
        config=config2,
        pretrained_ckpt_path=pretrained,
        seed=123,
        resume_trainer_state=True,
        config_dict={},
        out_dir=str(tmp_path),
        ckpt_path=ckpt_path,
    )
    _, k2 = trainer2.fit.call_args
    assert k2.get("ckpt_path") == pretrained


@patch("src.tasks.load_file")
@patch("src.tasks.Trainer")
@patch("src.tasks.os.path.exists")
def test_finetune_resume_trainer_state_asserts_on_non_ckpt(mock_exists, MockTrainer, mock_load_file, tmp_path):
    config = _make_min_config()
    pretrained = str(tmp_path / "pretrained.safetensors")
    ckpt_path = str(tmp_path / "last.ckpt")

    
    def _exists(p):
        if p == pretrained:
            return True
        if p == ckpt_path:
            return False
        return Path(p).exists()
    mock_exists.side_effect = _exists
    mock_load_file.return_value = {}

    with pytest.raises(AssertionError):
        finetune_fn(
            config=config,
            pretrained_ckpt_path=pretrained,
            seed=123,
            resume_trainer_state=True,
            config_dict={},
            out_dir=str(tmp_path),
            ckpt_path=ckpt_path,
        )


@patch("src.tasks.Trainer")
@patch("src.tasks.os.path.exists", return_value=False)
def test_finetune_without_pretrained_starts_from_scratch(mock_exists, MockTrainer, tmp_path):
    config = _make_min_config()
    trainer = MagicMock()
    MockTrainer.return_value = trainer

    finetune_fn(
        config=config,
        pretrained_ckpt_path=str(tmp_path / "missing.ckpt"),
        seed=123,
        resume_trainer_state=False,
        config_dict={},
        out_dir=str(tmp_path / "fresh_out"),
        ckpt_path=str(tmp_path / "last.ckpt"),
    )

    # Should call configure_model() without state_dict
    config["model"].configure_model.assert_called_once()
    args, kwargs = config["model"].configure_model.call_args
    assert "state_dict" not in kwargs


