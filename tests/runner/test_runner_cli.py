# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import patch, MagicMock

import pytest


@patch("src.runner.get_config")
@patch("src.runner.fdl.build")
@patch("src.runner.finetune")
def test_runner_main_finetune_dispatch(mock_finetune, mock_build, mock_get_config, monkeypatch):
    # Build a fake argv for finetune mode
    argv = [
        "prog",
        "finetune",
        "--exp_name", "run1",
        "--data_path", "/data",
        "--process_item", "codon_sequence",
        "--dataset_name", "CodonBertDataset",
        "--model_name", "encodon_80m",
        "--out_dir", "/out",
        "--checkpoints_dir", "/out/ckpts",
        "--checkpoint_path", "/pretrained/model.ckpt",
        "--resume_trainer_state",
        "--finetune_strategy", "lora",
        "--lora",
    ]
    monkeypatch.setenv("WANDB_API_KEY", "")  # Ensure branch without W&B
    with patch.object(sys, "argv", argv):
        # Import inside to use patched argv
        import importlib
        mod = importlib.import_module("src.runner")
        # Spy on parser to avoid running the job by setting dryrun
        with patch.object(mod, "get_parser") as mock_get_parser:
            parser = mod.get_parser()
            parser.set_defaults(dryrun=True)
            mock_get_parser.return_value = parser
            mod.main()

    # Ensure config was constructed and finetune selected
    assert mock_get_config.called
    assert mock_build.called
    # In dryrun, finetune should not be called
    mock_finetune.assert_not_called()


@patch("src.runner.get_config")
@patch("src.runner.fdl.build")
@patch("src.runner.finetune")
def test_runner_finetune_args_propagation(mock_finetune, mock_build, mock_get_config, monkeypatch):
    argv = [
        "prog",
        "finetune",
        "--exp_name", "run2",
        "--data_path", "/data",
        "--process_item", "codon_sequence",
        "--dataset_name", "CodonBertDataset",
        "--model_name", "encodon_80m",
        "--out_dir", "/out",
        "--checkpoints_dir", "/out/ckpts",
        "--checkpoint_path", "/pretrained/initial.ckpt",
    ]
    monkeypatch.setenv("WANDB_API_KEY", "")
    with patch.object(sys, "argv", argv):
        import importlib
        mod = importlib.import_module("src.runner")
        # Let it run fully to call finetune (not dryrun)
        mod.main()

    # finetune should be invoked exactly once with keyword args
    assert mock_finetune.called
    _, kwargs = mock_finetune.call_args
    assert "config" in kwargs and "pretrained_ckpt_path" in kwargs and "ckpt_path" in kwargs
    assert kwargs["pretrained_ckpt_path"] is None or isinstance(kwargs["pretrained_ckpt_path"], str)



@patch("src.runner.get_config")
@patch("src.runner.fdl.build")
@patch("src.runner.train")
def test_runner_pretrain_dispatch(mock_train, mock_build, mock_get_config, monkeypatch):
    argv = [
        "prog",
        "pretrain",
        "--exp_name", "run_pre",
        "--data_path", "/data",
        "--process_item", "codon_sequence",
        "--dataset_name", "CodonBertDataset",
        "--model_name", "encodon_80m",
        "--out_dir", "/out",
        "--checkpoints_dir", "/out/ckpts",
    ]
    monkeypatch.setenv("WANDB_API_KEY", "")
    with patch.object(sys, "argv", argv):
        import importlib
        mod = importlib.import_module("src.runner")
        mod.main()

    assert mock_train.called
    _, kwargs = mock_train.call_args
    assert "config" in kwargs and "ckpt_path" in kwargs and kwargs["ckpt_path"].endswith("last.ckpt")


@patch("src.runner.get_config")
@patch("src.runner.fdl.build")
@patch("src.runner.evaluate")
def test_runner_eval_dispatch(mock_evaluate, mock_build, mock_get_config, monkeypatch):
    argv = [
        "prog",
        "eval",
        "--exp_name", "run_eval",
        "--data_path", "/data",
        "--process_item", "codon_sequence",
        "--dataset_name", "CodonBertDataset",
        "--model_name", "encodon_80m",
        "--out_dir", "/out",
        "--checkpoints_dir", "/out/ckpts",
        "--checkpoint_path", "/ckpt/model.ckpt",
        "--task_type", "masked_language_modeling",
    ]
    monkeypatch.setenv("WANDB_API_KEY", "")
    with patch.object(sys, "argv", argv):
        import importlib
        mod = importlib.import_module("src.runner")
        mod.main()

    assert mock_evaluate.called
    _, kwargs = mock_evaluate.call_args
    assert kwargs["model_ckpt_path"] == "/ckpt/model.ckpt"


def test_runner_eval_requires_checkpoint(monkeypatch):
    argv = [
        "prog",
        "eval",
        "--exp_name", "run_eval",
        "--data_path", "/data",
        "--process_item", "codon_sequence",
        "--dataset_name", "CodonBertDataset",
        "--model_name", "encodon_80m",
        "--out_dir", "/out",
        "--checkpoints_dir", "/out/ckpts",
    ]
    monkeypatch.setenv("WANDB_API_KEY", "")
    with patch.object(sys, "argv", argv):
        import importlib
        mod = importlib.import_module("src.runner")
        with pytest.raises(SystemExit):
            mod.main()


def test_runner_wandb_requires_project_and_entity(monkeypatch):
    argv = [
        "prog",
        "finetune",
        "--exp_name", "run_wb",
        "--data_path", "/data",
        "--process_item", "codon_sequence",
        "--dataset_name", "CodonBertDataset",
        "--model_name", "encodon_80m",
        "--out_dir", "/out",
        "--checkpoints_dir", "/out/ckpts",
        "--enable_wandb",
    ]
    monkeypatch.setenv("WANDB_API_KEY", "key")
    with patch.object(sys, "argv", argv):
        import importlib
        mod = importlib.import_module("src.runner")
        with pytest.raises(SystemExit):
            mod.main()

