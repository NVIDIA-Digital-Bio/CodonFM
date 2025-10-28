# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import fiddle as fdl

from src.config import get_model_config, get_trainer_config


def _base_args(**overrides):
    d = dict(
        mode="finetune",
        model_name="encodon_80m",
        max_steps=100,
        warmup_iterations=10,
        lr=1e-3,
        weight_decay=0.01,
        finetune_strategy="full",
        loss_type="regression",
        num_nodes=1,
        num_gpus=1,
        out_dir="/tmp",
        limit_val_batches=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        gradient_accumulation_steps=1,
        enable_fsdp=False,
        check_val_every_n_epoch=None,
        val_check_interval=50,
        context_length=256,
        train_batch_size=1,
        val_batch_size=1,
    )
    d.update(overrides)
    return SimpleNamespace(**d)


def test_model_config_lora_auto_enable_when_strategy_lora():
    args = _base_args(finetune_strategy="lora", lora=False)
    cfg = get_model_config(args)
    built = fdl.build(cfg)
    assert built.hparams.lora is True
    assert built.hparams.finetune_strategy == "lora"


def test_trainer_strategy_for_finetune_uses_find_unused():
    args = _base_args()
    trainer_kwargs = get_trainer_config(args)
    assert trainer_kwargs["strategy"] == "ddp_find_unused_parameters_true"


