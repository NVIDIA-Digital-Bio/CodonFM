---
name: codonfm-finetune
description: Fine-tune public CodonFM Encodon checkpoints on labeled coding-sequence or coding-variant data using LoRA, head-only, or full fine-tuning. Use when a user explicitly asks to fine-tune CodonFM or Encodon for regression or classification. Support generic public-v1 Encodon workflows only; reject Decodon, MissenseDataset, missense_synom_agg, and generation workflows.
---

# Fine-tune public Encodon

Use `--pretrained_ckpt_path` for public v1. Do not substitute
`--checkpoint_path`: the public runner does not forward that argument to the
fine-tuning task.

## Supported strategies

- `lora`: adapter fine-tuning; default choice for smaller datasets.
- `head_only_random`: freeze the backbone and train a new head.
- `head_only_pretrained`: train an existing compatible pretrained head.
- `full`: update the complete model.

Accept only `encodon_80m`, `encodon_600m`, or `encodon_1b`.

## Sequence-level regression or classification

Require `id`, `ref_seq`, `value`, and `split` columns. `split` values must be
`train`, `val`, or `test`, and every split must be non-empty. Normalize
sequences to uppercase DNA (`A/C/G/T`) and require lengths divisible by three.
Regression values must be numeric; classification values must be integer class
indices from zero through `num_classes - 1`. Use a downstream head for scalar
targets. Ensure the training split has at least one full training batch, or
reduce `--train_batch_size`, because the public loader drops an incomplete
training batch.

Start with a configuration-only run:

```bash
python -m src.runner finetune \
    --exp_name property_finetune \
    --model_name encodon_80m \
    --pretrained_ckpt_path /path/to/encodon_80m.safetensors \
    --data_path /path/to/labeled_sequences.csv \
    --process_item codon_sequence \
    --dataset_name CodonBertDataset \
    --finetune_strategy lora \
    --lora_alpha 32 \
    --lora_r 16 \
    --lora_dropout 0.1 \
    --loss_type regression \
    --use_downstream_head \
    --lr 2e-5 \
    --max_steps 1000 \
    --warmup_iterations 100 \
    --check_val_every_n_epoch 1 \
    --train_batch_size 4 \
    --val_batch_size 4 \
    --num_nodes 1 \
    --num_gpus 1 \
    --out_dir /path/to/run \
    --checkpoints_dir /path/to/run/checkpoints \
    --dryrun
```

For classification, replace `--loss_type regression` with
`--loss_type classification` and pass the correct `--num_classes`.

## Generic coding-variant classification

Use `MutationDataset` only for an ordinary labeled variant head, not the newer
synonymous-codon aggregation loss. Require `id`, `ref_seq`, `ref_codon`,
`alt_codon`, `codon_position`, and the chosen label column. Starting from the
sequence-level command, change/add:

```text
--process_item mutation_pred_mlm
--dataset_name MutationDataset
--label_col label
--loss_type classification
--num_classes 2
--use_downstream_head
--extract-seq
--mask_mutation
--train_val_test_ratio 0.8 0.1 0.1
```

Always keep `--mask_mutation` for masked-codon variant inputs.
Keep each variant CSV in a directory without stale `train_idx.npy`,
`val_idx.npy`, or `test_idx.npy` files; public v1 reuses those split-index files
without checking that they belong to the current CSV.

## Execute and outputs

After `--dryrun` succeeds, rerun the same command without `--dryrun`.

Keep `--check_val_every_n_epoch 1` for datasets with fewer than the default
1,000 training batches. Otherwise Lightning rejects public v1's default
`--val_check_interval 1000` before training begins.

- Checkpoints are written under the explicitly supplied `--checkpoints_dir`,
  including `last.ckpt` and configured best checkpoints.
- CSV metrics are written below `--out_dir/<exp_name>/version_*` unless W&B is
  enabled.
- W&B requires `--enable_wandb`, `--project_name`, and `--entity` together.
- Fine-tuning does not produce prediction arrays; run an evaluation task
  separately against the resulting checkpoint.

## Boundaries

- Do not use `MissenseDataset`, `missense_seq`, `missense_inference`,
  `missense_synom_agg`, or any `--missense_*` flag. They are absent publicly.
- Do not use Decodon model names, CLM preprocessing, organism tokens, or
  generation datasets.
- Require an explicit learning rate. Public v1 passes `lr=None` otherwise.
- Treat scientific and clinical validity as a separate validation problem;
  successful training does not certify the resulting model.
