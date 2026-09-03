---
name: codonfm-embed
description: Extract frozen CLS embeddings from public CodonFM Encodon checkpoints for coding-sequence property modeling. Use when a user explicitly asks for CodonFM or Encodon embeddings, or wants Encodon features for translation-efficiency, expression, or mRNA-stability modeling. Support Encodon embedding_prediction only; do not claim Decodon embedding support in public CodonFM v1.
metadata:
  author: "NVIDIA BioNeMo <bionemofeedback@nvidia.com>"
---

# Extract public Encodon embeddings

Extract one frozen CLS vector per coding sequence. This workflow writes
embeddings only; it does not automatically train a downstream regressor.

## Preflight and inputs

1. Confirm `src/runner.py`, `src/data/codon_bert_dataset.py`, and
   `src/inference/encodon.py` exist.
2. Accept only `encodon_80m`, `encodon_600m`, or `encodon_1b`.
3. Require a `.ckpt`, or `.safetensors` with sibling `config.json`.
4. Require CSV columns `id`, `ref_seq`, `value`, and `split`.

`ref_seq` must be a coding sequence. For extraction-only data, set `value` to
`0.0` and `split` to `test` on every row. Although the public dataset labels
`split` optional, its evaluation path calls the test split and fails without
that column. Normalize sequences to uppercase DNA (`A/C/G/T`) and require
lengths divisible by three. Sequences longer than `--context_length - 2`
codons are truncated rather than embedded in full.

## Run

Validate configuration first:

```bash
python -m src.runner eval \
    --exp_name embed_extract \
    --model_name encodon_1b \
    --checkpoint_path /path/to/encodon_1b.safetensors \
    --data_path /path/to/sequences.csv \
    --process_item codon_sequence \
    --dataset_name CodonBertDataset \
    --task_type embedding_prediction \
    --num_nodes 1 \
    --num_gpus 1 \
    --out_dir /path/to/run \
    --predictions_output_dir /path/to/run/predictions \
    --dryrun
```

After the dry run succeeds, rerun without `--dryrun`.

## Outputs

- `embeddings_merged.npy`: shape `(number_of_rows, hidden_size)`.
- `ids_merged.npy`: IDs aligned with the embedding rows.

Use the checked-in Encodon notebooks as downstream-model references:

- `notebooks/4-EnCodon-Downstream-Task-riboNN.ipynb`
- `notebooks/5-EnCodon-Downstream-Task-mRFP-expression.ipynb`
- `notebooks/6-EnCodon-Downstream-Task-mRNA-stability.ipynb`

Do not reference `notebooks/te_predictor.py`, `notebooks/mfe_predictor.py`, or
Decodon notebooks because they are absent from public v1.

## Boundaries

- Do not use for Decodon; the public repository has no Decodon model or
  inference class.
- Do not claim a benchmark-trained regressor generalizes to a new organism,
  cell type, or assay without new labeled validation data.
- Do not invoke this skill for a generic expression-prediction request that
  does not mention CodonFM or Encodon.
