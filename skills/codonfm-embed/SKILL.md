---
name: codonfm-embed
description: Extract frozen CLS embeddings from public CodonFM Encodon checkpoints for coding-sequence property modeling. Use when a user explicitly asks for CodonFM or Encodon embeddings, or wants Encodon features for translation-efficiency, expression, or mRNA-stability modeling. Support Encodon embedding_prediction only; do not claim Decodon embedding support in public CodonFM v1.
metadata:
  author: "NVIDIA BioNeMo <bionemofeedback@nvidia.com>"
---

# Extract public Encodon embeddings

Extract one frozen CLS vector per coding sequence. This workflow writes
embeddings only; it does not automatically train a downstream regressor.

## Instructions

Resolve the sequence CSV, checkpoint, and output directory from the request and
available files. Validate inputs before extraction. Execution requires the
project's ML dependencies and a compatible NVIDIA GPU. If a required resource
is unavailable, complete the available preparation and return the command with
that prerequisite identified. When extraction is requested and resources are
ready, execute and verify the embedding arrays. A request for preparation ends
with the inputs and command. If no sequences were supplied, report the required
inputs. For Decodon, inspect the [public parser](../../src/runner.py) and
[model configuration](../../src/config.py), explain the missing implementation,
and finish without attempting installation or model development.

For a demonstration use `nvidia/NV-CodonFM-Encodon-80M-v1`, revision
`399ca9fe17b57941a7bebc6788033919b417413c`, file
`NV-CodonFM-Encodon-80M-v1.safetensors` with sibling `config.json`.
Reuse an existing checkpoint or download it when needed for the requested work.
Preserve a user's explicit checkpoint choice.

## Preflight and inputs

1. Confirm `src/runner.py`, `src/data/codon_bert_dataset.py`, and
   `src/inference/encodon.py` exist.
2. Accept only `encodon_80m`, `encodon_600m`, or `encodon_1b`.
3. For execution require a `.ckpt`, or `.safetensors` with sibling `config.json`;
   input preparation can use a planned path.
4. Require CSV columns `id`, `ref_seq`, `value`, and `split`.

`ref_seq` must be a coding sequence. For extraction-only data, set `value` to
`0.0` and `split` to `test` on every row. Although the public dataset labels
`split` optional, its evaluation path calls the test split and fails without
that column. Normalize sequences to uppercase DNA (`A/C/G/T`) and require
lengths divisible by three. Sequences longer than `--context_length - 2`
codons are truncated rather than embedded in full.

## Examples

Set `CODONFM_DATA_PATH` to the sequence CSV, `CODONFM_CHECKPOINT_PATH` to the
checkpoint, and `CODONFM_RUN_DIR` to your chosen output directory:

```bash
python -m src.runner eval \
    --exp_name embed_extract \
    --model_name encodon_80m \
    --checkpoint_path "$CODONFM_CHECKPOINT_PATH" \
    --data_path "$CODONFM_DATA_PATH" \
    --process_item codon_sequence \
    --dataset_name CodonBertDataset \
    --task_type embedding_prediction \
    --num_nodes 1 \
    --num_gpus 1 \
    --num_workers 0 \
    --val_batch_size 2 \
    --out_dir "$CODONFM_RUN_DIR" \
    --predictions_output_dir "$CODONFM_RUN_DIR/predictions"
```

For preparation requests, inspect the CSV directly against the input schema
above and report the test-row count and sequence checks. Extra columns are
allowed; extraction does not require a measured target. This does not require
the ML runtime. The command above performs extraction when resources are ready.

The existing `--dryrun` optionally builds runtime configuration and skips
execution. It requires the ML dependencies, can create the prediction directory,
and does not read the CSV or load weights. Do not use it as evidence that inputs,
checkpoint compatibility, or embedding quality have been validated.

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
