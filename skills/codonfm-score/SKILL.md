---
name: codonfm-score
description: Score synonymous or missense coding variants with public CodonFM Encodon checkpoints using masked-codon reference-versus-alternate log-likelihood ratios. Use when a user explicitly asks for CodonFM or Encodon zero-shot variant scoring. Support the public mutation_prediction workflow only; reject Decodon and the newer synonymous-codon-aggregated missense_prediction workflow because they are not present in public CodonFM v1.
metadata:
  author: "NVIDIA BioNeMo <bionemofeedback@nvidia.com>"
---

# Score variants with public Encodon

Run general masked-codon `mutation_prediction` only. This produces a research
signal, not a clinical diagnosis or an expression-direction prediction.

## Instructions

Check whether the request is executable in public v1 before installing or
downloading anything. For synonymous-codon aggregation or Decodon, inspect the
[parser](../../src/runner.py) and [model configuration](../../src/config.py),
explain the missing feature, and finish. Do not implement the missing workflow,
search private code, or keep retrying unsupported commands.

Resolve the variant CSV, checkpoint, and output directory from the request and
available files. Validate inputs before inference. Execution requires the
project's ML dependencies and a compatible NVIDIA GPU. If a required resource
is unavailable, return the validated inputs where possible and a command with
the missing prerequisite identified. When scoring is requested and resources
are ready, execute and verify the score arrays. A request for preparation ends
with the inputs and command. If variants are missing, report the required
schema; do not invent variants or silently switch to a public dataset.

Default to the public 80M checkpoint for demonstrations:
`nvidia/NV-CodonFM-Encodon-80M-v1`, revision
`399ca9fe17b57941a7bebc6788033919b417413c`, file
`NV-CodonFM-Encodon-80M-v1.safetensors` and sibling `config.json`.
Reuse an existing checkpoint or download it when needed for the requested work.
Preserve an explicitly requested model size.

## Preflight

1. Confirm `src/runner.py`, `src/data/mutation_dataset.py`, and
   `src/inference/encodon.py` exist.
2. Accept only `encodon_80m`, `encodon_600m`, or `encodon_1b` as
   `--model_name`. The public parser lists larger names, but its model
   configuration does not implement them.
3. For model execution, require a `.ckpt` file, or a `.safetensors` file with
   sibling `config.json`. Input preparation can use a planned path.
4. Validate the CSV headers before starting a GPU job.

## Inputs

Require these CSV columns:

- `id`: unique row identifier.
- `ref_seq`: reference coding sequence, not genomic DNA with introns, UTR-only
  sequence, or protein sequence.
- `ref_codon` and `alt_codon`: three-nucleotide codons.
- `codon_position`: zero-based codon position relative to the CDS.

With `--extract-seq`, `MutationDataset` extracts an appropriate sequence window
from `ref_seq`; it does not derive or require `alt_seq`.

Before running, normalize sequences and codons to uppercase DNA (`A/C/G/T`),
require CDS lengths divisible by three, and check every row satisfies:

```text
0 <= codon_position < len(ref_seq) / 3
ref_seq[3 * codon_position : 3 * codon_position + 3] == ref_codon
```

The public extractor asserts the second condition and otherwise stops the job.

## Examples

Set `CODONFM_DATA_PATH` to the variant CSV, `CODONFM_CHECKPOINT_PATH` to the
checkpoint, and `CODONFM_RUN_DIR` to your chosen output directory:

```bash
python -m src.runner eval \
    --exp_name variant_scoring \
    --model_name encodon_80m \
    --checkpoint_path "$CODONFM_CHECKPOINT_PATH" \
    --data_path "$CODONFM_DATA_PATH" \
    --process_item mutation_pred_mlm \
    --dataset_name MutationDataset \
    --task_type mutation_prediction \
    --extract-seq \
    --mask_mutation \
    --num_nodes 1 \
    --num_gpus 1 \
    --num_workers 0 \
    --val_batch_size 2 \
    --out_dir "$CODONFM_RUN_DIR" \
    --predictions_output_dir "$CODONFM_RUN_DIR/predictions"
```

Do not remove `--mask_mutation`: without it, the reference codon remains
visible at the scored position and invalidates masked-codon LLR scoring.
For preparation requests, inspect the CSV directly against the input schema
and reference-position checks above, then report the rows checked and provide
the scoring command. Extra columns are allowed; use `--ref_seq_col` if the
reference sequence has a different column name. These checks do not require
the ML runtime. The command above performs inference when resources are ready.

The existing `--dryrun` optionally builds runtime configuration and skips
execution. It requires the ML dependencies, can create the prediction directory,
and does not read the CSV or load weights. Do not use it as evidence that inputs,
checkpoint compatibility, or prediction quality have been validated.

## Outputs

`--predictions_output_dir` receives:

- `ref_likelihoods_merged.npy`
- `alt_likelihoods_merged.npy`
- `likelihood_ratios_merged.npy`
- `ids_merged.npy`

Load the arrays with NumPy and align scores by `ids_merged.npy`. The reported
LLR is `log p(ref_codon) - log p(alt_codon)`; a larger positive value means the
alternate codon is less probable in context. It does not say whether
expression goes up or down.

## Boundaries

- General `mutation_prediction` handles both synonymous and missense changes.
- Do not use `missense_prediction`, `missense_inference`, `MissenseDataset`,
  `mutation_pred_clm`, `--organism_token`, or `--causal`; those are newer
  unavailable public-release features.
- If a user asks specifically for synonymous-codon-aggregated missense
  scoring, explain that public v1 only provides the general ref/alt LLR. Do not
  silently substitute the two methods.
- Do not invoke this skill for a bare “score this variant” request that does
  not name CodonFM or Encodon.
