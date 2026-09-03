---
name: codonfm-score
description: Score synonymous or missense coding variants with public CodonFM Encodon checkpoints using masked-codon reference-versus-alternate log-likelihood ratios. Use when a user explicitly asks for CodonFM or Encodon zero-shot variant scoring. Support the public mutation_prediction workflow only; reject Decodon and the newer synonymous-codon-aggregated missense_prediction workflow because they are not present in public CodonFM v1.
---

# Score variants with public Encodon

Run general masked-codon `mutation_prediction` only. This produces a research
signal, not a clinical diagnosis or an expression-direction prediction.

## Preflight

1. Confirm `src/runner.py`, `src/data/mutation_dataset.py`, and
   `src/inference/encodon.py` exist.
2. Accept only `encodon_80m`, `encodon_600m`, or `encodon_1b` as
   `--model_name`. The public parser lists larger names, but its model
   configuration does not implement them.
3. Require a `.ckpt` file, or a `.safetensors` file with sibling
   `config.json`.
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

## Run

First validate configuration with `--dryrun`:

```bash
python -m src.runner eval \
    --exp_name variant_scoring \
    --model_name encodon_1b \
    --checkpoint_path /path/to/encodon_1b.safetensors \
    --data_path /path/to/variants.csv \
    --process_item mutation_pred_mlm \
    --dataset_name MutationDataset \
    --task_type mutation_prediction \
    --extract-seq \
    --mask_mutation \
    --num_nodes 1 \
    --num_gpus 1 \
    --out_dir /path/to/run \
    --predictions_output_dir /path/to/run/predictions \
    --dryrun
```

Do not remove `--mask_mutation`: without it, the reference codon remains
visible at the scored position and invalidates masked-codon LLR scoring. After
the dry run succeeds, rerun the same command without `--dryrun`.

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
