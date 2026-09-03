---
name: codonfm-setup
description: Set up the public CodonFM v1 repository and download public Encodon checkpoints. Use for requests to build or launch the CodonFM development container, configure local data/checkpoint mounts, verify GPU access, or download public Encodon 80M, 600M, 1B, or Cdwt-1B weights. Do not use for Decodon, Encodon 5B/10B, missense-aggregation, or codon-optimization setup because those implementations are not in the public repository.
metadata:
  author: "NVIDIA BioNeMo <bionemofeedback@nvidia.com>"
---

# CodonFM public setup

Operate from the public CodonFM repository root. Support only the checked-in
public v1 code and public Encodon checkpoints.

## Preflight

1. Confirm `Dockerfile`, `run_dev.sh`, and `src/runner.py` exist.
2. Confirm `docker info` succeeds and `nvidia-smi` sees the intended GPU.
3. Run `bash -n run_dev.sh` before launching it.
4. Resolve explicit host paths for data and checkpoints. Do not rely on the
   `/data/codonfm` defaults unless the user confirms they exist.
5. Check for an existing container before launch:

```bash
docker ps -a --filter name='^/codon-fm-dev-container$'
```

If an exact-name container is running, `run_dev.sh` stops and removes it; tell
the user before replacement. If it is stopped, the script cannot reuse the
name, so obtain confirmation before removing it with
`docker rm codon-fm-dev-container`. The public script also uses host
networking/IPC and mounts the user's SSH directory read-only; disclose this
before execution.

## Build and launch

```bash
cd /path/to/CodonFM
bash run_dev.sh \
    --data-dir /absolute/path/to/data \
    --checkpoints-dir /absolute/path/to/checkpoints
```

The host checkpoint directory is mounted at `/data/checkpoints` inside the
container. The image is `codon-fm-dev`; the container is
`codon-fm-dev-container`.

Use only the checked-in public code and the dependency versions declared in
its `Dockerfile` and `requirements.txt`.

## Run directly without Docker

Use this path when Docker is unavailable and the host has a compatible NVIDIA
driver. The tested baseline is Python 3.11, CUDA-capable PyTorch, and one GPU.
Operate from a writable checkout and use a dedicated virtual environment:

```bash
cd /path/to/CodonFM
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p /absolute/path/to/codonfm-matplotlib-cache
export MPLCONFIGDIR=/absolute/path/to/codonfm-matplotlib-cache
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True` and the selected GPU name. The requirements file configures the
CUDA 12.4 PyTorch index for xFormers. Use explicit host paths in all subsequent
runner commands; unlike the container path, no `/data/checkpoints` mount is
created.

## Download a checkpoint

Run inside the container, or in another environment with Hugging Face Hub:

```bash
hf download nvidia/NV-CodonFM-Encodon-1B-v1 \
    --local-dir /data/checkpoints/encodon-1b
```

Other supported public model IDs are:

- `nvidia/NV-CodonFM-Encodon-80M-v1`
- `nvidia/NV-CodonFM-Encodon-600M-v1`
- `nvidia/NV-CodonFM-Encodon-Cdwt-1B-v1`

Use `--model_name encodon_80m`, `encodon_600m`, or `encodon_1b` according to
architecture size. Cdwt-1B uses `encodon_1b` because Cdwt is a checkpoint
training property, not a separate architecture.

For `.safetensors`, keep `config.json` in the same directory as the model
file. Never invent a Decodon or undocumented checkpoint path.

## Verify

```bash
docker exec codon-fm-dev-container python -c \
    "import torch; print(torch.cuda.is_available())"
```

Expect `True` on a configured NVIDIA GPU host. If Docker or a GPU is
unavailable, report the missing prerequisite; do not claim setup succeeded.

## Public-v1 boundaries

- Supported: Encodon 80M, 600M, 1B, and Cdwt-1B.
- Not supported: Decodon, Encodon 5B/10B, sequence generation, specialized
  missense aggregation/fine-tuning, and `scripts/codon_optimize.py`.
- CodonFM consumes coding sequences. It is not a variant caller, aligner, GTF
  annotator, or general VCF analysis tool.
