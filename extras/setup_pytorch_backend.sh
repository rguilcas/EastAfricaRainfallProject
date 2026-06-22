#!/usr/bin/env bash

# --- File Documentation ---
# This script sets up the PyTorch backend in a conda environment. 
# It detects the appropriate backend (CUDA, MPS, or CPU) based on the system and installs the necessary PyTorch packages. 
# You can specify the environment name, backend, and CUDA version as command-line arguments.
# --------------------------

set -euo pipefail

ENV_NAME="bccr-ml-project"
BACKEND="auto"
CUDA_VERSION="12.1"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash scripts/setup_pytorch_backend.sh [--env-name NAME] [--backend auto|cuda|mps|cpu] [--cuda-version 12.1]"
            exit 1
            ;;
    esac
done

if [[ "$BACKEND" == "auto" ]]; then
    case "$(uname -s)" in
        Darwin)
            BACKEND="mps"
            ;;
        Linux)
            if command -v nvidia-smi >/dev/null 2>&1; then
                BACKEND="cuda"
            else
                BACKEND="cpu"
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            BACKEND="cuda"
            ;;
        *)
            BACKEND="cpu"
            ;;
    esac
fi

echo "Installing PyTorch backend '$BACKEND' in conda env '$ENV_NAME'..."

if [[ "$BACKEND" == "cuda" ]]; then
    conda install -n "$ENV_NAME" -y pytorch pytorch-cuda="$CUDA_VERSION" -c pytorch -c nvidia
elif [[ "$BACKEND" == "mps" ]]; then
    conda install -n "$ENV_NAME" -y pytorch -c pytorch
elif [[ "$BACKEND" == "cpu" ]]; then
    conda install -n "$ENV_NAME" -y pytorch cpuonly -c pytorch
else
    echo "Invalid backend: $BACKEND"
    echo "Valid values: auto, cuda, mps, cpu"
    exit 1
fi

echo "Done."
echo "Validate with: conda run -n $ENV_NAME python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())'"
