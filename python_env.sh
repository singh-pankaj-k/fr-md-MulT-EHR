#!/bin/bash
# ==============================================================================
# Python Environment Setup Script
# ==============================================================================
# This script loads required modules, sets up the virtual environment,
# and installs dependencies.
# ==============================================================================

# Default venv directory if not provided (checks argument, then env var, then default)
VENV_DIR=${1:-${VENV_DIR:-".venv"}}

# 1. Load required modules
echo "Loading required modules..."
module purge
module load StdEnv/2023                # Ensure consistent software environment
module load python/3.12                # Use Python 3.12 (required for PyHealth 2.0 and latest wheels)
module load scipy-stack/2024a          # Optimized numpy, pandas, scipy, etc.
module load gcc arrow/17.0.0           # Load GCC and Apache Arrow
module load cuda/12.2                  # Required for GPU support in PyTorch
module load cudnn                      # Required for deep learning acceleration

# 2. Setup virtual environment if not already setup
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    virtualenv --no-download "$VENV_DIR"
    
    source "$VENV_DIR/bin/activate"
    pip install --no-index --upgrade pip
    
    echo "Installing/Updating dependencies from Compute Canada wheelhouse..."
    # Explicitly install torch first to avoid issues with extensions
    pip install --no-index torch
    pip install --no-index torch-scatter torch-sparse torch-cluster torch-spline-conv torch_geometric PyYAML
    
    # 2.1 Setup offline bundle for pyhealth (as it is not available in standard wheelhouse)
    if [ ! -d "offline_bundle" ]; then
        echo "Downloading pyhealth offline bundle (requires internet access, best done on login node)..."
        pip download -d offline_bundle pyhealth==2.0
    fi
    
    echo "Installing project requirements using wheelhouse and offline bundle..."
    pip install
    pip install --no-index --find-links=offline_bundle -r requirements.txt
    
    # Verification step: Ensure CUDA and torch_geometric are working
    echo "Verifying environment..."
    python -c "
import torch
print(f'Torch version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU Count: {torch.cuda.device_count()}')
import torch_geometric
print(f'torch_geometric version: {torch_geometric.__version__}')
if not torch.cuda.is_available():
    import sys; print('ERROR: CUDA not available on compute node!'); sys.exit(1)
" || { echo "ERROR: Verification failed!"; exit 1; }
else
    echo "Environment already exists at $VENV_DIR. Activating..."
    source "$VENV_DIR/bin/activate"
fi

echo "Environment setup complete and activated."
