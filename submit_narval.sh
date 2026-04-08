#!/bin/bash
#SBATCH --job-name=MulT-EHR-Full
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --account=def-mahyarh
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pankajpriscilla@example.com

# ==============================================================================
# MulT-EHR Full Production Submission Script (Narval Cluster)
# ==============================================================================
# This script runs the full end-to-end pipeline (MIMIC-IV and MIMIC-III)
# in production mode using A100 GPUs.
# ==============================================================================

set -e                                   # Exit script on any error

# 1. Load required modules
module purge
module load python/3.11.5                # Load Python 3.11.5
module load gcc arrow/17.0.0             # Load GCC and Apache Arrow

# 2. Setup virtual environment
VENV_DIR=".venv_narval"
if [ ! -d "$VENV_DIR" ]; then             # Create venv if missing
    echo "Creating virtual environment at $VENV_DIR..."
    python -m venv --system-site-packages "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip
    # Install project in editable mode as preferred on Narval
    python -m pip install --editable .
else
    source "$VENV_DIR/bin/activate"
fi

# 3. Environment variables for production run
export MODE=full
# CUDA_VISIBLE_DEVICES is handled internally by run_train.py for parallel training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Memory optimization
export OMP_NUM_THREADS=12                # 48 CPU cores / 4 GPUs = 12 threads per GPU

# Check available GPUs
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# 4. Cleanup previous artifacts to ensure a fresh start
echo "Cleaning environment for production run..."
./cleanup.sh

# 5. Execute the full pipeline
# This will run: Graph Creation -> Pretraining -> Training (4 models in parallel) -> Benchmarking
# for both MIMIC-IV and MIMIC-III.
echo "Starting full pipeline execution at $(date)"
./run_full_pipeline.sh

echo "Full pipeline execution finished at $(date)"

# Submit on Narval cluster using
# sbatch --account=def-mahyarh submit_narval.sh
