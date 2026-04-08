#!/bin/bash
#SBATCH --job-name=MulT-EHR-Full
#SBATCH --account=def-pankajpriscilla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pankajpriscilla@example.com

# ==============================================================================
# MulT-EHR Full Production Submission Script (Narval Cluster)
# ==============================================================================
# This script runs the full end-to-end pipeline (MIMIC-IV and MIMIC-III)
# in production mode using a single A100 GPU.
# ==============================================================================

# 1. Load required modules (Adjust based on cluster environment)
module load python/3.10
module load cuda/11.8

# 2. Setup virtual environment
if [ ! -d ".venv_narval" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv_narval
    source .venv_narval/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    # Ensure torch-geometric is installed with the correct CUDA backend
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
else
    source .venv_narval/bin/activate
fi

# 3. Environment variables for production run
export MODE=full
export PYTORCH_ENABLE_MPS_FALLBACK=0  # Not needed on Linux/CUDA
export CUDA_VISIBLE_DEVICES=0

# 4. Cleanup previous artifacts to ensure a fresh start
echo "Cleaning environment for production run..."
./cleanup.sh

# 5. Execute the full pipeline
# This will run: Graph Creation -> Pretraining -> Training (4 models) -> Benchmarking
# for both MIMIC-IV and MIMIC-III.
echo "Starting full pipeline execution at $(date)"
./run_full_pipeline.sh

echo "Full pipeline execution finished at $(date)"
