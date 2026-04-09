#!/bin/bash
#SBATCH --job-name=MulT-EHR-Full
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=2:00:00
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

# 1. Setup python environment
# Extracting setup logic to python_env.sh for reusability.
# The script handles module loading and venv setup (only if missing).
# Default to .venv in current directory unless VENV_DIR is provided by user.
VENV_DIR=${VENV_DIR:-".venv"}
source ./python_env.sh "$VENV_DIR"

# 3. Environment variables for production run
export MODE=full
# CUDA_VISIBLE_DEVICES is handled internally by run_train.py for parallel training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Memory optimization
export OMP_NUM_THREADS=12                # 48 CPU cores / 4 GPUs = 12 threads per GPU

# Check available GPUs
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# 4. Cleanup previous artifacts to ensure a fresh start
# This is handled by run_pipeline_unified.sh when CLEAN=true
# Or we can do it explicitly here as well.
echo "Cleaning environment for production run..."
./cleanup.sh

# 5. Execute the full pipeline using srun to ensure proper resource allocation
echo "Starting full unified pipeline execution at $(date)"
# We run the unified pipeline for both datasets sequentially in production mode.
# First run cleans, second run doesn't (to avoid deleting the first dataset's results).

echo ">>> Running MIMIC-IV Pipeline <<<"
srun ./run_pipeline_unified.sh mimiciv full true

echo ">>> Running MIMIC-III Pipeline <<<"
srun ./run_pipeline_unified.sh mimiciii full false

echo "Full pipeline execution finished at $(date)"

# Submit on Narval cluster using
# sbatch --account=def-mahyarh submit_narval.sh
