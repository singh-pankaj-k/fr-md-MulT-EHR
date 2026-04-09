#!/bin/bash

# Multi-task Heterogeneous Graph Learning on EHR - Unified Pipeline Script
# Three variables control the execution:
# 1. DATASET: mimiciv (default) or mimiciii
# 2. MODE: dev (default) or full
# 3. CLEAN: true or false (default)

DATASET=${1:-mimiciv}
MODE=${2:-dev}
CLEAN=${3:-false}

set -e # Exit immediately if a command exits with a non-zero status.

echo "================================================================"
echo "Starting Unified EHR Pipeline"
echo "DATASET: $DATASET"
echo "MODE:    $MODE"
echo "CLEAN:   $CLEAN"
echo "================================================================"

# Step 0: Optionally determine clean flag
CLEAN_FLAG=""
if [ "$CLEAN" == "true" ]; then
    ./cleanup.sh
    CLEAN_FLAG="--clean"
fi

# Determine python command
PYTHON_CMD="python"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python3"
fi

# Enable MPS fallback for better compatibility on macOS
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Export MODE for downstream scripts (like run_train.py)
export MODE=$MODE

# Set dev-mode hyperparameter defaults if in dev mode
if [ "$MODE" == "dev" ]; then
    export DEV_EPOCHS=${DEV_EPOCHS:-2}
    export DEV_BATCH_SIZE=${DEV_BATCH_SIZE:-1024}
    export DEV_SAMPLES=${DEV_SAMPLES:-100}
    export DEV_PRETRAIN_EPOCHS=${DEV_PRETRAIN_EPOCHS:-2}
fi

# Determine config suffix based on dataset
if [ "$DATASET" == "mimiciv" ]; then
    CONFIG_DS="MIMIC4"
    CONFIG_PT="MIMIC4_TransE"
    CONFIG_TRAIN="HGT_Causal_MIMIC4"
elif [ "$DATASET" == "mimiciii" ]; then
    CONFIG_DS="MIMIC3"
    CONFIG_PT="MIMIC3_TransE"
    CONFIG_TRAIN="HGT_Causal_MIMIC3"
else
    echo "Unknown DATASET: $DATASET. Use mimiciv or mimiciii."
    exit 1
fi

echo ""
echo ">>> Processing $DATASET in $MODE mode <<<"

echo "Step 1: Creating heterogeneous graph..."
$PYTHON_CMD run_graph_creation.py configs/construct_graph/${CONFIG_DS}.yml $CLEAN_FLAG

echo "Step 2: Pretraining node embeddings..."
$PYTHON_CMD run_pretrain.py ${CONFIG_PT}.yml $CLEAN_FLAG

echo "Step 3: Training models..."
# Run all 4 model variants in parallel on different GPUs if available
# We use & to run in background and wait at the end
# Note: On clusters like Narval, make sure you have enough GPUs allocated.
CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD run_train.py HGT_Causal_${CONFIG_DS}.yml $CLEAN_FLAG &
CUDA_VISIBLE_DEVICES=1 $PYTHON_CMD run_train.py HGT_${CONFIG_DS}.yml $CLEAN_FLAG &
CUDA_VISIBLE_DEVICES=2 $PYTHON_CMD run_train.py HGT_ST_${CONFIG_DS}.yml $CLEAN_FLAG &
CUDA_VISIBLE_DEVICES=3 $PYTHON_CMD run_train.py Baselines_${CONFIG_DS}.yml $CLEAN_FLAG &
wait

echo "Step 4: Running benchmarks..."
export BENCHMARK_DATASET=$CONFIG_DS
$PYTHON_CMD benchmark.py

echo ""
echo "Step 5: Generating final summary report and visualizations..."
$PYTHON_CMD generate_benchmark_report.py

echo ""
echo "================================================================"
echo "Pipeline execution for $DATASET ($MODE) completed successfully!"
echo "================================================================"
