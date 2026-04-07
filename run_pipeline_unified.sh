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

# Step 0: Optionally clean previous results
if [ "$CLEAN" == "true" ]; then
    echo "Cleaning previous output and artifacts..."
    rm -rf checkpoints/*
    rm -rf data/graphs/*
    rm -rf data/dataset_objects/*
    rm -rf data/mimic3_objects/*
    rm -rf benchmark/*
    
    # Recreate necessary directories
    mkdir -p data/graphs data/dataset_objects data/mimic3_objects benchmark/plots
    echo "Done cleaning."
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

# Determine config path and modify if MODE=dev
# Note: we previously ensured configs point to dev_... folders for dev mode.
# For full mode, we should ideally have separate configs or modify them on the fly.
# Currently, the provided configs for MIMIC4 point to dev_mimiciv.
# Let's assume for this session that we are running in dev mode as requested.

echo ""
echo ">>> Processing $DATASET in $MODE mode <<<"

echo "Step 1: Creating heterogeneous graph..."
python run_graph_creation.py configs/construct_graph/${CONFIG_DS}.yml

echo "Step 2: Pretraining node embeddings..."
python run_pretrain.py ${CONFIG_PT}.yml

echo "Step 3: Training models..."
python run_train.py ${CONFIG_TRAIN}.yml

echo "Step 4: Running benchmarks..."
export BENCHMARK_DATASET=$CONFIG_DS
python benchmark.py

echo ""
echo "Step 5: Generating final summary report and visualizations..."
python generate_benchmark_report.py

echo ""
echo "================================================================"
echo "Pipeline execution for $DATASET ($MODE) completed successfully!"
echo "================================================================"
