#!/bin/bash

# Multi-task Heterogeneous Graph Learning on EHR - Full Pipeline Script
# This script automates the creation of graphs, pretraining of embeddings,
# training of models, and benchmarking for both MIMIC-III and MIMIC-IV.
# Supports MODE=dev (default) or MODE=full.

MODE=${MODE:-dev}
set -e # Exit immediately if a command exits with a non-zero status.

echo "================================================================"
echo "Starting Multi-task EHR Pipeline (MIMIC-IV and MIMIC-III)"
echo "MODE: $MODE"
echo "================================================================"

# Enable MPS fallback for better compatibility on macOS
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MODE=$MODE

# Set dev-mode hyperparameter defaults if in dev mode
if [ "$MODE" == "dev" ]; then
    export DEV_EPOCHS=${DEV_EPOCHS:-2}
    export DEV_BATCH_SIZE=${DEV_BATCH_SIZE:-1024}
    export DEV_SAMPLES=${DEV_SAMPLES:-100}
    export DEV_PRETRAIN_EPOCHS=${DEV_PRETRAIN_EPOCHS:-2}
fi

# --- MIMIC-IV Pipeline ---
echo ""
echo ">>> Processing MIMIC-IV <<<"

echo "Step 1: Creating heterogeneous graph (MIMIC-IV)..."
python run_graph_creation.py configs/construct_graph/MIMIC4.yml

echo "Step 2: Pretraining node embeddings (MIMIC-IV)..."
python run_pretrain.py MIMIC4_TransE.yml

echo "Step 3: Training all models (MIMIC-IV)..."
# Now run_train.py runs all models (Causal-GNN, GNN, Causal-GNN-ST, Baselines)
python run_train.py HGT_Causal_MIMIC4.yml

echo "Step 4: Running benchmarks (MIMIC-IV)..."
# benchmark.py now just does additional ablations if needed,
# but our main models are already trained by Step 3.
export BENCHMARK_DATASET=MIMIC4
python benchmark.py

# --- MIMIC-III Pipeline ---
echo ""
echo ">>> Processing MIMIC-III <<<"

echo "Step 1: Creating heterogeneous graph (MIMIC-III)..."
python run_graph_creation.py configs/construct_graph/MIMIC3.yml

echo "Step 2: Pretraining node embeddings (MIMIC-III)..."
python run_pretrain.py MIMIC3_TransE.yml

echo "Step 3: Training all models (MIMIC-III)..."
python run_train.py HGT_Causal_MIMIC3.yml

echo "Step 4: Running benchmarks (MIMIC-III)..."
export BENCHMARK_DATASET=MIMIC3
python benchmark.py

# --- Final Reporting ---
echo ""
echo "Step 5: Generating final summary report and visualizations..."
python generate_benchmark_report.py

echo ""
echo "================================================================"
echo "Full pipeline execution completed successfully!"
echo "================================================================"
