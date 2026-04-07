#!/bin/bash

# Multi-task Heterogeneous Graph Learning on EHR - Full Pipeline Script
# This script automates the creation of graphs, pretraining of embeddings,
# training of models, and benchmarking for both MIMIC-III and MIMIC-IV.

set -e # Exit immediately if a command exits with a non-zero status.

echo "================================================================"
echo "Starting Full Multi-task EHR Pipeline (MIMIC-IV and MIMIC-III)"
echo "================================================================"

# Enable MPS fallback for better compatibility on macOS
export PYTORCH_ENABLE_MPS_FALLBACK=1

# --- MIMIC-IV Pipeline ---
echo ""
echo ">>> Processing MIMIC-IV <<<"

echo "Step 1: Creating heterogeneous graph (MIMIC-IV)..."
python run_graph_creation.py configs/construct_graph/MIMIC4.yml

echo "Step 2: Pretraining node embeddings (MIMIC-IV)..."
python run_pretrain.py MIMIC4_TransE.yml

echo "Step 3: Training models (MIMIC-IV)..."
# Main HGT Causal model
python run_train.py HGT_Causal_MIMIC4.yml
# Other models can be added here if needed for single run
# python run_train.py GCN_Causal_MIMIC4.yml
# python run_train.py GAT_Causal_MIMIC4.yml

echo "Step 4: Running benchmarks (MIMIC-IV)..."
# Note: benchmark.py needs to be configured to run desired benchmarks
# For now, we'll use a modified version or ensure it runs what we want.
export BENCHMARK_DATASET=MIMIC4
python benchmark.py

# --- MIMIC-III Pipeline ---
echo ""
echo ">>> Processing MIMIC-III <<<"

echo "Step 1: Creating heterogeneous graph (MIMIC-III)..."
python run_graph_creation.py configs/construct_graph/MIMIC3.yml

echo "Step 2: Pretraining node embeddings (MIMIC-III)..."
python run_pretrain.py MIMIC3_TransE.yml

echo "Step 3: Training models (MIMIC-III)..."
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
