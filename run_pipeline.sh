#!/bin/bash

# Multi-task Heterogeneous Graph Learning on EHR - Full Pipeline Script
# This script automates the creation of graphs, pretraining of embeddings,
# training of models, and benchmarking.

set -e # Exit immediately if a command exits with a non-zero status.

echo "================================================================"
echo "Starting Multi-task Heterogeneous Graph Learning Pipeline"
echo "================================================================"

# 1. Create Graph (using pyhealth dev mode for speed)
echo ""
echo "Step 1: Creating heterogeneous graph (Dev Mode)..."
# You can change the config file here (e.g., configs/construct_graph/MIMIC4.yml)
python run_graph_creation_dev.py configs/construct_graph/MIMIC4.yml

# 2. Pretrain Embeddings
echo ""
echo "Step 2: Pretraining node embeddings..."
# You can change the pretrain config file here (e.g., MIMIC4_TransE.yml)
python run_pretrain.py MIMIC4_TransE.yml

# 3. Train Models
echo ""
echo "Step 3: Training models..."
# Exemplar training for HGT Causal model on MIMIC-IV
# You can add more models here by calling run_train.py with different configs
python run_train.py HGT_Causal_MIMIC4.yml

# 4. Benchmark and Obtain Results
echo ""
echo "Step 4: Running benchmarks..."
# This runs the benchmarking logic defined in benchmark.py
# Note: You might want to edit benchmark.py to select specific benchmarks to run.
python benchmark.py

echo ""
echo "================================================================"
echo "Pipeline execution completed successfully!"
echo "================================================================"
