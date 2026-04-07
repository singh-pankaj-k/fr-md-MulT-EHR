### Project Structure Specification

This document outlines the directory and file structure of the `md-MulT-EHR` project.

#### Root Directory Files
- `main.py`: Entry point for training and pretraining.
- `benchmark.py`: Benchmarking script for evaluating models.
- `checkpoint.py`: Utilities for saving and loading model checkpoints.
- `data.py`: Data loading and processing utilities.
- `get_graph.py`: Script to construct the heterogeneous graph from raw tabular data.
- `losses.py`: Custom loss functions.
- `parse.py`: Argument parsing for CLI.
- `utils.py`: General utility functions (YAML loading, etc.).
- `README.md`: Project documentation and citations.
- `visualizations.ipynb`: Jupyter notebook for data/result visualization.

#### Directories
- `configs/`: YAML configuration files for different models and datasets (MIMIC-III/IV).
    - `GNN/`: GNN-specific configurations.
    - `construct_graph/`: Graph construction parameters.
    - `pretrain/`: Pretraining configurations.
- `construct_graph/`: Python scripts for building the EHR graph.
- `data/`: Data storage directory (contains `graphs/` and `root/`).
- `explainers/`: Model explanation tools (GNN explainers).
- `gram/`: Implementation of GRAM (Graph-based Attention Model) for EHR.
- `layers/`: Custom neural network layers (BLinear, BGraphConv).
- `models/`: Model definitions (BGCN, GAT, GCN, GIN, GNN, HAN, HGT, HetRGCN).
- `pretrainers/`: Logic for pretraining node embeddings.
- `trainers/`: Training loops for various model types (GNN, CausalGNN, etc.).
- `references/`: Reference implementations and external libraries (e.g., PyHealth).
- `checkpoints/`: Directory where trained models are saved.
