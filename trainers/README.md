# Trainers

This directory contains the core orchestration logic for training, evaluating, and testing models.

## Directory Contents

- `trainer.py`: Generic base class or utilities for training.
- `train_gnn.py`, `train_causal_gnn.py`, `train_causal_gnn_st.py`: Training scripts for different GNN variants.
- `train_baselines.py`: Scripts for training baseline models (e.g., Deepr, StageNet).
- `__init__.py`: Package initializer.

## Functionality

These scripts manage the training loop, including:
1. Loading datasets and models.
2. Handling optimizer and loss function setup.
3. Managing training epochs, logging metrics, and checkpointing.
4. Orchestrating multi-task evaluation for heterogeneous graphs.
