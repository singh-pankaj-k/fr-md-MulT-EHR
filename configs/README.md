# Configurations

This directory contains YAML configuration files used to define hyperparameters, dataset settings, and model architectures for training and evaluation.

## Directory Contents

- `Baselines_MIMIC3.yml`, `Baselines_MIMIC4.yml`: Configurations for baseline models on MIMIC-III and MIMIC-IV datasets.
- `GAT_*.yml`, `GCN_*.yml`, `GIN_*.yml`, `HAN_*.yml`: Configurations for specific GNN architectures (Graph Attention Network, Graph Convolutional Network, Graph Isomorphism Network, Heterogeneous Attention Network).
- `HGT_*.yml`: Configurations for Heterogeneous Graph Transformer (HGT) models, including Single-Task (ST) and Multi-Task (MT) variants.
- `*_Causal_*.yml`: Configurations for causal GNN variants.
- `GNN/`: Subdirectory containing base GNN configurations.
- `construct_graph/`: Configurations for graph construction processes.
- `pretrain/`: Configurations for model pre-training phases.

## Usage

You can specify a configuration file when running `main.py`:

```bash
python main.py --config configs/HGT_MT_MIMIC4.yml
```
