# Graph Neural Network Models

This directory contains the implementations of various GNN architectures used for clinical prediction tasks on EHR graphs.

## Directory Contents

- `HGT.py`: Heterogeneous Graph Transformer, the core model proposed in the paper.
- `HetRGCN.py`: Heterogeneous Relational GCN implementation.
- `GNN.py`, `GAT.py`, `GCN.py`, `GIN.py`, `HAN.py`: Implementations of standard GNN models (Graph Attention Network, Graph Convolutional Network, Graph Isomorphism Network, and Heterogeneous Attention Network).
- `BGCN.py`: Bayesian GCN implementation for uncertainty quantification.
- `__init__.py`: Package initializer.

## Model Selection

Different models can be selected for training and evaluation by specifying them in the configuration YAML files under the `configs/` directory.
