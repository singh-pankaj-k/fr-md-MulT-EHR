# Pretraining

This directory contains scripts and logic for pre-training GNN models.

## Directory Contents

- `pretrainer.py`: Script to execute the pre-training task (e.g., using masked node feature prediction).
- `__init__.py`: Package initializer.

## Purpose

Pre-training on medical domain data allows the model to learn a good initial representation of nodes (patients, medical codes) before performing specific downstream clinical tasks. This can lead to better performance and faster convergence.
