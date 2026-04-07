# Data Directory

This directory is used for storing raw and processed datasets, including the constructed heterogeneous graph objects.

## Directory Contents

- `graphs/`: Subdirectory for storing processed graph objects (`.pkl`, `.pt`, etc.).
- `root/`: Potential directory for storing raw MIMIC-III/IV tabular files.

## Usage

When you run `get_graph.py`, the generated graph data will be saved into the `graphs/` subdirectory. These saved objects are later loaded by `data.py` or directly by the `trainers/` during the model training process.
