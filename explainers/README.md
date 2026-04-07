# Model Explainability

This directory contains tools and scripts for interpreting and explaining GNN model predictions.

## Directory Contents

- `explainer.py`: Generic base class or utilities for model explanation.
- `gc_explainer.py`: Implementation of a graph-based explainer (e.g., GCExplainer) to identify influential nodes/edges for a prediction.
- `__init__.py`: Package initializer.

## Usage

Explainers are used to provide insights into why the model predicted a certain clinical outcome for a patient, such as identifying key medical codes (diagnoses or medications) that contributed to the mortality risk or length of stay prediction.
