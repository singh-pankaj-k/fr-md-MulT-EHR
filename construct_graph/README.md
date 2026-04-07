# Graph Construction

This directory contains the logic for constructing heterogeneous graphs from Electronic Health Records (EHR) data.

## Directory Contents

- `graph_constructor.py`: The core script that processes raw EHR tables (from MIMIC-III or MIMIC-IV) and builds a graph representation.
- `__init__.py`: Package initializer.

## Workflow

The graph construction typically involves:
1. Parsing tabular EHR data (patients, admissions, diagnoses, medications, etc.).
2. Mapping medical codes to unique identifiers.
3. Defining node types (e.g., Patient, Admission, Medical Code) and edge types (e.g., ADMITTED_TO, HAS_DIAGNOSIS).
4. Creating a heterogeneous graph object compatible with PyTorch Geometric (PyG) or Deep Graph Library (DGL).

This process is usually triggered by `get_graph.py` in the root directory.
