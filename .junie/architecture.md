### Model Architecture Specification

The `md-MulT-EHR` project implements heterogeneous graph neural networks (GNNs) for multi-task learning on Electronic Health Records (EHR).

#### Core Model Components
- **Heterogeneous Graph Transformer (HGT)**: The default model architecture used for learning over multiple node and edge types in EHR.
- **Causal Graph Learning**: Implementation of causal mechanisms within GNNs (CausalGNN) for more robust and explainable EHR representations.
- **Multi-task Learning**: Jointly training on different medical prediction tasks (e.g., mortality, readmission).
- **Custom Layers**: `BLinear` and `BGraphConv` are used to handle Bayesian or specialized graph convolutional operations.

#### Data Flow and Pipeline
1. **Raw Data Processing**: Raw MIMIC-III/IV tables are processed via `get_graph.py` and `pyhealth` components into a graph format.
2. **Graph Construction**: A heterogeneous graph is built where nodes represent patients, diagnoses, procedures, etc., and edges represent medical relationships.
3. **Training (Trainer Classes)**:
    - `GNNTrainer`: Standard GNN training.
    - `CausalGNNTrainer`: Training with causal graph modeling.
    - `CausalSTGNNTrainer`: Spatio-temporal causal GNN modeling.
    - `BaselinesTrainer`: Benchmark training for non-graph baseline models.
4. **Evaluation**: Performance is assessed using metrics like AUROC, AUPRC, and F1 score, managed by `benchmark.py` and `trainers`.

#### Key Modules
- `models/`: Contains the actual GNN architectures (HGT, GAT, GCN, etc.).
- `trainers/`: Handles the training loops, loss calculation (via `losses.py`), and optimization.
- `explainers/`: Provides post-hoc explanations for model predictions using GNN explainer techniques.
- `gram/`: Leverages hierarchy in medical codes (ICD codes) for better representation learning.
