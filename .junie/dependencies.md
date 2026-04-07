### Dependencies Specification

The `md-MulT-EHR` project is built using Python and several deep learning and data processing libraries.

#### Core Requirements
- `python` >= 3.8
- `torch`: Deep learning framework (PyTorch).
- `pyhealth`: Library for EHR data processing and modeling.
- `wandb`: Weight and Biases for experiment tracking.
- `yaml`: For configuration file management (`PyYAML`).
- `numpy`, `pandas`: Data manipulation.
- `scikit-learn`: Evaluation metrics.
- `matplotlib`, `seaborn`: Visualization.

#### Domain-Specific Libraries
- `torch-geometric`: For GNN implementations (likely used as a base for custom layers).
- `gram`: Integrated version of GRAM for EHR modeling.

#### Installation (Estimated)
```bash
pip install torch pyhealth wandb PyYAML numpy pandas scikit-learn matplotlib seaborn
# Additional dependencies for graph neural networks
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

*Note: Ensure `wandb` is logged in if experiment tracking is enabled in configuration.*
