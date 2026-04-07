# Custom Layers

This directory contains specialized neural network layers used within the project's models.

## Directory Contents

- `BGraphConv.py`: Bayesian Graph Convolutional layer implementation.
- `BLinear.py`: Bayesian Linear layer implementation.
- `module_wrapper.py`: Wrappers to help apply layers to different types of GNN inputs.
- `__init__.py`: Package initializer.

## Functionality

These layers provide the building blocks for creating more complex GNN architectures. The Bayesian layers (`BGraphConv`, `BLinear`) enable uncertainty estimation during prediction, which is crucial for medical decision-making tasks.
