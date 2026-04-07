# [Neural Networks'24] Multi-task Heterogeneous Graph Learning on Electronic Health Records

![image](https://github.com/user-attachments/assets/6dd5a971-8e42-48e5-a807-3a026db75286)

This repository contains the implementation for the paper "Multi-task heterogeneous graph learning on electronic health records" published in *Neural Networks* (2024).

## Project Overview

The project focuses on building a multi-task heterogeneous Graph Neural Network (GNN) framework for Electronic Health Records (EHR) data. It leverages patient-centric graphs to perform multiple clinical prediction tasks simultaneously, such as mortality prediction, readmission prediction, and length of stay.

## Directory Structure

- `checkpoints/`: Pre-trained model weights and baseline checkpoints.
- `configs/`: YAML configuration files for model hyperparameters and dataset settings.
- `construct_graph/`: Logic for constructing heterogeneous graphs from EHR data.
- `data/`: Directory for storing raw data and processed graph objects (`.pkl`).
- `explainers/`: Scripts for model interpretability and graph-based explanations.
- `gram/`: Implementation of the Graph-based Attention Model (GRAM) baseline.
- `layers/`: Custom neural network layers, including Bayesian and GNN components.
- `models/`: Implementations of various GNN architectures (HGT, GAT, GCN, etc.).
- `pretrainers/`: Modules for pre-training models on clinical data.
- `trainers/`: Core training loops and orchestration logic.
- `references/`: Supporting documentation, external codebases, and scripts.

## Installation

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

Note: DGL is not used in the main pipeline to ensure compatibility with macOS (Apple Silicon). We use **PyTorch Geometric (PyG)** as the graph learning backend.

## Dataset

We use the MIMIC-III and MIMIC-IV datasets to benchmark our method.

The download of the data is available at [PhysioNet](https://physionet.org/). You need to complete a short course to obtain access as required by the data issuer. Once you download the data in tabular form, you can construct the graph using `get_graph.py`.

## Training and Testing

The processed data (in `pkl` formats) will be stored in the respective subdirectory under the `data` folder. You may call `main.py` to start a training. Exemplar training configurations are provided in `./configs` in yaml formats. Benchmarking is also available in `benchmark.py`. Testing performance will be recorded after every epoch. We adopt `wandb` for results management and results will be uploaded to `wandb` online if switched on. 

### Quick Start

```bash
python main.py --config configs/HGT_MT_MIMIC4.yml
```

## Acknowledgement

We implement our method based on the [pyhealth](https://pyhealth.readthedocs.io/en/latest/) package.

## Citations

If you find our work useful, please cite us at:

```
@article{chan2024multi,
  title={Multi-task heterogeneous graph learning on electronic health records},
  author={Chan, Tsai Hor and Yin, Guosheng and Bae, Kyongtae and Yu, Lequan},
  journal={Neural Networks},
  pages={106644},
  year={2024},
  publisher={Elsevier}
}
```
