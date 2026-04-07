### Execution Guide Specification

This guide explains how to run the multi-task heterogeneous graph learning pipeline for EHR.

#### Configuration
Settings are managed through YAML files in the `configs/` directory.

#### Graph Construction
Before training, the heterogeneous graph must be constructed from tabular EHR data (MIMIC-III or MIMIC-IV).
```bash
python get_graph.py
```
*Modify `configs/construct_graph/MIMIC3.yml` or `MIMIC4.yml` to specify data paths.*

#### Node Embedding Pretraining
Optionally, pretrain node embeddings (e.g., using TransE).
1. Set `mode = "pretrain"` in `main.py`.
2. Ensure `config_name` in `main.main()` points to a file in `./configs/pretrain/`.
3. Run:
```bash
python main.py
```

#### Training and Evaluation
The primary way to train and evaluate models is through `main.py`.
1. Set `mode = "train"` in `main.py`.
2. Configure `config_name` in `main.main()` (e.g., `HGT_Causal_MIMIC4.yml`).
3. Run:
```bash
python main.py
```
*Models are automatically evaluated after every epoch, and performance is recorded.*

#### Benchmarking
To perform comprehensive benchmarking:
```bash
python benchmark.py
```
*Ensure relevant configuration files for benchmarking are correctly set up in `configs/`.*

#### Experiment Tracking
Results are managed by `wandb`. Ensure `wandb` is enabled in the configuration if online tracking is desired.
