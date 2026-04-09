import os
import yaml
from utils import ordered_yaml

import pickle
import random
import torch
import wandb

from trainers import (
    GNNTrainer,
    CausalGNNTrainer,
    BaselinesTrainer
)


def apply_dev_mode(config):
    if os.environ.get("MODE") == "dev":
        print(f"Dev mode detected: Overriding hyperparameters for fast execution.")
        config["train"]["num_epochs"] = int(os.environ.get("DEV_EPOCHS", 2))
        config["train"]["batch_size"] = int(os.environ.get("DEV_BATCH_SIZE", 1024))
        if "n_samples" in config["train"]:
            config["train"]["n_samples"] = int(os.environ.get("DEV_SAMPLES", 100))
    return config


def benchmark_baselines(dataset="MIMIC4"):
    # Load config based on dataset
    config_file = f"Baselines_{dataset}.yml"
    config_path = f"./configs/{config_file}"
    with open(config_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
    
    config = apply_dev_mode(config)

    # Initialize baseline models
    with open(config["datasets"]["dataset_path"], 'rb') as inp:
        unp = pickle.Unpickler(inp)
        mimic_base = unp.load()

    for method in [
        "DrAgent",
        "StageNet",
        "AdaCare",
        "Transformer",
        "RNN",
        "ConCare",
        "GRSAP",
        "Deepr",
        "MICRON",
        "GAMENet",
        "MoleRec",
        "SafeDrug",
        "SparcNet",
    ]:
        for task in [
            "readm",
            "mort_pred",
            "los",
            "drug_rec"
        ]:
            config["train"]["baseline_name"] = method
            config["train"]["task"] = task
            dataset_name = config["datasets"]["name"]
            config["checkpoint"]["path"] = f"./checkpoints/{method}/{dataset_name}/{task}/"
            print(f"Training {method} on task {task}")

            try:
                trainer = BaselinesTrainer(config, mimic_base)
                trainer.train()
                del trainer
            except Exception as e:
                print(f"Error training {method} on task {task}: {e}")
                import traceback
                traceback.print_exc()


def benchmark_gnns(dataset="MIMIC4"):
    # Load base config
    config_file = f"HGT_Causal_{dataset}.yml"
    config_path = f"./configs/{config_file}"
    with open(config_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
    
    config = apply_dev_mode(config)

    # Load GNN specific configs
    gnn_config_file = f"GNN_MIMIC4_Configs.yml" if dataset == "MIMIC4" else "GNN_Configs.yml"
    with open(f"./configs/GNN/{gnn_config_file}", mode='r') as f:
        loader, _ = ordered_yaml()
        gnn_config = yaml.load(f, loader)

    for archi in [
        "GCN",
        "GAT",
        "GIN",
        "HetRGCN",
        "HGT"
    ]:
        print(f"Training {archi}")
        try:
            config["GNN"] = gnn_config[archi]
            dataset_name = config["datasets"]["name"]
            config["name"] = f"{archi}_MTCausal_MIMIC{dataset_name[-1]}_RMDL"
            config["checkpoint"]["path"] = f"./checkpoints/GNN_ablation/{dataset_name}/{archi}/"
            config["logging"]["tags"] += [archi]

            trainer = CausalGNNTrainer(config)
            trainer.train()
            wandb.finish()
            del trainer
        except Exception as e:
            print(f"Error training {archi}: {e}")
            import traceback
            traceback.print_exc()


def benchmark_dropouts(dataset="MIMIC4"):
    config_file = f"HGT_Causal_{dataset}.yml"
    config_path = f"./configs/{config_file}"
    with open(config_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
    
    config = apply_dev_mode(config)

    for dp in [
        0.1, 0.3, 0.5
    ]:
        print(f"Training dropout {dp}")
        try:
            config["GNN"]["feat_drop"] = dp
            config["name"] = f"HGT_MTCausal_{dataset}_RMDL_dp{dp}"
            dataset_name = config["datasets"]["name"]
            config["checkpoint"]["path"] = f"./checkpoints/Dropout_ablation/{dataset_name}/{dp}/"
            config["logging"]["tags"] += ["abl_dropout"]

            trainer = CausalGNNTrainer(config)
            trainer.train()
            wandb.finish()
            del trainer
        except Exception as e:
            print(f"Error training dropout {dp}: {e}")
            import traceback
            traceback.print_exc()


def benchmark_hidden_dim(dataset="MIMIC4"):
    config_file = f"HGT_Causal_{dataset}.yml"
    config_path = f"./configs/{config_file}"
    with open(config_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
    
    config = apply_dev_mode(config)

    for dim in [
        32, 64
    ]:
        print(f"Training hidden dim {dim}")
        try:
            config["GNN"]["hidden_dim"] = dim
            config["name"] = f"HGT_MTCausal_{dataset}_RMDL_dim{dim}"
            dataset_name = config["datasets"]["name"]
            config["checkpoint"]["path"] = f"./checkpoints/Hidden_Dim_ablation/{dataset_name}/{dim}/"
            config["logging"]["tags"] += ["abl_dim"]

            trainer = CausalGNNTrainer(config)
            trainer.train()
            wandb.finish()
            del trainer
        except Exception as e:
            print(f"Error training hidden dim {dim}: {e}")
            import traceback
            traceback.print_exc()


# Set seed
seed = 611
random.seed(seed)
torch.manual_seed(seed)

if __name__ == "__main__":
    dataset = os.environ.get("BENCHMARK_DATASET", "MIMIC4")
    print(f"Running benchmarks for {dataset}")
    
    benchmark_gnns(dataset)
    benchmark_dropouts(dataset)
    benchmark_hidden_dim(dataset)
    benchmark_baselines(dataset)