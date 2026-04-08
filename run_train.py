import sys
import os
import yaml
from utils import load_config
from trainers import (
    GNNTrainer,
    CausalGNNTrainer,
    CausalSTGNNTrainer,
    BaselinesTrainer
)

def run_model(config_name, train_type_override=None):
    config = load_config(config_name)
    
    if train_type_override:
        config["train_type"] = train_type_override
        # Update name and checkpoint path to avoid collisions
        config["name"] = f"{config['name']}_{train_type_override}"
        if "checkpoint" in config:
            config["checkpoint"]["path"] = f"{config['checkpoint']['path']}_{train_type_override}"

    # Allow environment variable overrides for dev mode
    if os.environ.get("MODE") == "dev":
        print(f"Dev mode detected for {config['train_type']}: Overriding hyperparameters for fast execution.")
        config["train"]["num_epochs"] = int(os.environ.get("DEV_EPOCHS", 2))
        config["train"]["batch_size"] = int(os.environ.get("DEV_BATCH_SIZE", 1024))
        if "n_samples" in config["train"]:
            config["train"]["n_samples"] = int(os.environ.get("DEV_SAMPLES", 100))
    
    print(f"Starting training with {config_name} (type: {config['train_type']})")

    if config["train_type"] == "gnn":
        trainer = GNNTrainer(config)
    elif config["train_type"] == "causal-gnn":
        trainer = CausalGNNTrainer(config)
    elif config["train_type"] == "causal-gnn-st":
        trainer = CausalSTGNNTrainer(config)
    elif config["train_type"] == "baseline":
        # Baselines require the base dataset object
        import pickle
        dataset_path = config["datasets"]["dataset_path"]
        print(f"Loading base dataset from {dataset_path} for BaselinesTrainer...")
        with open(dataset_path, "rb") as f:
            base_dataset = pickle.load(f)
        trainer = BaselinesTrainer(config, base_dataset)
    else:
        print(f"Warning: train_type '{config['train_type']}' not implemented. Skipping.")
        return
    
    trainer.train()
    print(f"Training completed for {config['train_type']}.")

def main(config_name="HGT_Causal_MIMIC4.yml"):
    # The user wants to run all models one after another.
    # We will use the provided config as a base and run all 4 types if applicable,
    # or look for corresponding config files.
    
    dataset = "MIMIC4" if "MIMIC4" in config_name else "MIMIC3"
    
    print(f"Running all models for {dataset}...")
    
    # 1. Causal GNN (Main)
    run_model(f"HGT_Causal_{dataset}.yml")
    
    # 2. Standard GNN
    run_model(f"HGT_{dataset}.yml")
    
    # 3. Causal GNN ST
    run_model(f"HGT_ST_{dataset}.yml")
    
    # 4. Baselines
    run_model(f"Baselines_{dataset}.yml")

if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "HGT_Causal_MIMIC4.yml"
    main(config_arg)
