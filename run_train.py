import sys
import os
import argparse
import shutil
from utils import load_config
from trainers import (
    GNNTrainer,
    CausalGNNTrainer,
    CausalSTGNNTrainer,
    BaselinesTrainer
)

def main():
    parser = argparse.ArgumentParser(description="Run EHR training pipeline.")
    parser.add_argument("config", nargs="?", default="HGT_Causal_MIMIC4.yml", help="Configuration file name.")
    parser.add_argument("-c", "--clean", action="store_true", help="Start training from scratch by deleting old checkpoints.")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Handle clean start by removing checkpoint directory if it exists
    if args.clean and "checkpoint" in config and "path" in config["checkpoint"]:
        checkpoint_path = config["checkpoint"]["path"]
        if os.path.exists(checkpoint_path):
            print(f"Cleaning previous checkpoints at {checkpoint_path}...")
            shutil.rmtree(checkpoint_path)

    # Allow environment variable overrides for dev mode
    if os.environ.get("MODE") == "dev":
        print(f"Dev mode detected: Overriding hyperparameters for fast execution.")
        config["train"]["num_epochs"] = int(os.environ.get("DEV_EPOCHS", 2))
        config["train"]["batch_size"] = int(os.environ.get("DEV_BATCH_SIZE", 1024))
        if "n_samples" in config["train"]:
            config["train"]["n_samples"] = int(os.environ.get("DEV_SAMPLES", 100))
    
    print(f"Starting training with {args.config} (type: {config['train_type']})")

    if config["train_type"] == "gnn":
        trainer = GNNTrainer(config)
    elif config["train_type"] == "causal-gnn":
        trainer = CausalGNNTrainer(config)
    elif config["train_type"] == "causal-gnn-st":
        trainer = CausalSTGNNTrainer(config)
    elif config["train_type"] == "baseline":
        import pickle
        dataset_path = config["datasets"]["dataset_path"]
        with open(dataset_path, "rb") as f:
            base_dataset = pickle.load(f)
        trainer = BaselinesTrainer(config, base_dataset)
    else:
        print(f"Error: train_type '{config['train_type']}' not recognized.")
        sys.exit(1)
    
    trainer.train()
    print(f"Training completed for {args.config}.")

if __name__ == "__main__":
    main()
