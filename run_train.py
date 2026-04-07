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

def main(config_name="HGT_Causal_MIMIC4.yml"):
    config = load_config(config_name)
    
    # Allow environment variable overrides for dev mode
    if os.environ.get("MODE") == "dev":
        print("Dev mode detected: Overriding hyperparameters for fast execution.")
        config["train"]["num_epochs"] = int(os.environ.get("DEV_EPOCHS", 2))
        config["train"]["batch_size"] = int(os.environ.get("DEV_BATCH_SIZE", 1024))
        if "n_samples" in config["train"]:
            config["train"]["n_samples"] = int(os.environ.get("DEV_SAMPLES", 100))
    
    print(f"Starting training with {config_name}")

    if config["train_type"] == "gnn":
        trainer = GNNTrainer(config)
    elif config["train_type"] == "causal-gnn":
        trainer = CausalGNNTrainer(config)
    elif config["train_type"] == "causal-gnn-st":
        trainer = CausalSTGNNTrainer(config)
    elif config["train_type"] == "baseline":
        trainer = BaselinesTrainer(config)
    else:
        raise NotImplementedError("This type of model is not implemented")
    
    trainer.train()
    print("Training completed.")

if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "HGT_Causal_MIMIC4.yml"
    main(config_arg)
