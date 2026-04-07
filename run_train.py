import sys
import yaml
from utils import load_config
from trainers import (
    GNNTrainer,
    CausalGNNTrainer,
    CausalSTGNNTrainer,
    BaselinesTrainer
)

def main(config_name="HGT_Causal_MIMIC3.yml"):
    config = load_config(config_name)
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
    config = sys.argv[1] if len(sys.argv) > 1 else "HGT_Causal_MIMIC3.yml"
    main(config)
