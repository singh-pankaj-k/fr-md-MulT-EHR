import sys
import os
import yaml
from utils import load_config
from pretrainers import Pretrainer

def main(config_name="MIMIC4_TransE.yml"):
    config = load_config(config_name, "./configs/pretrain/")
    
    # Allow environment variable overrides for dev mode
    if os.environ.get("MODE") == "dev":
        print("Dev mode detected: Overriding pretrain epochs for fast execution.")
        config["n_epoch"] = int(os.environ.get("DEV_PRETRAIN_EPOCHS", 2))
    
    print(f"Starting pretraining with {config_name}")
    pretrainer = Pretrainer(config)
    pretrainer.train()
    print("Pretraining completed.")

if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else "MIMIC4_TransE.yml"
    main(config_arg)
