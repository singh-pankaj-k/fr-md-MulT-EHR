import sys
import os
import argparse
from utils import load_config
from pretrainers import Pretrainer

def main():
    parser = argparse.ArgumentParser(description="Pretrain EHR node embeddings.")
    parser.add_argument("config", nargs="?", default="MIMIC4_TransE.yml", help="Configuration file name.")
    parser.add_argument("-c", "--clean", action="store_true", help="Start pretraining from scratch by deleting existing graph embedding outputs.")
    args = parser.parse_args()

    config = load_config(args.config, "./configs/pretrain/")
    
    # Handle clean start
    if args.clean and "graph_output_path" in config:
        output_path = config["graph_output_path"]
        if os.path.exists(output_path):
            print(f"Cleaning previous pretrained graph at {output_path}...")
            os.remove(output_path)

    # Allow environment variable overrides for dev mode
    if os.environ.get("MODE") == "dev":
        print("Dev mode detected: Overriding pretrain epochs for fast execution.")
        config["n_epoch"] = int(os.environ.get("DEV_PRETRAIN_EPOCHS", 2))
    
    print(f"Starting pretraining with {args.config}")
    pretrainer = Pretrainer(config)
    pretrainer.train()
    print("Pretraining completed.")

if __name__ == "__main__":
    main()
