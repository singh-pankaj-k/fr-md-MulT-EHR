import sys
import yaml
from utils import load_config
from pretrainers import Pretrainer

def main(config_name="MIMIC3_TransE.yml"):
    config = load_config(config_name, "./configs/pretrain/")
    print(f"Starting pretraining with {config_name}")
    pretrainer = Pretrainer(config)
    pretrainer.train()
    print("Pretraining completed.")

if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "MIMIC3_TransE.yml"
    main(config)
