import os
import yaml
import pickle
from utils import ordered_yaml
from trainers import BaselinesTrainer

def test_single_baseline():
    os.environ["MODE"] = "dev"
    config_path = "./configs/Baselines_MIMIC4.yml"
    with open(config_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)

    # Apply dev mode
    config["train"]["num_epochs"] = 1
    config["train"]["batch_size"] = 1024
    
    with open(config["datasets"]["dataset_path"], 'rb') as inp:
        mimic_base = pickle.load(inp)

    method = "AdaCare"
    task = "mort_pred"
    
    config["train"]["baseline_name"] = method
    config["train"]["task"] = task
    config["checkpoint"]["path"] = f"./checkpoints/test_{method}_{task}/"
    
    print(f"Testing {method} on {task}")
    trainer = BaselinesTrainer(config, mimic_base)
    trainer.train()
    print("Test completed successfully")

if __name__ == "__main__":
    test_single_baseline()
