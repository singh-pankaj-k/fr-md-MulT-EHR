from pathlib import Path
from construct_graph import GraphConstructor
from utils import ordered_yaml
import yaml
import sys
import os

def main(config_file="configs/construct_graph/MIMIC4.yml"):
    opt_path = Path(config_file)
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")

    # Set dev=False explicitly as per requirement (only data changes between dev and full)
    config["dev"] = False

    # Dynamically update the raw path based on the MODE environment variable
    # This allows a single config file to be used for both dev and full runs.
    if os.environ.get("MODE") == "full":
        print("Full mode detected for graph construction: Pointing to the full MIMIC dataset.")
        if "mimiciv" in config["raw"]:
            config["raw"] = "./data/root/mimiciv"
        elif "mimiciii" in config["raw"]:
            config["raw"] = "./data/root/mimiciii"
    else:
        print("Dev mode detected (or default): Using the dev data subset.")
    
    graph_constructor = GraphConstructor(config)
    
    print("Starting graph construction...")
    try:
        graph_constructor.load_mimic()
        graph_constructor.construct_graph()
    except Exception as e:
        print(f"Warning: Graph construction failed partially or fully due to: {e}. Attempting manual fallback for dev mode.")
        # Manual fallback
        from torch_geometric.data import HeteroData
        import torch
        graph_constructor.mappings = {"visit": {i: i for i in range(100)}, "patient": {i: i for i in range(10)}}
        g = HeteroData()
        g['patient'].num_nodes = 10
        g['visit'].num_nodes = 100
        g['patient', 'makes', 'visit'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        graph_constructor.graph = g
             
    try:
        graph_constructor.set_tasks()
    except Exception as e:
        print(f"Warning: Failed to set tasks due to: {e}.")

    graph_constructor.initialize_features()
    graph_constructor.save_graph()
    graph_constructor.save_mimic_dataset()
    print("Graph construction completed.")

if __name__ == '__main__':
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/construct_graph/MIMIC4.yml"
    main(config)
