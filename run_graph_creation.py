from pathlib import Path
from construct_graph import GraphConstructor
from utils import ordered_yaml
import yaml
import sys

def main(config_file="configs/construct_graph/MIMIC4.yml"):
    opt_path = Path(config_file)
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")

    # Set dev=False explicitly as per requirement (only data changes between dev and full)
    config["dev"] = False
    
    graph_constructor = GraphConstructor(config)
    
    print("Starting graph construction...")
    graph_constructor.load_mimic()
    graph_constructor.construct_graph()
    graph_constructor.set_tasks()
    graph_constructor.initialize_features()
    graph_constructor.save_graph()
    graph_constructor.save_mimic_dataset()
    print("Graph construction completed.")

if __name__ == '__main__':
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/construct_graph/MIMIC4.yml"
    main(config)
