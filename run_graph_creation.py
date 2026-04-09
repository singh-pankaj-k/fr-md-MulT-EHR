import sys
import os
import argparse
import shutil
import yaml
from pathlib import Path
from construct_graph import GraphConstructor
from utils import ordered_yaml

def main():
    parser = argparse.ArgumentParser(description="Create heterogeneous EHR graph.")
    parser.add_argument("config", nargs="?", default="configs/construct_graph/MIMIC4.yml", help="Configuration file path.")
    parser.add_argument("-c", "--clean", action="store_true", help="Delete previous graph and dataset objects before starting.")
    args = parser.parse_args()

    opt_path = Path(args.config)
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)

    # Handle clean start by removing processed data paths if they exist
    dataset_name = config.get("dataset_name", "")
    if args.clean:
        for key in ["processed_path", "graph_output_path"]:
            if key in config:
                path = config[key]
                if os.path.exists(path) and os.path.isdir(path):
                    print(f"Cleaning previous artifacts for {dataset_name} in {path}...")
                    for f in os.listdir(path):
                        if f.startswith(dataset_name):
                            file_path = os.path.join(path, f)
                            if os.path.isfile(file_path):
                                os.remove(file_path)

    # Set dev=False explicitly as per requirement
    config["dev"] = False

    # Dynamically update the raw path based on the MODE environment variable
    if os.environ.get("MODE") == "full":
        if "mimiciv" in config["raw"]:
            config["raw"] = "./data/root/mimiciv"
        elif "mimiciii" in config["raw"]:
            config["raw"] = "./data/root/mimiciii"

    print(f"Starting graph construction with {args.config}")
    graph_constructor = GraphConstructor(config)
    graph_constructor.load_mimic()
    graph_constructor.construct_graph()
    graph_constructor.set_tasks()
    graph_constructor.initialize_features()
    graph_constructor.save_graph()
    graph_constructor.save_mimic_dataset()
    print("Graph construction completed.")

if __name__ == '__main__':
    main()
