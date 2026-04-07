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

    graph_constructor = GraphConstructor(config)
    
    # Patch load_mimic to use dev=True for pyhealth datasets
    original_load_mimic = graph_constructor.load_mimic
    
    def load_mimic_dev():
        from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
        raw_path = graph_constructor.config_graph["raw"]
        if "mimiciii" in raw_path:
            graph_constructor.dataset = MIMIC3Dataset(
                root=raw_path,
                tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS", "LABEVENTS"],
                code_mapping={},
                dev=True,  # Set dev=True for small fraction of data
            )
        elif "mimiciv" in raw_path:
            graph_constructor.dataset = MIMIC4Dataset(
                root=raw_path,
                tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
                code_mapping={},
                dev=True,  # Set dev=True for small fraction of data
            )
        else:
            raise NotImplementedError
    
    graph_constructor.load_mimic = load_mimic_dev
    
    print("Starting graph construction in dev mode...")
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
