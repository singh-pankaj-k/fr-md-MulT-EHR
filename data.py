import numpy as np
import pickle
import torch
from torch_geometric.data import HeteroData


def load_graph(graph_path, labels_path, feat_dim=128, pretrained=None):

    # Load graph from cache
    with open(graph_path, 'rb') as inp:
        unp = pickle.Unpickler(inp)
        g = unp.load()

    # Load labels
    try:
        with open(labels_path, 'rb') as inp:
            unp = pickle.Unpickler(inp)
            labels = unp.load()
    except FileNotFoundError:
        print(f"Warning: labels file {labels_path} not found. Returning empty labels.")
        labels = {"mort_pred": {}, "drug_rec": {}, "all_drugs": [], "los": {}, "readm": {}}

    n_node_types = len(g.node_types)

    # Set masks for entities
    if not pretrained:
        for tp in g.node_types:
            n_nodes = g[tp].num_nodes if tp in g.node_types else 0
            if n_nodes == 0 and tp in ["patient", "visit", "diagnosis", "procedure", "prescription"]:
                # In case num_nodes is not inferred, we might need another way to get it
                # or ensure it's set during construction
                pass

            # Initialize features
            feat = torch.randn(n_nodes, feat_dim)
            g[tp].x = feat
    else:
        with open(pretrained, 'rb') as inp:
            unp = pickle.Unpickler(inp)
            pre_g = unp.load()

            for tp in g.node_types:
                if tp in pre_g.node_types:
                    g[tp].x = pre_g[tp].x.clone()
            del pre_g

    # Arrange masks by tasks
    train_masks = {}
    test_masks = {}
    
    # Set seed for reproducible split
    np.random.seed(42)
    
    for k, lb in labels.items():
        if k == "all_drugs":
            train_masks.update({k: lb})
            test_masks.update({k: lb})
            continue
            
        all_visits = np.array(list(lb.keys()))
        if k == "drug_rec":
            # For multi-label, just do random split but with fixed seed
            indices = np.random.permutation(len(lb))
            split = int(0.9 * len(lb))
            train_visits = all_visits[indices[:split]]
            test_visits = all_visits[indices[split:]]
        else:
            # Stratified split for binary/multiclass tasks
            all_labels = np.array([lb[v] for v in all_visits])
            unique_labels = np.unique(all_labels)
            train_visits_list = []
            test_visits_list = []
            
            for label in unique_labels:
                label_indices = np.where(all_labels == label)[0]
                np.random.shuffle(label_indices)
                # Use 90/10 split
                split_idx = int(0.9 * len(label_indices))
                
                # Ensure at least one sample in test if possible for better metrics
                if split_idx == len(label_indices) and len(label_indices) >= 2:
                    split_idx = len(label_indices) - 1
                
                train_visits_list.extend(all_visits[label_indices[:split_idx]])
                test_visits_list.extend(all_visits[label_indices[split_idx:]])
            
            train_visits = np.array(train_visits_list)
            test_visits = np.array(test_visits_list)

        train_masks.update({k: train_visits})
        test_masks.update({k: test_visits})

    return g, labels, train_masks, test_masks