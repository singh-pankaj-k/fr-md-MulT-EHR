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
    with open(labels_path, 'rb') as inp:
        unp = pickle.Unpickler(inp)
        labels = unp.load()

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
    for k, lb in labels.items():
        if k == "all_drugs":
            train_masks.update({k: lb})
            test_masks.update({k: lb})
            continue
        indices = np.random.permutation(len(lb))
        split = int(0.9 * len(lb))

        all_visits = np.array([k for k in lb.keys()])
        train_visits = all_visits[indices[:split]]
        test_visits = all_visits[indices[split:]]

        train_masks.update({k: train_visits})
        test_masks.update({k: test_visits})

    return g, labels, train_masks, test_masks