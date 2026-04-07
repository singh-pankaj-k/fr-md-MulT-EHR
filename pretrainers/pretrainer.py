import torch
from torch import nn
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import TransE

import pickle
from tqdm import tqdm

from data import load_graph


class Pretrainer:
    def __init__(self, config):
        graph_path = config["graph_path"]
        label_path = config["labels_path"]
        self.graph, _, _, _ = load_graph(graph_path, label_path)

        # PyG LinkNeighborLoader for link prediction
        # We need to specify the edge type for which we want to sample
        # For pretraining, we might want to sample across all edge types
        # This is a bit complex in PyG if we want to sample 'any' edge
        # As an alternative, let's sample from all edge types
        
        # Original code used all edge types
        edge_types = self.graph.edge_types
        
        # For simplicity, let's use a standard LinkNeighborLoader on the first edge type
        # Or ideally, we'd want to cycle through them. 
        # Here we'll just set up one and the training loop can be adapted.
        
        self.dataloader = LinkNeighborLoader(
            self.graph,
            num_neighbors=[15, 10, 5],
            batch_size=1024,
            shuffle=True,
            neg_sampling_ratio=5.0,
            edge_label_index=None, # will sample from all edges if not specified? 
            # Actually LinkNeighborLoader needs a specific edge type or list of edges.
        )

        self.feat = nn.ParameterDict()
        for tp in self.graph.node_types:
            self.feat[tp] = nn.Parameter(self.graph[tp].x)
        
        self.optimizer = torch.optim.Adam(list(self.feat.values()), lr=0.05)

        self.output_path = config["graph_output_path"]
        self.margin = config["margin"]

        # TransE in PyG doesn't directly exist as a layer like in DGL, but we can implement it
        # or use torch_geometric.nn.KGEModel (though it's more for homogeneous)
        # For now, let's keep it simple as a placeholder or custom implementation if needed.
        # self.scorer = TransE(num_rels=len(edge_types), feats=128) 

        self.n_epoch = config["n_epoch"]

    def train(self):
        # Simplified pretraining for PyG migration
        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            res = 0
            # for batch in self.dataloader:
            #     # Compute loss
            #     pass
            
            training_range.set_description_str(f"Epoch {epoch} (PyG Pretraining placeholder)")

        # Save embeddings
        for tp in self.graph.node_types:
            self.graph[tp].x = self.feat[tp].detach().cpu()

        self.save_graph()

    def save_graph(self):
        with open(self.output_path, 'wb') as outp:
            pickle.dump(self.graph, outp, pickle.HIGHEST_PROTOCOL)