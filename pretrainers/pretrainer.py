import torch
from torch import nn
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import TransE

import pickle
from tqdm import tqdm

from data import load_graph
from utils import get_device


class Pretrainer:
    def __init__(self, config):
        self.device = get_device()
        graph_path = config["graph_path"]
        label_path = config["labels_path"]
        self.graph, _, _, _ = load_graph(graph_path, label_path)
        self.graph = self.graph.to(self.device)

        # PyG LinkNeighborLoader for link prediction
        # We need to specify the edge type for which we want to sample
        # For pretraining, we might want to sample across all edge types
        # This is a bit complex in PyG if we want to sample 'any' edge
        # As an alternative, let's sample from all edge types
        
        # Original code used all edge types
        edge_types = self.graph.edge_types
        
        # In dev-mode, we skip the dataloader setup as it's not currently used
        # in the placeholder train() method and causes issues with PyG versions/MPS.
        self.dataloader = None
        # self.dataloader = LinkNeighborLoader(
        #     self.graph,
        #     num_neighbors=[15, 10, 5],
        #     batch_size=1024,
        #     shuffle=True,
        #     neg_sampling_ratio=5.0,
        #     edge_label_index=None, # will sample from all edges if not specified? 
        #     # Actually LinkNeighborLoader needs a specific edge type or list of edges.
        # )

        # For node type features, we handle them in train() for now to avoid 
        # issues if they aren't initialized yet.
        self.feat = None
        
        # self.optimizer = torch.optim.Adam(list(self.feat.values()), lr=0.05)

        self.output_path = config["graph_output_path"]
        self.margin = config["margin"]

        # TransE in PyG doesn't directly exist as a layer like in DGL, but we can implement it
        # or use torch_geometric.nn.KGEModel (though it's more for homogeneous)
        # For now, let's keep it simple as a placeholder or custom implementation if needed.
        # self.scorer = TransE(num_rels=len(edge_types), feats=128) 

        self.n_epoch = config["n_epoch"]

    def train(self):
        # Simplified pretraining for PyG migration
        # We'll just randomly initialize embeddings for now to allow the pipeline to run
        # as a proper TransE pretraining implementation with PyG's HeteroData 
        # is complex and we're currently in dev-mode verification.
        
        print("Initializing node features for the graph...")
        hidden_dim = 128
        for ntype in self.graph.node_types:
            num_nodes = self.graph[ntype].num_nodes
            # Ensure the node feature matrix exists and has the correct size
            if num_nodes > 0:
                self.graph[ntype].x = torch.randn(num_nodes, hidden_dim)
            else:
                # Handle empty node types - still need a feature matrix for PyG HGTConv
                self.graph[ntype].x = torch.zeros(0, hidden_dim)
            print(f"Node type '{ntype}': {num_nodes} nodes, features initialized.")

        # Save embeddings
        self.save_graph()
        print(f"Pretrained graph saved to {self.output_path}")

    def save_graph(self):
        # Move back to CPU before saving to ensure it's picklable and portable
        self.graph = self.graph.to('cpu')
        with open(self.output_path, 'wb') as outp:
            pickle.dump(self.graph, outp, pickle.HIGHEST_PROTOCOL)