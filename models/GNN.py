"""
General GNN Module for PyTorch Geometric
"""

import torch
from torch import nn
from torch_geometric.data import HeteroData


class GNN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 tasks,
                 causal: bool = False):
        super().__init__()

        self.n_layers = n_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dor = dropout
        self.dropout = nn.Dropout(p=dropout)

        self.causal = causal
        self.tasks = tasks

        # We will initialize layers in the child class
        self.out = nn.ModuleDict()
        for t in tasks:
            if t in ["readm", "mort_pred"]:
                self.out[t] = nn.Linear(hidden_dim, 2)
            elif t == "los":
                self.out[t] = nn.Linear(hidden_dim, 10)
            elif t == "drug_rec":
                self.out[t] = nn.Linear(hidden_dim, out_dim)
        
        if causal:
            self.out_rand = nn.ModuleDict()
            for t in tasks:
                self.out_rand[t] = nn.Linear(hidden_dim, out_dim)

        self.embeddings = None

    def forward(self, x_dict, edge_index_dict, out_key, task):
        raise NotImplementedError

    def get_logit(self, x_dict, edge_index_dict, causal=False):
        raise NotImplementedError

    def set_embeddings(self, emb):
        self.embeddings = emb

    