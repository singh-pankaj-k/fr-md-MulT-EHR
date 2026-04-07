"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.
Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv, Linear
from .GNN import GNN


class HAN(GNN):
    def __init__(self, metadata, in_size, hidden_size, out_size, num_heads, dropout, tasks, causal):
        # Note: PyG's HANConv expects metadata (node_types, edge_types)
        super(HAN, self).__init__(in_size, hidden_size, out_size, len(num_heads), F.relu, dropout, tasks, causal)

        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(in_size, hidden_size)

        self.layers = nn.ModuleList()
        # HANConv in PyG handles multiple layers and heads
        for i in range(len(num_heads)):
            in_ch = hidden_size # after initial linear
            self.layers.append(HANConv(in_ch, hidden_size, heads=num_heads[i], metadata=metadata, dropout=dropout))

        if causal:
            self.rand_layers = nn.ModuleList()
            for i in range(len(num_heads)):
                self.rand_layers.append(HANConv(hidden_size, hidden_size, heads=num_heads[i], metadata=metadata, dropout=dropout))

    def forward(self, x_dict, edge_index_dict, out_key, task):
        logits_dict = self.get_logit(x_dict, edge_index_dict)
        logits = logits_dict[out_key]
        self.embeddings = torch.cat(list(logits_dict.values()), dim=0)
        
        out = self.out[task](logits)

        if self.causal:
            feat_rand_dict = self.get_logit(x_dict, edge_index_dict, causal=True)
            feat_rand = feat_rand_dict[out_key]
            feat_interv = logits + feat_rand
            out_interv = self.out[task](feat_interv)
            feat_rand_cat = torch.cat(list(feat_rand_dict.values()), dim=0)
            return out, feat_rand_cat, out_interv

        return out

    def get_logit(self, x_dict, edge_index_dict, causal=False):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu()
            for node_type, x in x_dict.items()
        }

        layers = self.layers if not causal else self.rand_layers
        for layer in layers:
            x_dict = layer(x_dict, edge_index_dict)
        return x_dict