import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HeteroConv, Linear
from .GNN import GNN

class GCN(GNN):
    def __init__(self,
                 metadata,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 tasks,
                 causal=False):
        super().__init__(in_dim, hidden_dim, out_dim, n_layers, activation, dropout, tasks, causal)

        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = GCNConv(hidden_dim, hidden_dim)
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        if causal:
            self.rand_convs = nn.ModuleList()
            for _ in range(n_layers):
                conv_dict = {}
                for edge_type in metadata[1]:
                    conv_dict[edge_type] = GCNConv(hidden_dim, hidden_dim)
                self.rand_convs.append(HeteroConv(conv_dict, aggr='sum'))

    def forward(self, x_dict, edge_index_dict, out_key, task):
        logits = self.get_logit(x_dict, edge_index_dict)
        self.embeddings = torch.cat(list(logits.values()), dim=0)
        out = self.out[task](logits[out_key])

        if self.causal:
            feat_rand_dict = self.get_logit(x_dict, edge_index_dict, causal=True)
            feat_rand = feat_rand_dict[out_key]
            feat_interv = {k: logits[k] + feat_rand_dict[k] for k in feat_rand_dict.keys()}
            out_interv = self.out[task](feat_interv[out_key])
            feat_rand_cat = torch.cat(list(feat_rand_dict.values()), dim=0)
            return out, feat_rand_cat, out_interv

        return out

    def get_logit(self, x_dict, edge_index_dict, causal=False):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu()
            for node_type, x in x_dict.items()
        }

        convs = self.convs if not causal else self.rand_convs
        for i, conv in enumerate(convs):
            if i != 0:
                x_dict = {k: self.dropout(x) for k, x in x_dict.items()}
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.activation(x) for k, x in x_dict.items()}
        return x_dict
