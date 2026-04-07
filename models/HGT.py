import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from .GNN import GNN


class HGT(GNN):
    def __init__(self, metadata, n_inp, n_hid, n_out, n_layers, n_heads, tasks, causal, dropout):
        super(HGT, self).__init__(n_inp, n_hid, n_out, n_layers, F.relu, dropout, tasks, causal)

        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(n_inp, n_hid)

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv = HGTConv(n_hid, n_hid, metadata, n_heads, group='sum')
            self.convs.append(conv)

        if causal:
            self.rand_convs = nn.ModuleList()
            for _ in range(n_layers):
                conv = HGTConv(n_hid, n_hid, metadata, n_heads, group='sum')
                self.rand_convs.append(conv)

    def forward(self, x_dict, edge_index_dict, out_key, task):
        logits = self.get_logit(x_dict, edge_index_dict)
        
        # Flatten and store embeddings for potential use
        self.embeddings = torch.cat(list(logits.values()), dim=0)
        
        out = self.out[task](logits[out_key])

        if self.causal:
            feat_rand = self.get_logit(x_dict, edge_index_dict, causal=True)
            feat_interv = {k: logits[k] + feat_rand[k] for k in feat_rand.keys()}
            out_interv = self.out[task](feat_interv[out_key])
            feat_rand_cat = torch.cat(list(feat_rand.values()), dim=0)
            return out, feat_rand_cat, out_interv

        return out

    def get_logit(self, x_dict, edge_index_dict, causal=False):
        # Initial linear transformation
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu()
            for node_type, x in x_dict.items()
        }

        convs = self.convs if not causal else self.rand_convs
        
        for conv in convs:
            x_dict = conv(x_dict, edge_index_dict)
        
        return x_dict
