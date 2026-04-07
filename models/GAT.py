import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, HeteroConv, Linear
from .GNN import GNN


class GAT(GNN):
    def __init__(self,
                 metadata,
                 n_layers,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 tasks,
                 causal):

        self.heads = heads
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual

        super().__init__(in_dim, hidden_dim, out_dim, n_layers, activation, feat_drop, tasks, causal)

        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for l in range(n_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                # In PyG GATConv, if it's the first layer, in_channels = hidden_dim
                # due to the linear projection we do in get_logit
                in_ch = hidden_dim if l == 0 else hidden_dim * heads[l-1]
                conv_dict[edge_type] = GATConv(in_ch, hidden_dim, heads[l], 
                                               dropout=attn_drop, 
                                               negative_slope=negative_slope, 
                                               concat=True)
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        if causal:
            self.rand_convs = nn.ModuleList()
            for l in range(n_layers):
                conv_dict = {}
                for edge_type in metadata[1]:
                    in_ch = hidden_dim if l == 0 else hidden_dim * heads[l-1]
                    conv_dict[edge_type] = GATConv(in_ch, hidden_dim, heads[l], 
                                                   dropout=attn_drop, 
                                                   negative_slope=negative_slope, 
                                                   concat=True)
                self.rand_convs.append(HeteroConv(conv_dict, aggr='sum'))

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

        convs = self.convs if not causal else self.rand_convs
        
        for i, conv in enumerate(convs):
            if i != 0:
                x_dict = {k: self.dropout(x) for k, x in x_dict.items()}
            x_dict = conv(x_dict, edge_index_dict)
            
            # GATConv output is (N, heads * hidden_dim) if concat=True
            # Original logic used flatten(1) which is consistent with concat=True
            # Except for the last layer where it used mean(1)
            if i == len(convs) - 1:
                # mean across heads
                x_dict = {k: x.view(-1, self.heads[i], self.hidden_dim).mean(dim=1) for k, x in x_dict.items()}
            
            x_dict = {k: self.activation(x) for k, x in x_dict.items()}

        return x_dict