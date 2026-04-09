import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, HeteroConv, Linear
from .GNN import GNN


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(GNN):
    """GIN model"""
    def __init__(self, metadata, in_dim, hidden_dim,
                 out_dim, num_layers, num_mlp_layers,
                 final_dropout, tasks,
                 causal, neighbor_pooling_type="mean", learn_eps=True):
        
        self.learn_eps = learn_eps
        self.num_mlp_layers = num_mlp_layers
        self.neighbor_pooling_type = neighbor_pooling_type

        super().__init__(in_dim, hidden_dim, out_dim, num_layers, F.relu, final_dropout, tasks, causal)

        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
                conv_dict[edge_type] = GINConv(mlp, train_eps=learn_eps)
            self.convs.append(HeteroConv(conv_dict, aggr=neighbor_pooling_type))

        if causal:
            self.rand_convs = nn.ModuleList()
            for layer in range(num_layers):
                conv_dict = {}
                for edge_type in metadata[1]:
                    mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
                    conv_dict[edge_type] = GINConv(mlp, train_eps=learn_eps)
                self.rand_convs.append(HeteroConv(conv_dict, aggr=neighbor_pooling_type))

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
            
            # Use merging to preserve node features if they are not updated by conv
            new_x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: new_x_dict[k] if k in new_x_dict else x_dict[k] for k in x_dict}
            
            x_dict = {k: self.activation(x) for k, x in x_dict.items()}
        return x_dict