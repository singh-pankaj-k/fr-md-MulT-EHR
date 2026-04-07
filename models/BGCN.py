import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, HeteroConv
from layers import BBBGraphConv, BBBLinear
from .GNN import GNN

class BGCN(GNN):
    def __init__(self,
                 metadata,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 priors,
                 tasks,
                 causal=False,
                 graph_pooling_type="max"):
        super(BGCN, self).__init__(in_dim, hidden_dim, out_dim, n_layers, activation, dropout, tasks, causal)

        self.in_feats = in_dim
        self.convs = nn.ModuleList()
        # input layer
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = BBBGraphConv(in_dim, hidden_dim, activation=activation, priors=priors)
        self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # hidden layers
        for i in range(n_layers - 1):
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = BBBGraphConv(hidden_dim, hidden_dim, activation=activation, priors=priors)
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        if graph_pooling_type == 'sum':
            self.pool = global_add_pool
        elif graph_pooling_type == 'mean':
            self.pool = global_mean_pool
        elif graph_pooling_type == 'max':
            self.pool = global_max_pool
        else:
            raise NotImplementedError

    def forward(self, x_dict, edge_index_dict, out_key, task):
        # BGCN in original code was quite different, let's adapt it to our GNN interface
        logits_dict, kl = self.get_logit(x_dict, edge_index_dict)
        logits = logits_dict[out_key]
        self.embeddings = torch.cat(list(logits_dict.values()), dim=0)
        
        out = self.out[task](logits)
        # Note: BGCN might need to return kl as well, but our GNN.forward doesn't support it.
        # This might need a special trainer or handling.
        return out

    def get_logit(self, x_dict, edge_index_dict, causal=False):
        # We don't support causal for BGCN yet as it wasn't in original
        kl = 0
        current_x_dict = x_dict
        for i, conv in enumerate(self.convs):
            if i != 0:
                current_x_dict = {k: self.dropout(x) for k, x in current_x_dict.items()}
            
            # HeteroConv doesn't easily expose the underlying layers' kl_loss
            # We'd need to iterate over conv.convs.values()
            current_x_dict = conv(current_x_dict, edge_index_dict)
            
            for c in conv.convs.values():
                if hasattr(c, 'kl_loss'):
                    kl += c.kl_loss()
                    
        return current_x_dict, kl
