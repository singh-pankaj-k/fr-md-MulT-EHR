import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from .GNN import GNN


class HGT(GNN):
    def __init__(self, metadata, n_inp, n_hid, n_out, n_layers, n_heads, tasks, causal, dropout):
        super(HGT, self).__init__(n_inp, n_hid, n_out, n_layers, F.relu, dropout, tasks, causal)
        self.metadata = metadata
        self.n_hid = n_hid

        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(n_inp, n_hid)

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            # PyG HGTConv doesn't take 'group' in __init__
            conv = HGTConv(n_hid, n_hid, metadata, n_heads)
            self.convs.append(conv)

        if causal:
            self.rand_convs = nn.ModuleList()
            for _ in range(n_layers):
                conv = HGTConv(n_hid, n_hid, metadata, n_heads)
                self.rand_convs.append(conv)

    def forward(self, x_dict, edge_index_dict, out_key, task):
        # Filter x_dict and edge_index_dict to only include node/edge types that exist in the current graph
        # This is needed because some edge types might be empty and not present in edge_index_dict
        # but HGTConv might expect them if they are in metadata.
        
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
        # Filter x_dict to only include node types present in self.lin_dict
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu()
            for node_type, x in x_dict.items() if node_type in self.lin_dict
        }

        # Filter edge_index_dict to only include edge types present in x_dict
        # and skip empty edge types. Also, HGTConv expects all node types in metadata
        # to be present in x_dict even if they have no edges.
        
        # Ensure all node types from metadata are in x_dict
        metadata_node_types = self.metadata[0]
        # Find any existing device from x_dict
        device = next(iter(x_dict.values())).device
        
        for ntype in metadata_node_types:
            if ntype not in x_dict:
                # If a node type is missing but is in metadata, 
                # initialize it with zeros on the correct device.
                x_dict[ntype] = torch.zeros((0, self.n_hid), device=device)

        # HGTConv in forward() computes k_dict and v_dict for ALL node types in metadata
        # regardless of whether they have edges in edge_index_dict.
        # However, our get_logit only transforms node types that were in the input x_dict.
        # If a node type was in metadata but NOT in input x_dict, it will be missing from k_dict/v_dict.

        filtered_edge_index_dict = {}
        for (src, rel, dst), edge_index in edge_index_dict.items():
            if src in x_dict and dst in x_dict and edge_index.size(1) > 0:
                filtered_edge_index_dict[(src, rel, dst)] = edge_index

        convs = self.convs if not causal else self.rand_convs
        
        for conv in convs:
            # Re-ensure all metadata node types are in x_dict before each conv
            # because HGTConv might return a dict with only active node types
            for ntype in metadata_node_types:
                if ntype not in x_dict:
                    x_dict[ntype] = torch.zeros((0, self.n_hid), device=device)
            x_dict = conv(x_dict, filtered_edge_index_dict)
        
        return x_dict
