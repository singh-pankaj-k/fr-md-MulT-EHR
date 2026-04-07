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
        # Filter x_dict and edge_index_dict to only include node/edge types that exist in the current graph
        metadata_node_types = self.metadata[0]
        metadata_edge_types = self.metadata[1]

        # Ensure all node types from metadata are in x_dict and have enough capacity BEFORE linear
        if len(x_dict) > 0:
            device = next(iter(x_dict.values())).device
        else:
            device = torch.device('cpu')

        for ntype in metadata_node_types:
            max_idx = -1
            for (src, rel, dst), edge_index in edge_index_dict.items():
                if src == ntype:
                    if edge_index.size(1) > 0:
                        max_idx = max(max_idx, int(edge_index[0].max().item()))
                if dst == ntype:
                    if edge_index.size(1) > 0:
                        max_idx = max(max_idx, int(edge_index[1].max().item()))
            
            num_nodes_needed = max_idx + 1
            
            if ntype not in x_dict or x_dict[ntype] is None:
                x_dict[ntype] = torch.zeros((max(0, num_nodes_needed), self.n_inp), device=device)
            elif x_dict[ntype].size(0) < num_nodes_needed:
                padding = torch.zeros((num_nodes_needed - x_dict[ntype].size(0), self.n_inp), device=device)
                x_dict[ntype] = torch.cat([x_dict[ntype], padding], dim=0)

        # Initial linear transformation
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu()
            for node_type, x in x_dict.items() if node_type in self.lin_dict
        }

        filtered_edge_index_dict = {}
        for (src, rel, dst), edge_index in edge_index_dict.items():
            if src in x_dict and dst in x_dict and edge_index.size(1) > 0:
                filtered_edge_index_dict[(src, rel, dst)] = edge_index

        convs = self.convs if not causal else self.rand_convs
        
        for conv in convs:
            # Re-ensure all metadata node types are in x_dict before each conv
            for ntype in metadata_node_types:
                max_idx = -1
                for (src, rel, dst), edge_index in filtered_edge_index_dict.items():
                    if src == ntype:
                        max_idx = max(max_idx, int(edge_index[0].max().item()))
                    if dst == ntype:
                        max_idx = max(max_idx, int(edge_index[1].max().item()))
                
                num_nodes_needed = max_idx + 1
                if ntype not in x_dict:
                    x_dict[ntype] = torch.zeros((max(0, num_nodes_needed), self.n_hid), device=device)
                elif x_dict[ntype].size(0) < num_nodes_needed:
                    padding = torch.zeros((num_nodes_needed - x_dict[ntype].size(0), self.n_hid), device=device)
                    x_dict[ntype] = torch.cat([x_dict[ntype], padding], dim=0)

            if len(filtered_edge_index_dict) > 0:
                x_dict = conv(x_dict, filtered_edge_index_dict)
        
        return x_dict
