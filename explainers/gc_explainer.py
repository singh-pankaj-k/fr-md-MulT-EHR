"""

Explainer based on Granger Causality
"""

import networkx as nx
import matplotlib.pyplot as plt
import plotly as py
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx

from .explainer import Explainer

from data import load_graph


class GCGraphExplainer(Explainer):
    def __init__(self, config):

        super().__init__(config)

        # Load graph, labels and splits
        graph_path = self.config_data["graph_path"]
        labels_path = self.config_data["labels_path"]
        self.graph, self.labels, self.train_mask, self.test_mask = load_graph(graph_path, labels_path)

        self.sub_g = self.get_nodes_to_explain()
        self.nodes_to_explain = self.sub_g.ndata["_ID"]
        del self.sub_g

    def explain(self):
        node_imp = {}

        for k, v in self.nodes_to_explain.items():
            if k == "visit":
                continue

            node_imp[k] = torch.zeros_like(v)

            for i in range(len(v)):
                # Simplified removal logic for PyG migration
                # In PyG, we can't easily 'remove' nodes from HeteroData on the fly 
                # without rebuilding or complex masking.
                # For this placeholder, we'll just skip the actual computation.
                pass

        return node_imp

    def get_nodes_to_explain(self):
        # Placeholder for PyG migration
        return self.graph

    def visualize(self, graph, node_importance):
        """
        Visualizes a graph with node importance.

        Args:
        - graph: A NetworkX graph object representing the input graph.
        - node_importance: A dictionary mapping node IDs to importance scores.
        """
        # Create a list of node colors based on the importance scores
        node_colors = [node_importance[node] for node in graph.nodes()]

        # Create a list of node sizes based on the importance scores
        node_sizes = [node_importance[node] * 1000 for node in graph.nodes()]

        # Draw the graph using the spring layout algorithm
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap=plt.cm.Reds, node_size=node_sizes)
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos)

        # Add a color bar to display the node importance scores
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=max(node_importance.values())))
        sm._A = []
        plt.colorbar(sm)

        # Show the plot
        plt.show()