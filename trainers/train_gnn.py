import os
from collections import OrderedDict

from tqdm import tqdm

import torch
from torch.nn import functional as F

import numpy as np

from .trainer import Trainer
from parse import (
    parse_optimizer,
    parse_gnn_model,
    parse_loss
)

from data import load_graph
from utils import metrics


class GNNTrainer(Trainer):
    def __init__(self, config: OrderedDict):
        super().__init__(config)

        self.config_gnn = config["GNN"]

        # Initialize GNN model and optimizer
        self.tasks = ["readm"]

        # Load graph, labels and splits
        graph_path = self.config_data["graph_path"]
        dataset_path = self.config_data["dataset_path"]
        labels_path = self.config_data["labels_path"]
        entity_mapping = self.config_data["entity_mapping"]
        self.graph, self.labels, self.train_mask, self.test_mask = load_graph(graph_path, labels_path)

        # Update GNN out_dim based on the number of drugs in the labels
        if "drug_rec" in self.tasks and "all_drugs" in self.labels:
            actual_n_drugs = len(self.labels["all_drugs"])
            if self.config_gnn["out_dim"] != actual_n_drugs:
                print(f"Updating GNN out_dim from {self.config_gnn['out_dim']} to {actual_n_drugs} to match drug vocabulary.")
                self.config_gnn["out_dim"] = actual_n_drugs

        # PyG: No need for dgl.AddReverse(), it's usually handled by adding reverse edges to HeteroData or during construction
        # If reverse edges are needed and not in the graph, we can use T.ToUndirected()

        # Read node_dict (In PyG, x_dict and edge_index_dict are passed directly)
        self.x_dict = {tp: self.graph[tp].x for tp in self.graph.node_types}
        self.edge_index_dict = self.graph.edge_index_dict

        self.gnn = parse_gnn_model(self.config_gnn, self.graph, self.tasks).to(self.device)
        self.optimizer = parse_optimizer(self.config_optim, self.gnn)

    def train(self) -> None:
        print(f"Start training GNN")

        self.load_checkpoint()

        training_range = tqdm(range(self.start_epoch, self.n_epoch), nrows=3)

        for epoch in training_range:
            self.gnn.train()
            epoch_stats = {"Epoch": epoch + 1}
            preds, labels = None, None

            # Perform aggregation on visits
            self.optimizer.zero_grad()
            
            # Simple subgraphing for PyG (full graph training for now as in original)
            # In original code, it creates a subgraph for each task's training indices
            for t in self.tasks:
                indices = self.train_mask[t]
                # To simulate subgraphing on 'visit' nodes in PyG:
                # We can either pass a mask or truly subgraph
                # Here we pass the full x_dict/edge_index_dict and then slice the output
                preds_dict = self.gnn(self.x_dict, self.edge_index_dict, "visit", t)
                preds = preds_dict # The HGT model I wrote returns the output for the specified task and key already
                
                labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)
                # Slice preds to only include the training indices for 'visit' nodes
                # Note: This assumes the model output corresponds to all 'visit' nodes
                # The way I wrote HGT.forward, it returns self.out[task](logits[out_key])
                # where logits[out_key] is for all nodes of that type.
                
                preds = preds[indices]
                loss = F.cross_entropy(preds, labels)

            loss.backward()
            self.optimizer.step()

            train_metrics = metrics(preds, labels, "readm", prefix="train")

            # Perform validation and testing
            test_metrics = self.evaluate()

            training_range.set_description_str("Epoch {} | loss: {:.4f}| Train AUC: {:.4f} | Test AUC: {:.4f} | Test ACC: {:.4f} ".format(
                epoch, loss.item(), train_metrics["train_roc_auc"], test_metrics["test_roc_auc"], test_metrics["test_accuracy"]))

            epoch_stats.update({"Train Loss: ": loss.item()})
            epoch_stats.update(train_metrics)
            epoch_stats.update(test_metrics)

            if self.should_save(epoch):
                # State dict of the model including embeddings
                checkpoint = {
                    "model_state_dict": self.gnn.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch + 1
                }
                self.checkpoint_manager.write_new_version(
                    self.config,
                    checkpoint,
                    epoch_stats
                )

                # Remove previous checkpoint
                self.checkpoint_manager.remove_old_version()

    def evaluate(self):
        self.gnn.eval()
        for t in self.tasks:
            indices = self.test_mask[t]
            labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)

            with torch.no_grad():
                preds_dict = self.gnn(self.x_dict, self.edge_index_dict, "visit", t)
                preds = preds_dict[indices]

        test_metrics = metrics(preds, labels, "readm", prefix="test")

        return test_metrics

    def get_masks(self, g, train: bool, task: str):
        if train:
            masks = self.train_mask[task]
        else:
            masks = self.test_mask[task]

        m = {}

        for tp in g.node_types:
            if tp == "visit":
                m[tp] = torch.from_numpy(masks.astype("int32"))
            else:
                m[tp] = torch.zeros(0)

        return m

    def get_labels(self, train: bool, task: str):
        if train:
            masks = self.train_mask[task]
            labels = [self.labels[task][v] for v in masks]
        else:
            masks = self.test_mask[task]
            labels = [self.labels[task][v] for v in masks]

        return masks, labels

    def up_sample(self, scores, label):
        """
        Up sample labels to ensure data balance
        :param scores:
        :param label:
        :return:
        """

    # def train_one_step(self, label):
    #     self.optimizer.zero_grad()
    #
    #     for t in self.tasks:
    #         pred = self.gnn(self.graph, "visit", t)
    #         prob = F.softmax(pred)
    #
    #     loss = F.cross_entropy(pred, label)
    #
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     accuracy = acc(pred, label)
    #
    #     pred = pred.detach().cpu().numpy().argmax(axis=1)
    #     prob = prob.detach().cpu().numpy()
    #     label = label.detach().cpu().numpy()
    #
    #     return loss.item(), accuracy, pred, prob, label
