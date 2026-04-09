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
        self.tasks = self.config_train.get("tasks", ["readm"])

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
        
        self.n_samples = self.config_train.get("n_samples", 2000)

    def get_indices_labels(self, t, train=True):
        indices = self.train_mask[t] if train else self.test_mask[t]
        # Convert to tensor to keep everything on the same device
        indices = torch.tensor(indices, device=self.device)

        if train:
            # Subsample to n_samples
            if self.n_samples > 0 and len(indices) > self.n_samples:
                if t == "mort_pred" and os.environ.get("MODE") == "dev":
                    # In dev mode, ensure we get some positives for mortality
                    all_labels = torch.tensor([self.labels[t][i.item()] for i in indices], device=self.device)
                    pos_mask = (all_labels == 1)
                    neg_mask = (all_labels == 0)
                    
                    pos_idx = indices[pos_mask]
                    neg_idx = indices[neg_mask]
                    
                    n_pos = min(len(pos_idx), self.n_samples // 2)
                    n_neg = self.n_samples - n_pos
                    
                    selected_pos = pos_idx[torch.randperm(len(pos_idx))[:n_pos]]
                    selected_neg = neg_idx[torch.randperm(len(neg_idx))[:n_neg]]
                    indices = torch.cat([selected_pos, selected_neg])
                else:
                    indices = indices[torch.randperm(len(indices))[:self.n_samples]]

        if t == "drug_rec":
            all_drugs = self.labels["all_drugs"]
            drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
            labels_list = []
            for i in indices:
                multi_hot = torch.zeros(len(all_drugs))
                for drug in self.labels[t][i.item()]:
                    if drug in drug_to_idx:
                        multi_hot[drug_to_idx[drug]] = 1
                labels_list.append(multi_hot)
            labels = torch.stack(labels_list).to(self.device)
        else:
            labels = torch.LongTensor([self.labels[t][i.item()] for i in indices]).to(self.device)

        if t == "mort_pred" and train:
            indices = self.down_sample(indices, labels)
            labels = torch.LongTensor([self.labels[t][i.item()] for i in indices]).to(self.device)

        return indices, labels

    def down_sample(self, indices, labels):
        """
        Down sample labels to ensure data balance (stays on GPU/device)
        """
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)
            
        neg_mask = (labels == 0)
        pos_mask = (labels == 1)
        
        neg_indices = indices[neg_mask]
        pos_indices = indices[pos_mask]
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return indices
            
        n = len(pos_indices)
        # Use torch for random selection to stay on device
        perm = torch.randperm(len(neg_indices), device=self.device)[:n]
        neg_indices_balanced = neg_indices[perm]

        return torch.cat([pos_indices, neg_indices_balanced])

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
            
            total_loss = 0
            all_train_metrics = {}
            for t in self.tasks:
                indices, labels = self.get_indices_labels(t, train=True)
                
                if len(indices) == 0:
                    continue
                    
                preds_dict = self.gnn(self.x_dict, self.edge_index_dict, "visit", t)
                preds = preds_dict[indices]
                
                if t == "drug_rec":
                    task_loss = F.binary_cross_entropy_with_logits(preds, labels)
                else:
                    task_loss = F.cross_entropy(preds, labels)
                
                total_loss += task_loss
                all_train_metrics.update(metrics(preds, labels, t, prefix=f"tr_{t}"))

            total_loss.backward()
            self.optimizer.step()

            # Perform validation and testing
            test_metrics = self.evaluate()

            prog_task = self.tasks[0]
            training_range.set_description_str("Epoch {} | loss: {:.4f}| Test {} AUC: {:.4f} ".format(
                epoch, total_loss.item(), prog_task, test_metrics.get(f"{prog_task}_roc_auc", 0.0)))

            epoch_stats.update({"Train Loss": total_loss.item()})
            epoch_stats.update(all_train_metrics)
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
        test_metrics = {}
        for t in self.tasks:
            indices, labels = self.get_indices_labels(t, train=False)
            
            if len(indices) == 0:
                continue

            with torch.no_grad():
                preds_dict = self.gnn(self.x_dict, self.edge_index_dict, "visit", t)
                preds = preds_dict[indices]
                test_metrics.update(metrics(preds, labels, t, prefix=f"{t}"))

        return test_metrics

    def get_masks(self, g, train: bool, task: str):
        if train:
            masks = self.train_mask[task]
        else:
            masks = self.test_mask[task]

        m = {}

        for tp in g.node_types:
            if tp == "visit":
                if isinstance(masks, torch.Tensor):
                    m[tp] = masks.int()
                else:
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
