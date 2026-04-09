import wandb
import random
from collections import OrderedDict

from tqdm import tqdm

import numpy as np
import torch
from torch.nn import functional as F

from .trainer import Trainer
from parse import (
    parse_optimizer,
    parse_gnn_model,
)

import plotly.graph_objects as go

from data import load_graph
from utils import metrics
from losses import KLDivergence

import pickle


class CausalGNNTrainer(Trainer):
    def __init__(self, config: OrderedDict):
        super().__init__(config)

        # Set loggers
        self.initialize_logger(config["name"])

        self.config_gnn = config["GNN"]

        # Initialize GNN model and optimizer
        self.tasks = self.config_train.get("tasks", ["readm"])

        # Load graph, labels and splits
        graph_path = self.config_data["graph_path"]
        labels_path = self.config_data["labels_path"]
        pretrained = self.config_data["pretrained"]
        self.graph, self.labels, self.train_mask, self.test_mask = load_graph(
            graph_path, labels_path, pretrained=pretrained
        )

        # Update GNN out_dim based on the number of drugs in the labels
        if "drug_rec" in self.tasks and "all_drugs" in self.labels:
            actual_n_drugs = len(self.labels["all_drugs"])
            if self.config_gnn["out_dim"] != actual_n_drugs:
                print(f"Updating GNN out_dim from {self.config_gnn['out_dim']} to {actual_n_drugs} to match drug vocabulary.")
                self.config_gnn["out_dim"] = actual_n_drugs

        # self.graph = dgl.AddReverse()(self.graph)
        self.x_dict = {}
        for tp in self.graph.node_types:
            # Ensure each node type has features, even if they're zero
            if not hasattr(self.graph[tp], 'x') or self.graph[tp].x is None:
                print(f"Initializing empty features for node type '{tp}'")
                self.graph[tp].x = torch.zeros((self.graph[tp].num_nodes, self.config_gnn["in_dim"]))
            
            self.x_dict[tp] = self.graph[tp].x.to(self.device)
            
        self.edge_index_dict = {k: v.to(self.device) for k, v in self.graph.edge_index_dict.items()}

        # Data augmentations
        # self.graph_aug = dgl.transforms.Compose(
        #     [
        #         # dgl.transforms.DropEdge(0.2),
        #         dgl.transforms.FeatMask(p=0.2, node_feat_names=['feat'])
        #     ]
        # )
        self.graph_aug = None

        # Read node_dict
        self.node_dict = {}
        for tp in self.graph.node_types:
            self.node_dict.update({tp: torch.arange(self.graph[tp].num_nodes)})

        self.gnn = parse_gnn_model(self.config_gnn, self.graph, self.tasks, causal=True).to(self.device)
        # read lists of edges
        self.optimizer = parse_optimizer(self.config_optim, self.gnn)

        self.causal = self.config_train["causal"]
        self.reg_coeff = self.config_train["reg"]
        self.n_samples = self.config_train["n_samples"]
        self.init_temperature = self.config_train["temperature"]

    def train(self) -> None:
        print(f"Start training GNN")

        self.load_checkpoint()

        training_range = tqdm(range(self.start_epoch, self.n_epoch), nrows=3)

        for epoch in training_range:
            self.gnn.train()
            self.anneal_temperature(epoch)
            epoch_stats = {"Epoch": epoch + 1}
            preds, labels = None, None
            losses = []

            # Perform aggregation on visits
            self.optimizer.zero_grad()
            # random.shuffle(self.tasks)
            all_train_metrics = {}
            for t in self.tasks:
                indices, labels = self.get_indices_labels(t)
                
                if len(indices) == 0:
                    continue

                preds, rand_feat, preds_interv = self.gnn(self.x_dict, self.edge_index_dict, "visit", t)
                
                # Check output size for drug_rec to handle label dimension mismatch
                if labels.dim() < 2 and t == "drug_rec":
                    # skip drug_rec if labels are empty
                    continue

                preds = preds[indices]
                preds_interv = preds_interv[indices]

                unif_loss = self.unif_loss(rand_feat) if self.causal else 0

                if t == "drug_rec":
                    preds = preds / self.temperature * 10  # Temperature scaling
                    loss = F.binary_cross_entropy_with_logits(preds, labels) + \
                           unif_loss * self.reg_coeff
                else:
                    preds /= self.temperature
                    loss = F.cross_entropy(preds, labels) + \
                           unif_loss * self.reg_coeff

                losses.append(loss.view(-1))
                
                # Collect train metrics for this task
                all_train_metrics.update(metrics(preds, labels, t, prefix=f"tr_{t}"))

            if not losses:
                print("Warning: No losses computed for this epoch (all tasks skipped).")
                continue
            var, mean = torch.var_mean(torch.cat(losses))
            loss = mean + torch.nan_to_num(var, 0).item()
            loss.backward()

            self.optimizer.step()

            # Perform validation and testing
            test_metrics = self.evaluate()

            # Use a more representative task for the progress bar (e.g., first available task)
            prog_task = self.tasks[0]
            auc_key = f"{prog_task}_roc_auc"
            if prog_task == 'los': auc_key = "los_roc_auc_weighted_ovo"
            if prog_task == 'drug_rec': auc_key = "drug_rec_roc_auc_samples"

            training_range.set_description_str(
                "Epoch {} | loss: {:.4f}| Test {} AUC: {:.4f} ".format(
                    epoch, loss.item(), prog_task,
                    test_metrics.get(auc_key, 0.0)
                )
            )

            epoch_stats.update({"Train Loss: ": loss.item()})
            epoch_stats.update(all_train_metrics)
            epoch_stats.update(test_metrics)
            # self.interpret()
            self.logging(loss, all_train_metrics, test_metrics)
            self.visualize_embeddings()

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
            indices, labels = self.get_indices_labels(t, False)

            all_preds = []
            # indices is a numpy array from self.test_mask[t]
            for chunk in torch.split(torch.from_numpy(indices), self.n_samples):

                # sg = self.get_subgraphs(chunk, "visit", False)

                with torch.no_grad():
                    preds, _, _ = self.gnn(self.x_dict, self.edge_index_dict, "visit", t)
                    preds = preds[chunk]
                    # preds *= self.temperature
                    if t == "drug_rec":
                        preds = preds.sigmoid()
                    all_preds.append(preds)

            all_preds = torch.cat(all_preds)
            
            # self.save_graph(sg, t)

            test_metrics.update(metrics(all_preds, labels, t, prefix=f"{t}"))

        return test_metrics

    def visualize_embeddings(self):
        # Skip visualization in dev-mode to avoid TSNE issues and speed up
        return
        
        layout = go.Layout(
            autosize=False,
            width=600,
            height=600)
        fig = go.Figure(layout=layout)
        embeddings = self.gnn.embeddings.detach().cpu().numpy()

        from sklearn.manifold import Isomap, TSNE

        offset = 0
        for k, v in self.node_dict.items():
            indices = [i for i in range(offset, offset + 250)]
            tsne = TSNE(n_components=2)
            embeddings_2d = tsne.fit_transform(embeddings[indices])
            offset += len(v)

            fig.add_trace(go.Scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], mode='markers', name=k))

        wandb.log({"chart": fig})

    def interpret(self):
        task = "readm"
        indices, labels = self.get_indices_labels(task, False, -1)

        sg = self.get_subgraphs(indices, "visit")

        preds, _ = self.gnn(sg, "visit", task)
        loss = F.cross_entropy(preds, labels)

        graph_explainer = GraphExplainer(sg, task, self.gnn, labels, loss, self.device)
        graph_explainer.explain()

    def unif_loss(self, feat):
        loss_fcn = KLDivergence()
        unif_feat = torch.rand_like(feat).to(self.device)
        feat = (feat - feat.min()) / (feat.max() - feat.min())
        loss = (loss_fcn(feat, unif_feat) + loss_fcn(unif_feat, feat)) / 2
        return loss

    def get_masks(self, g, train: bool, task: str):
        if train:
            masks = self.train_mask[task]
            labels = [self.labels[task][v] for v in masks]
        else:
            masks = self.test_mask[task]
            labels = [self.labels[task][v] for v in masks]

        m = {}

        for tp in g.ntypes:
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

    def get_subgraphs(self, indices, nt, train=True):
        d = self.node_dict.copy()
        d[nt] = self.node_dict[nt][indices]
        sg = self.graph.subgraph(d).to(self.device)
        if train and self.graph_aug:
            sg = self.graph_aug(sg)

        return sg

    def get_indices_labels(self, t, train=True):
        indices = self.train_mask[t] if train else self.test_mask[t]
        if train:  # set -1 tp use all indices
            indices = indices[torch.randperm(len(indices))[:self.n_samples]]

        if t == "drug_rec":
            all_drugs = self.train_mask["all_drugs"]
            labels = []
            for i in indices:
                drugs = self.labels[t][i]
                labels.append([1 if d in drugs else 0 for d in all_drugs])
            labels = torch.FloatTensor(labels).to(self.device)

        else:
            labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)

        if t == "mort_pred" and train:
            indices = self.down_sample(indices, labels)
            labels = torch.LongTensor([self.labels[t][i] for i in indices]).to(self.device)

        return indices, labels

    def down_sample(self, indices, labels):
        """
        Down sample labels to ensure data balance
        """
        neg_indices = indices[labels.detach().cpu() == 0]
        pos_indices = indices[labels.detach().cpu() == 1]
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return indices
            
        indices_balanced = np.random.choice(len(neg_indices), size=len(pos_indices), replace=True)
        neg_indices = neg_indices[indices_balanced]

        return np.concatenate(
            [pos_indices, neg_indices]
        )

    def save_graph(self, g, task):
        with open(f'{self.checkpoint_manager.path}/graph_{task}.pkl', 'wb') as outp:
            pickle.dump(g.cpu(), outp, pickle.HIGHEST_PROTOCOL)

    def logging(self, loss, train_metrics, test_metrics):
        wandb.log({"loss": loss})
        wandb.log(train_metrics)
        wandb.log(test_metrics)
