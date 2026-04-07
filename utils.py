import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from collections import OrderedDict

import numpy as np

from pyhealth.metrics import binary_metrics_fn, multilabel_metrics_fn, multiclass_metrics_fn

import warnings
warnings.filterwarnings('ignore')


import torch

def get_device():
    """
    Returns the best available device: cuda, mps, or cpu.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def ordered_yaml():
    """
    yaml orderedDict support
    """
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def load_config(name, config_dir="./configs/"):
    config_path = f"{config_dir}{name}"

    with open(config_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {config_path}")
    return config


def metrics(outputs, targets, t, prefix="tr"):

    if t in ["mort_pred", "readm"]:
        met = binary_metrics_fn(
            targets.detach().cpu().numpy(),
            outputs.softmax(1)[:, 1].detach().cpu().numpy(),
            metrics=["accuracy", "roc_auc", "f1", "pr_auc"]
        )
    elif t == "los":
        try:
            met = multiclass_metrics_fn(
                targets.detach().cpu().numpy(),
                outputs.softmax(1).detach().cpu().numpy(),
                metrics=["roc_auc_weighted_ovo", "f1_weighted", "accuracy"]
            )
        except ValueError:
            # Fallback for dev mode where not all classes might be present in labels
            print(f"Warning: Multiclass metrics failed for {t} (likely missing classes in dev set). Using accuracy only.")
            from sklearn.metrics import accuracy_score
            y_true = targets.detach().cpu().numpy()
            y_pred = outputs.argmax(1).detach().cpu().numpy()
            met = {"accuracy": accuracy_score(y_true, y_pred), "roc_auc_weighted_ovo": 0.0, "f1_weighted": 0.0}
    elif t == "drug_rec":
        met = multilabel_metrics_fn(
            targets.detach().cpu().numpy(),
            outputs.detach().cpu().numpy(),
            metrics=["roc_auc_samples", "pr_auc_samples", "accuracy", "f1_weighted", "jaccard_weighted"]
        )
    else:
        raise ValueError

    met = {f"{prefix}_{k}": v for k, v in met.items()}
    return met
    #
    # return {
    #     # f"{prefix}_prec": precision,
    #     # f"{prefix}_recall": recall,
    #     f"{prefix}_accuracy": accuracy,
    #     f"{prefix}_auroc": aucroc,
    #     f"{prefix}_f1": f1
    # }
