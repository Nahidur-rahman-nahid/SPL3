"""
Train FraudGCN and FraudGNN (GraphSAGE) under both a transductive and an
inductive setting, and compare all four results side by side. This is the
"prove the difference" script for the demo.

  transductive : the WHOLE graph (all nodes + all edges) is visible during
                  training; only the fraud labels of the test accounts are
                  hidden from the loss. Standard GCN assumption.

  inductive    : test accounts (and every edge touching them) are removed
                  from the graph entirely during training. The trained model
                  is only shown the test accounts — and their real
                  connections — for the first time at evaluation. This is
                  what "a new account showed up today" looks like in
                  production, and it's the setting GraphSAGE was designed for.

Data is split into train/val/test (not just train/test). Val exists purely
to pick a decision threshold: with fraud at ~0.3% of the data, the default
0.5 cutoff on softmax output is badly calibrated (high recall, near-zero
precision). We sweep thresholds on val to maximize F1, then apply that fixed
threshold to test. ROC-AUC and PR-AUC are also reported — they're
threshold-independent and tell you how well the model *ranks* fraud, which
is the fairer number for comparing models before any cutoff is chosen.

Usage (Colab):
  Either save 05_gnn_model.py / 05_gnn_model_gcn.py alongside this file and
  run `python 06_train_compare.py`, or paste those two files into their own
  cells first (defining FraudGNN / FraudGCN), then paste/run this file — the
  import shim below falls back to whatever is already in globals().

Requires paysim_graph.pt from 03_graph_builder.py in the working directory.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.utils import subgraph

GRAPH_PATH = "paysim_graph.pt"
# Save into Drive, not local Colab storage — local /content is wiped when the
# runtime disconnects/restarts, so anything only saved there is lost.
MODEL_DIR = "/content/drive/MyDrive/SPL3/models"
TEST_SIZE = 0.2
VAL_SIZE = 0.1
SEED = 42
HIDDEN_CHANNELS = 64
EPOCHS = 100
LR = 0.01


def _load_class(module_name, file_path, class_name):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    except FileNotFoundError:
        # Fall back to a class already defined earlier in the notebook.
        if class_name in globals():
            return globals()[class_name]
        raise


FraudGNN = _load_class("gnn_model", "05_gnn_model.py", "FraudGNN")
FraudGCN = _load_class("gnn_model_gcn", "05_gnn_model_gcn.py", "FraudGCN")


def make_splits(data, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=SEED):
    y = data.y.numpy()
    idx = np.arange(data.num_nodes)

    train_val_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    val_fraction = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_fraction, stratify=y[train_val_idx], random_state=seed
    )

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def class_weights(y, train_idx):
    counts = torch.bincount(y[train_idx], minlength=2).float()
    return counts.sum() / (2.0 * counts)


def tune_threshold(y_val: torch.Tensor, probs_val: torch.Tensor) -> float:
    """Pick the probability cutoff that maximizes F1 on the validation set,
    instead of blindly using 0.5 (which is badly calibrated for ~0.3% fraud
    prevalence)."""
    y_val, probs_val = y_val.numpy(), probs_val.numpy()
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
    if len(thresholds) == 0:
        return 0.5
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    return float(thresholds[np.argmax(f1s[:-1])])


def save_checkpoint(model, in_channels, threshold, setting):
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"{model.__class__.__name__}_{setting}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "in_channels": in_channels,
        "hidden_channels": HIDDEN_CHANNELS,
        "num_classes": 2,
        "threshold": threshold,
    }, save_path)
    print(f"  Saved model to {save_path}")


def evaluate(y_true: torch.Tensor, probs: torch.Tensor, threshold: float) -> dict:
    y_true, probs = y_true.numpy(), probs.numpy()
    preds = (probs >= threshold).astype(int)
    try:
        roc_auc = roc_auc_score(y_true, probs)
    except ValueError:
        roc_auc = float("nan")  # only one class present among the test nodes
    try:
        pr_auc = average_precision_score(y_true, probs)
    except ValueError:
        pr_auc = float("nan")
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def train_transductive(model_cls, data, train_idx, val_idx, test_idx, device):
    model = model_cls(in_channels=data.num_node_features, hidden_channels=HIDDEN_CHANNELS, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(data.y, train_idx).to(device))

    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    train_idx_dev = train_idx.to(device)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_idx_dev], y[train_idx_dev])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == EPOCHS:
            print(f"  epoch {epoch:3d}  loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        probs = F.softmax(out, dim=1)[:, 1]

    threshold = tune_threshold(y[val_idx.to(device)].cpu(), probs[val_idx.to(device)].cpu())
    save_checkpoint(model, data.num_node_features, threshold, "transductive")
    return evaluate(y[test_idx.to(device)].cpu(), probs[test_idx.to(device)].cpu(), threshold)


def train_inductive(model_cls, data, train_idx, val_idx, test_idx, device):
    model = model_cls(in_channels=data.num_node_features, hidden_channels=HIDDEN_CHANNELS, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(data.y, train_idx).to(device))

    # Train graph: ONLY train accounts and edges between them exist here.
    # Val and test accounts, and everything touching them, are invisible
    # during training — both are genuinely unseen at eval time.
    train_edge_index, _ = subgraph(train_idx, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    x_train = data.x[train_idx].to(device)
    y_train = data.y[train_idx].to(device)
    train_edge_index = train_edge_index.to(device)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        out = model(x_train, train_edge_index)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == EPOCHS:
            print(f"  epoch {epoch:3d}  loss {loss.item():.4f}")

    # Evaluate on the FULL graph — val/test accounts and their real
    # connections appear here for the first time. This is the inductive check.
    model.eval()
    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    with torch.no_grad():
        out = model(x, edge_index)
        probs = F.softmax(out, dim=1)[:, 1]

    threshold = tune_threshold(y[val_idx.to(device)].cpu(), probs[val_idx.to(device)].cpu())
    save_checkpoint(model, data.num_node_features, threshold, "inductive")
    return evaluate(y[test_idx.to(device)].cpu(), probs[test_idx.to(device)].cpu(), threshold)


def print_table(results):
    rows = []
    for (model_name, setting), metrics in results.items():
        rows.append({"model": model_name, "setting": setting, **metrics})
    df = pd.DataFrame(rows)
    print("\nComparison table:")
    print(df.to_string(index=False))
    df.to_csv("comparison_results.csv", index=False)
    print("Saved comparison_results.csv")
    return df


def plot_comparison(results):
    import matplotlib.pyplot as plt

    labels = [f"{m}\n{s}" for (m, s) in results.keys()]
    f1_scores = [results[k]["f1"] for k in results]
    pr_auc_scores = [results[k]["pr_auc"] for k in results]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, f1_scores, width, label="F1 (tuned threshold)")
    plt.bar(x + width / 2, pr_auc_scores, width, label="PR-AUC")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Transductive vs Inductive: GCN vs GraphSAGE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("transductive_vs_inductive.png", dpi=150)
    print("Saved transductive_vs_inductive.png")
    plt.show()


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.load(GRAPH_PATH, weights_only=False)
    train_idx, val_idx, test_idx = make_splits(data)
    print(f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    results = {}
    for model_name, model_cls in [("GCN", FraudGCN), ("SAGE", FraudGNN)]:
        for setting, train_fn in [("transductive", train_transductive), ("inductive", train_inductive)]:
            print(f"\n=== {model_name} — {setting} ===")
            results[(model_name, setting)] = train_fn(model_cls, data, train_idx, val_idx, test_idx, device)
            print(f"  {results[(model_name, setting)]}")

    print_table(results)
    plot_comparison(results)


if __name__ == "__main__":
    main()
