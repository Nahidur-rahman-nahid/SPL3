"""
Train the fused BatchNet + RealTimeNet + LambdaNeuralNetwork FraudGNN
(main_gnn_model.py) and evaluate it against the fixed 4-tier decision policy
(architecture.md section 4):

    prob < 0.40         -> ALLOW,  LOW
    0.40 <= prob < 0.75  -> REVIEW, MEDIUM
    0.75 <= prob < 0.90  -> BLOCK,  HIGH
    prob >= 0.90         -> BLOCK,  CRITICAL

Unlike 06_train_compare.py (which picks whatever threshold maximizes F1),
these tiers are FIXED, business-defined risk bands. The reported result is
"how many real transactions fall into each band, and how many of those are
actually fraud" — a concrete, defensible number for a non-technical audience,
not a single accuracy figure.

Two evaluation passes are run on the test set:
  - use_batch=True  : full fused model (BatchNet + RealTimeNet). This is the
                       ceiling — what the model can do with the whole graph.
  - use_batch=False : RealTimeNet only, BatchNet zeroed out. This is what the
                       SAME trained weights do in production, with no full
                       graph available. Comparing the two demonstrates the
                       graceful-degradation design directly.

Requires paysim_graph.pt (7-feature FULL_FEATURE_COLS version — see
03_graph_builder.py) in the working directory.

Usage (Colab): paste main_gnn_model.py into its own cell and run it first
(defines BatchNet/RealTimeNet/LambdaNeuralNetwork/FraudGNN), then paste and
run this file — the import shim below falls back to whatever is already in
globals() if main_gnn_model.py wasn't saved as an actual file on disk.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

GRAPH_PATH = "paysim_graph.pt"
MODEL_DIR = "/content/drive/MyDrive/SPL3/models"

TEST_SIZE = 0.15
VAL_SIZE = 0.15
SEED = 42

HIDDEN_CHANNELS = 32  # architecture.md specifies 64, but full-batch dual-encoder
                       # (GCNConv + SAGEConv) training over 2.77M nodes at 64
                       # exceeds a T4's 16GB during backward. 32 fits; revisit
                       # if you move to mini-batch (NeighborLoader) training,
                       # which would let you go back to 64.
DROPOUT = 0.3
LR = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 10
MAX_EPOCHS = 200  # safety cap — full-batch dual-encoder epochs are expensive; early stopping should trigger well before this
VAL_EVERY_N_EPOCHS = 3  # the validation forward pass is a second full-graph
                         # pass through both encoders — running it every epoch
                         # adds unnecessary peak memory pressure on top of an
                         # already tight budget

# Fixed 4-tier decision policy. BLOCK_THRESHOLD (0.75) is what
# application.yml will expose as configurable in production.
LOW_MAX = 0.40
REVIEW_MAX = 0.75  # == BLOCK_THRESHOLD
HIGH_MAX = 0.90


def _load_class(module_name, file_path, class_name):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    except FileNotFoundError:
        if class_name in globals():
            return globals()[class_name]
        raise


FraudGNN = _load_class("main_gnn_model", "main_gnn_model.py", "FraudGNN")


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


def pos_weight_value(y: torch.Tensor, train_idx: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(y[train_idx].long(), minlength=2).float()
    n_legit, n_fraud = counts[0], counts[1]
    return n_legit / n_fraud


def tier_decision(probs: np.ndarray):
    decisions = np.select(
        [probs < LOW_MAX, probs < REVIEW_MAX, probs < HIGH_MAX],
        ["ALLOW", "REVIEW", "BLOCK"],
        default="BLOCK",
    )
    risk = np.select(
        [probs < LOW_MAX, probs < REVIEW_MAX, probs < HIGH_MAX],
        ["LOW", "MEDIUM", "HIGH"],
        default="CRITICAL",
    )
    return decisions, risk


def tier_breakdown(y_true: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    decisions, risk = tier_decision(probs)
    df = pd.DataFrame({"decision": decisions, "risk": risk, "is_fraud": y_true})
    grouped = df.groupby(["decision", "risk"], as_index=False).agg(
        count=("is_fraud", "size"), fraud_count=("is_fraud", "sum")
    )
    grouped["fraud_rate_in_tier"] = grouped["fraud_count"] / grouped["count"]
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    grouped["_order"] = grouped["risk"].map(order)
    return grouped.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def evaluate(y_true: np.ndarray, probs: np.ndarray) -> dict:
    # BLOCK boundary (0.75) is the point the system actually takes action,
    # so precision/recall/F1/FPR are computed against that cutoff.
    preds = (probs >= REVIEW_MAX).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    try:
        roc_auc = roc_auc_score(y_true, probs)
    except ValueError:
        roc_auc = float("nan")
    return {
        "roc_auc": roc_auc,
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "false_positive_rate": fpr,
    }


def plot_roc(y_true: np.ndarray, probs_full: np.ndarray, probs_realtime: np.ndarray):
    import matplotlib.pyplot as plt

    fpr_f, tpr_f, _ = roc_curve(y_true, probs_full)
    fpr_r, tpr_r, _ = roc_curve(y_true, probs_realtime)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_f, tpr_f, label=f"Full (BatchNet+RealTimeNet) AUC={roc_auc_score(y_true, probs_full):.3f}")
    plt.plot(fpr_r, tpr_r, label=f"RealTimeNet only AUC={roc_auc_score(y_true, probs_realtime):.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("FraudGNN ROC — full model vs real-time-only")
    plt.legend()
    plt.tight_layout()
    plt.savefig("main_gnn_roc.png", dpi=150)
    print("Saved main_gnn_roc.png")
    plt.show()


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.load(GRAPH_PATH, weights_only=False)
    if data.num_node_features != 7:
        print(f"WARNING: expected 7 input features (architecture.md section 3), "
              f"but paysim_graph.pt has {data.num_node_features}. "
              f"Rebuild it with FEATURE_COLS = FULL_FEATURE_COLS in 03_graph_builder.py.")

    train_idx, val_idx, test_idx = make_splits(data)
    print(f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    model = FraudGNN(in_channels=data.num_node_features, hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    x, edge_index = data.x.to(device), data.edge_index.to(device)
    y = data.y.to(device).float()
    train_idx_dev = train_idx.to(device)
    val_idx_dev = val_idx.to(device)
    test_idx_dev = test_idx.to(device)

    pw = pos_weight_value(data.y, train_idx).to(device)
    print(f"pos_weight (n_legit / n_fraud, train split): {pw.item():.2f}")

    best_val_auc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        probs = model(x, edge_index, use_batch=True)

        y_train = y[train_idx_dev]
        probs_train = probs[train_idx_dev]
        sample_weight = torch.where(y_train == 1, pw, torch.ones_like(y_train))
        loss = F.binary_cross_entropy(probs_train, y_train, weight=sample_weight)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        del probs, probs_train, y_train, sample_weight, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == 1 or epoch == MAX_EPOCHS:
            model.eval()
            with torch.no_grad():
                probs_eval = model(x, edge_index, use_batch=True)
                val_probs = probs_eval[val_idx_dev].cpu().numpy()
                val_y = y[val_idx_dev].cpu().numpy()
                val_auc = roc_auc_score(val_y, val_probs)
            del probs_eval
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(f"  epoch {epoch:3d}  loss {loss_value:.4f}  val_auc {val_auc:.4f}  "
                  f"best {best_val_auc:.4f}  patience {epochs_without_improvement}/{PATIENCE}")

            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val AUC-ROC improvement for "
                      f"{PATIENCE} checks, every {VAL_EVERY_N_EPOCHS} epochs)")
                break

    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        probs_full = model(x, edge_index, use_batch=True)[test_idx_dev].cpu().numpy()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        probs_realtime = model(x, edge_index, use_batch=False)[test_idx_dev].cpu().numpy()
    y_test = y[test_idx_dev].cpu().numpy()

    metrics_full = evaluate(y_test, probs_full)
    metrics_realtime = evaluate(y_test, probs_realtime)

    print("\n=== Full model (BatchNet + RealTimeNet, use_batch=True) — test set ===")
    print(metrics_full)
    print("\n=== Real-time only (RealTimeNet, use_batch=False) — test set ===")
    print(metrics_realtime)

    print("\nTier breakdown — full model:")
    breakdown_full = tier_breakdown(y_test, probs_full)
    print(breakdown_full.to_string(index=False))

    print("\nTier breakdown — real-time only:")
    breakdown_realtime = tier_breakdown(y_test, probs_realtime)
    print(breakdown_realtime.to_string(index=False))

    plot_roc(y_test, probs_full, probs_realtime)

    rows = [
        {"model": "FraudGNN", "setting": "full (use_batch=True)", **metrics_full},
        {"model": "FraudGNN", "setting": "realtime-only (use_batch=False)", **metrics_realtime},
    ]
    results_df = pd.DataFrame(rows)
    results_df.to_csv("main_gnn_results.csv", index=False)
    breakdown_full.to_csv("main_gnn_tier_breakdown_full.csv", index=False)
    breakdown_realtime.to_csv("main_gnn_tier_breakdown_realtime.csv", index=False)
    print("\nSaved main_gnn_results.csv, main_gnn_tier_breakdown_full.csv, main_gnn_tier_breakdown_realtime.csv")

    try:
        prior_df = pd.read_csv("full_comparison_results.csv")
        master_df = pd.concat([prior_df, results_df], ignore_index=True)
    except FileNotFoundError:
        master_df = results_df
    master_df.to_csv("master_comparison_results.csv", index=False)
    print("\nMaster comparison table:")
    print(master_df.to_string(index=False))
    print("Saved master_comparison_results.csv")

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "FraudGNN_main.pt")
    torch.save({
        "state_dict": best_state,
        "in_channels": data.num_node_features,
        "hidden_channels": HIDDEN_CHANNELS,
        "dropout": DROPOUT,
        # Cast off numpy.float64 (roc_auc_score's return type) before
        # pickling — a numpy scalar embedded in the checkpoint makes it only
        # loadable by an environment with a matching numpy major version
        # (numpy 2.x reorganized numpy.core -> numpy._core internally).
        "best_val_auc": float(best_val_auc),
        "tiers": {"low_max": LOW_MAX, "review_max": REVIEW_MAX, "high_max": HIGH_MAX},
    }, save_path)
    print(f"Saved model to {save_path}")

    meta_path = os.path.join(MODEL_DIR, "FraudGNN_main_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "best_val_auc": best_val_auc,
            "test_metrics_full": metrics_full,
            "test_metrics_realtime": metrics_realtime,
            "tiers": {"low_max": LOW_MAX, "review_max": REVIEW_MAX, "high_max": HIGH_MAX},
        }, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
