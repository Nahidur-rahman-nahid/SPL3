"""
XGBoost baseline — a plain tabular classifier on the same PaySim features
used by the GNNs, with NO graph structure at all. This answers the question
a supervisor or buyer will ask first: "does the graph actually help, or
would a standard classifier do just as well?"

Uses the exact same train/val/test split (same seed, same proportions) as
06_train_compare.py so the numbers are directly comparable, and the same
threshold-tuning approach (pick the cutoff that maximizes F1 on the
validation set, apply it to the test set) instead of a blind 0.5 cutoff.

Requires paysim_clean.pkl from 01_load_paysim.py in the working directory.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
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
from xgboost import XGBClassifier

DATA_PATH = "paysim_clean.pkl"
# Save into Drive, not local Colab storage — local /content is wiped when the
# runtime disconnects/restarts, so anything only saved there is lost.
MODEL_DIR = "/content/drive/MyDrive/SPL3/models"
FULL_FEATURE_COLS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
# See the matching comment in 03_graph_builder.py: PaySim's fraud pattern
# nearly always drains the origin account to ~0, so oldbalanceOrg/
# newbalanceOrig let a tabular model almost solve fraud detection alone.
# Flip to REDUCED_FEATURE_COLS (and keep 03_graph_builder.py's FEATURE_COLS
# in sync) to compare XGBoost against the GNNs on a fair, non-leaky feature
# set — that's the real test of whether graph structure adds value.
REDUCED_FEATURE_COLS = ["step", "type", "amount"]

FEATURE_COLS = FULL_FEATURE_COLS
LABEL_COL = "isFraud"

TEST_SIZE = 0.2
VAL_SIZE = 0.1
SEED = 42


def make_splits(y):
    """Same seed, same proportions, same order of operations as
    06_train_compare.py's make_splits, so the resulting row indices line up
    with the GNN splits (row order is identical: paysim_clean.pkl and
    paysim_graph.pt both come from the same reset_index(drop=True) dataframe)."""
    idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    val_fraction = VAL_SIZE / (1 - TEST_SIZE)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_fraction, stratify=y[train_val_idx], random_state=SEED
    )
    return train_idx, val_idx, test_idx


def tune_threshold(y_val, probs_val):
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
    if len(thresholds) == 0:
        return 0.5
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    return float(thresholds[np.argmax(f1s[:-1])])


def evaluate(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    try:
        roc_auc = roc_auc_score(y_true, probs)
    except ValueError:
        roc_auc = float("nan")
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


def main():
    with open(DATA_PATH, "rb") as f:
        df = pickle.load(f)
    df = df.reset_index(drop=True)

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    train_idx, val_idx, test_idx = make_splits(y)
    print(f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    neg, pos = np.bincount(y[train_idx])
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight (train): {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X[train_idx], y[train_idx])

    probs_val = model.predict_proba(X[val_idx])[:, 1]
    threshold = tune_threshold(y[val_idx], probs_val)
    print(f"Tuned threshold (from validation set): {threshold:.4f}")

    probs_test = model.predict_proba(X[test_idx])[:, 1]
    metrics = evaluate(y[test_idx], probs_test, threshold)

    print("\nXGBoost (tabular, no graph) — test set results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nFeature importance:")
    for name, importance in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:<16} {importance:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "xgboost_fraud.json")
    model.save_model(model_path)
    meta_path = os.path.join(MODEL_DIR, "xgboost_fraud_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"threshold": threshold, "feature_cols": FEATURE_COLS}, f)
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {meta_path}")

    row = {"model": "XGBoost", "setting": "tabular (no graph)", **metrics}
    try:
        gnn_df = pd.read_csv("comparison_results.csv")
        combined = pd.concat([gnn_df, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        print("\nNote: comparison_results.csv not found (run 06_train_compare.py first "
              "for a combined table) — saving XGBoost result on its own.")
        combined = pd.DataFrame([row])

    combined.to_csv("full_comparison_results.csv", index=False)
    print("\nFull comparison table:")
    print(combined.to_string(index=False))
    print("Saved full_comparison_results.csv")


if __name__ == "__main__":
    main()
