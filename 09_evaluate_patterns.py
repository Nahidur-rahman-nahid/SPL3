"""
Did the retrained FraudGNN learn the injected fraud patterns? Runs the
checks from INJECTED_PATTERNS.md section 3 and prints PASS/FAIL for each:

  A  recall per fraud pattern on holdout instances (>= 80% at REVIEW 0.40)
  B  false-alarm rate per legitimate twin on holdout instances (<= 10%).
     The ACCOUNT_TAKEOVER owner's own earlier payments are not a twin —
     they sit inside the fraud instance — so they're reported separately
     ("guilt by association"), not counted in B.
  C  neighbour effect: fraud scores go UP with neighbours (real-time mode)
  D  graph-only patterns: the GNN separates each fraud pattern from its
     ISOLATED twin (identical rows, no neighbours); XGBoost can't.
     separation = share of fraud flagged - share of twin flagged
  E  original PaySim fraud (test split, injected rows excluded) is not
     hurt: real-time AUC within 0.02 of the previous model

Scores come from the real-time path (use_batch=False, what the backend
runs) unless a column says "full". Injected instances never connect to
PaySim rows or to each other, so scoring the injected-only subgraph gives
exactly the same neighbourhoods as the full graph — cheap and exact.

The previous PaySim-only checkpoint (kept by train_main_gnn.py as
models/FraudGNN_main_v1_paysim_only.pt) is scored with ITS OWN
preprocessing: the raw_* money columns run through the previous scaler
(scaler_v1_paysim_only.pkl, kept by 01_load_paysim.py), with no log1p —
exactly how it was trained.

Run after train_main_gnn.py, in the same folder:
    python 09_evaluate_patterns.py           # all checks (E needs two full-graph passes)
    python 09_evaluate_patterns.py --quick   # skip E and XGBoost

Writes pattern_eval_results.csv, neighbour_effect.csv,
graph_vs_xgb_separation.csv and pattern_recall.png.
"""

import argparse
import importlib.util
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, roc_auc_score
from torch_geometric.utils import subgraph

CLEAN_PATH = "paysim_clean.pkl"
GRAPH_PATH = "paysim_graph.pt"
MODEL_DIR = "/content/drive/MyDrive/SPL3/models"
NEW_MODEL = "FraudGNN_main.pt"
OLD_MODEL = "FraudGNN_main_v1_paysim_only.pt"
OLD_SCALER = "scaler_v1_paysim_only.pkl"
MONEY_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

FEATURE_COLS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
REVIEW, BLOCK = 0.40, 0.75
RECALL_TARGET, FALSE_ALARM_LIMIT, AUC_TOLERANCE = 0.80, 0.10, 0.02
SEPARATION_MARGIN = 0.30  # D: GNN separation must beat XGBoost's by this much

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train = load_module("train_main_gnn", "train_main_gnn.py")  # FraudGNN + the exact same make_splits
# Graph-only pattern -> its isolated twin (identical rows, no neighbours).
ISOLATED_TWINS = load_module("inject_patterns", "00_inject_fraud_patterns.py").ISOLATED_TWINS


def load_model(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = train.FraudGNN(in_channels=ckpt["in_channels"], hidden_channels=ckpt["hidden_channels"], dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def injected_scores(model, x_inj, ei_inj):
    empty = torch.empty((2, 0), dtype=torch.long, device=device)
    return {
        "rt": model(x_inj, ei_inj, use_batch=False).cpu().numpy(),
        "full": model(x_inj, ei_inj, use_batch=True).cpu().numpy(),
        "rt_alone": model(x_inj, empty, use_batch=False).cpu().numpy(),
    }


def previous_model_features(df):
    """Feature matrix as the previous model was trained: raw money through
    the previous scaler, no log1p. None if the pieces aren't there."""
    if not os.path.exists(OLD_SCALER) or not all(f"raw_{c}" in df.columns for c in MONEY_COLS):
        return None
    import joblib

    old_scaler = joblib.load(OLD_SCALER)
    x = df[FEATURE_COLS].values.astype(np.float32).copy()
    money_idx = [FEATURE_COLS.index(c) for c in MONEY_COLS]
    x[:, money_idx] = old_scaler.transform(df[[f"raw_{c}" for c in MONEY_COLS]].values)
    return torch.tensor(x, dtype=torch.float)


@torch.no_grad()
def original_paysim_metrics(model, x_all, data, test_orig_idx):
    x, ei = x_all.to(device), data.edge_index.to(device)
    y = data.y[test_orig_idx].numpy()
    out = {}
    for mode, use_batch in (("rt", False), ("full", True)):
        probs = model(x, ei, use_batch=use_batch)[test_orig_idx].cpu().numpy()
        preds = probs >= BLOCK
        out[mode] = {
            "roc_auc": roc_auc_score(y, probs),
            "recall@0.75": recall_score(y, preds, zero_division=0),
            "precision@0.75": precision_score(y, preds, zero_division=0),
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def xgboost_scores(df, train_idx, val_idx, eval_rows):
    from xgboost import XGBClassifier

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["isFraud"].values
    spw = (y[train_idx] == 0).sum() / max((y[train_idx] == 1).sum(), 1)
    # Same settings and F1-tuned threshold as 07_xgboost_baseline.py.
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=spw,
                          eval_metric="logloss", random_state=42, n_jobs=-1)
    model.fit(X[train_idx], y[train_idx])
    precisions, recalls, thresholds = precision_recall_curve(y[val_idx], model.predict_proba(X[val_idx])[:, 1])
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-12, None)
    threshold = float(thresholds[np.argmax(f1s[:-1])])
    return model.predict_proba(X[eval_rows])[:, 1] >= threshold, threshold


def main():
    parser = argparse.ArgumentParser(description="Per-pattern evaluation of the retrained FraudGNN.")
    parser.add_argument("--quick", action="store_true", help="skip check E (full-graph passes) and XGBoost")
    args = parser.parse_args()

    with open(CLEAN_PATH, "rb") as f:
        df = pickle.load(f)
    data = torch.load(GRAPH_PATH, weights_only=False)
    if "pattern" not in df.columns or not (df["pattern"] != "").any():
        raise SystemExit("paysim_clean.pkl has no injected patterns — run 00, 01 and 03 on PS_augmented.csv first.")

    inj = np.flatnonzero(df["pattern"].values != "")
    ei_inj, _ = subgraph(torch.from_numpy(inj), data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    ei_inj = ei_inj.to(device)

    rows = df.iloc[inj][["pattern", "holdout", "isFraud"]].reset_index(drop=True)
    models = {"new": (os.path.join(MODEL_DIR, NEW_MODEL), data.x)}
    if os.path.exists(os.path.join(MODEL_DIR, OLD_MODEL)):
        old_x = previous_model_features(df)
        if old_x is None:
            print(f"Previous model found but not {OLD_SCALER} / raw_* columns — skipping the before/after comparison.")
        else:
            models["old"] = (os.path.join(MODEL_DIR, OLD_MODEL), old_x)
    loaded = {}
    for name, (path, x_all) in models.items():
        loaded[name] = (load_model(path), x_all)
        for mode, probs in injected_scores(loaded[name][0], x_all[inj].to(device), ei_inj).items():
            rows[f"{name}_{mode}"] = probs
        print(f"Scored {len(inj):,} injected rows with the {name} model ({path})")

    # Strict evaluation on instances never trained on; fall back to all injected rows.
    eval_mask = rows["holdout"].values if rows["holdout"].any() else np.ones(len(rows), dtype=bool)
    ev = rows[eval_mask].copy()

    train_idx, val_idx, test_idx = (t.numpy() for t in train.make_splits(data))
    xgb_threshold = None
    if not args.quick:
        print("Training XGBoost on the same train split (same settings as 07_xgboost_baseline.py)...")
        flags, xgb_threshold = xgboost_scores(df, train_idx, val_idx, inj[eval_mask])
        ev["xgb_flag"] = flags

    # ---------------------------------------------------- A, B, D per pattern
    results = []
    for (pattern, label), g in ev.groupby(["pattern", "isFraud"], sort=False):
        r = {
            "pattern": pattern,
            "label": "fraud" if label == 1 else "legit",
            "rows": len(g),
            "new_rt_flag@0.40": (g["new_rt"] >= REVIEW).mean(),
            "new_rt_flag@0.75": (g["new_rt"] >= BLOCK).mean(),
            "new_full_flag@0.40": (g["new_full"] >= REVIEW).mean(),
        }
        if "old_rt" in g:
            r["old_rt_flag@0.40"] = (g["old_rt"] >= REVIEW).mean()
        if "xgb_flag" in g:
            r["xgb_flag"] = g["xgb_flag"].mean()
        if label == 1:
            r["check"], r["pass"] = "A recall", r["new_rt_flag@0.40"] >= RECALL_TARGET
        elif not pattern.startswith("LEGIT_"):
            r["check"], r["pass"] = "info: owner history inside fraud instance", None
        else:
            r["check"], r["pass"] = "B false alarms", r["new_rt_flag@0.40"] <= FALSE_ALARM_LIMIT
        results.append(r)
    results = pd.DataFrame(results)
    results.to_csv("pattern_eval_results.csv", index=False)

    # ---------------------------------------------------- C neighbour effect
    effect = []
    for (pattern, label), g in ev.groupby(["pattern", "isFraud"], sort=False):
        e = {"pattern": pattern, "label": "fraud" if label == 1 else "legit"}
        for name in loaded:
            delta = g[f"{name}_rt"] - g[f"{name}_rt_alone"]
            e[f"{name}_alone"] = g[f"{name}_rt_alone"].mean()
            e[f"{name}_with_neighbours"] = g[f"{name}_rt"].mean()
            e[f"{name}_delta"] = delta.mean()
        effect.append(e)
    effect = pd.DataFrame(effect)
    effect.to_csv("neighbour_effect.csv", index=False)

    # ---------------------------------------------------- D graph vs tabular
    flag = results.set_index(["pattern", "label"])
    separation = []
    for pattern, twin in ISOLATED_TWINS.items():
        if (pattern, "fraud") not in flag.index or (twin, "legit") not in flag.index:
            continue
        r = {"pattern": pattern, "isolated_twin": twin}
        for col, name in (("new_rt_flag@0.40", "gnn"), ("old_rt_flag@0.40", "old_gnn"), ("xgb_flag", "xgb")):
            if col in flag.columns:
                r[f"{name}_separation"] = flag.loc[(pattern, "fraud"), col] - flag.loc[(twin, "legit"), col]
        separation.append(r)
    separation = pd.DataFrame(separation)
    separation.to_csv("graph_vs_xgb_separation.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(f"\n=== A/B: share flagged (REVIEW or above) on {'holdout' if rows['holdout'].any() else 'all'} injected rows ===")
    print(results.to_string(index=False))
    if xgb_threshold is not None:
        print(f"(XGBoost flags at its own F1-tuned threshold {xgb_threshold:.3f}, as in 07_xgboost_baseline.py)")
    print("\n=== C: mean real-time score alone vs with neighbours ===")
    print(effect.to_string(index=False))
    print("\n=== D: separation from the isolated twin (fraud flagged - twin flagged; 1.0 = perfect, 0 = can't tell) ===")
    print(separation.to_string(index=False))

    # ---------------------------------------------------- E original PaySim
    e_result = None
    if not args.quick:
        test_orig = test_idx[df["pattern"].values[test_idx] == ""]
        print(f"\n=== E: original PaySim fraud, {len(test_orig):,} test rows (injected excluded) ===")
        e_result = {name: original_paysim_metrics(m, x_all, data, torch.from_numpy(test_orig))
                    for name, (m, x_all) in loaded.items()}
        for name, metrics in e_result.items():
            for mode, m in metrics.items():
                print(f"  {name:3} {mode:4}  AUC {m['roc_auc']:.4f}  recall@0.75 {m['recall@0.75']:.3f}  precision@0.75 {m['precision@0.75']:.3f}")

    # ---------------------------------------------------- summary
    fraud_rows = results[results["label"] == "fraud"]
    legit_rows = results[results["check"] == "B false alarms"]
    context_rows = results[results["check"].str.startswith("info")]
    fraud_effect = effect[effect["label"] == "fraud"]
    print("\n=== Checklist ===")
    print(f"  A  every fraud pattern caught >= {RECALL_TARGET:.0%}: {'PASS' if fraud_rows['pass'].all() else 'FAIL'}"
          f"  (lowest: {fraud_rows.loc[fraud_rows['new_rt_flag@0.40'].idxmin(), 'pattern']} "
          f"{fraud_rows['new_rt_flag@0.40'].min():.0%})")
    print(f"  B  every legit twin false alarms <= {FALSE_ALARM_LIMIT:.0%}: {'PASS' if legit_rows['pass'].all() else 'FAIL'}"
          f"  (highest: {legit_rows.loc[legit_rows['new_rt_flag@0.40'].idxmax(), 'pattern']} "
          f"{legit_rows['new_rt_flag@0.40'].max():.0%})")
    for _, r in context_rows.iterrows():
        print(f"     (info) {r['pattern']} owner's normal payments in the same group flagged: {r['new_rt_flag@0.40']:.0%}"
              f" — guilt by association, not counted in B")
    print(f"  C  neighbours raise fraud scores (every pattern): {'PASS' if (fraud_effect['new_delta'] > 0).all() else 'FAIL'}")
    if "xgb_separation" in separation:
        beats = (separation["gnn_separation"] >= separation["xgb_separation"] + SEPARATION_MARGIN).all()
        print(f"  D  GNN separates every graph-only pattern from its identical isolated twin, XGBoost can't "
              f"(margin {SEPARATION_MARGIN}): {'PASS' if beats else 'FAIL'}  "
              f"(GNN min {separation['gnn_separation'].min():.2f}, XGBoost max {separation['xgb_separation'].max():.2f})")
    if e_result and "old" in e_result:
        drop = e_result["old"]["rt"]["roc_auc"] - e_result["new"]["rt"]["roc_auc"]
        print(f"  E  original PaySim real-time AUC drop <= {AUC_TOLERANCE}: {'PASS' if drop <= AUC_TOLERANCE else 'FAIL'}  (change {-drop:+.4f})")

    try:
        import matplotlib.pyplot as plt

        cols = [c for c in ("old_rt_flag@0.40", "xgb_flag", "new_rt_flag@0.40") if c in fraud_rows]
        labels = {"old_rt_flag@0.40": "Previous GNN", "xgb_flag": "XGBoost", "new_rt_flag@0.40": "Retrained GNN"}
        ax = fraud_rows.set_index("pattern")[cols].rename(columns=labels).plot.bar(figsize=(10, 5), rot=20)
        ax.axhline(RECALL_TARGET, color="k", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Share of fraud rows caught (REVIEW or above)")
        ax.set_title("Injected fraud patterns: caught on holdout instances")
        plt.tight_layout()
        plt.savefig("pattern_recall.png", dpi=150)
        print("\nSaved pattern_eval_results.csv, neighbour_effect.csv, graph_vs_xgb_separation.csv, pattern_recall.png")
    except ImportError:
        print("\nSaved pattern_eval_results.csv, neighbour_effect.csv, graph_vs_xgb_separation.csv")


if __name__ == "__main__":
    main()
