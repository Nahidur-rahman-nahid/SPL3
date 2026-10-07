"""
Load and clean the PaySim dataset.

Reads PS_augmented.csv — PaySim plus the graph fraud patterns from
00_inject_fraud_patterns.py — at DATA_PATH (Google Drive mounted in Colab).
Amount and balances are log1p-transformed before standard scaling. PaySim
money is extremely spread out (amount std ~BDT 887,000), so plain scaling
squeezed a 15,000 bill and an 80,000 drain to almost the same number and
the GNN couldn't tell rows apart — it fell back on "has neighbours". The
transform is written to preprocess.json; train_main_gnn.py stores it in
the checkpoint and the backend applies the same one (backend/ml_service.py).

The previous scaler/encoder are kept once as *_v1_paysim_only.pkl so
09_evaluate_patterns.py can still score the previous model correctly. The
raw (unscaled) money columns are kept as raw_* for the same reason.

If injected_manifest.csv sits next to it, every injected row is tagged with
its `pattern` and `holdout` flag (holdout rows go only to the test split in
train_main_gnn.py, and 09_evaluate_patterns.py reports per pattern). On the
plain PaySim CSV with no manifest the script behaves exactly as before.

Fraud in PaySim only occurs in TRANSFER and CASH_OUT transactions, so every
other type is dropped to cut memory usage and training time.
"""

import json
import os
import pickle
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Update this path if your file lives elsewhere (e.g. Colab + Drive mount).
DATA_PATH = "/content/drive/MyDrive/SPL3/PS_augmented.csv"
MANIFEST_PATH = "/content/drive/MyDrive/SPL3/injected_manifest.csv"
OUTPUT_PATH = "paysim_clean.pkl"
TYPE_ENCODER_PATH = "type_encoder.pkl"
SCALER_PATH = "scaler.pkl"
PREPROCESS_PATH = "preprocess.json"
MONEY_TRANSFORM = "log1p"  # applied to amount + balances before StandardScaler

FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

BALANCE_COLS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def tag_injected(df: pd.DataFrame, manifest_path: str) -> pd.DataFrame:
    """Injected rows are appended at the very end of PS_augmented.csv in
    manifest order (all TRANSFER/CASH_OUT, so the type filter keeps them),
    which makes them the last len(manifest) rows here. Verified by account
    id, not assumed."""
    df["pattern"] = ""
    df["holdout"] = False
    if not os.path.exists(manifest_path):
        print(f"No manifest at {manifest_path} — treating every row as plain PaySim.")
        return df

    manifest = pd.read_csv(manifest_path)
    tail = df.index[-len(manifest):]
    same_accounts = (
        (df.loc[tail, "nameOrig"].values == manifest["nameOrig"].values).all()
        and (df.loc[tail, "nameDest"].values == manifest["nameDest"].values).all()
    )
    if not same_accounts:
        raise ValueError("Last rows of the CSV don't match injected_manifest.csv — re-run 00_inject_fraud_patterns.py.")
    df.loc[tail, "pattern"] = manifest["pattern"].values
    df.loc[tail, "holdout"] = manifest["holdout"].values.astype(bool)
    print(f"Tagged {len(manifest):,} injected rows "
          f"({int(manifest['isFraud'].sum()):,} fraud, {int(manifest['holdout'].sum()):,} in holdout instances).")
    return df


def load_and_clean(path: str, manifest_path: str = MANIFEST_PATH):
    df = pd.read_csv(path)

    # Fraud only happens on TRANSFER / CASH_OUT — drop everything else.
    df = df[df["type"].isin(FRAUD_TYPES)].reset_index(drop=True)
    df = tag_injected(df, manifest_path)

    # Label-encode the transaction type.
    type_encoder = LabelEncoder()
    df["type"] = type_encoder.fit_transform(df["type"])

    # log1p, then standard-scale amount and balance columns. Raw copies are
    # kept for evaluating the previous (non-log) model.
    scale_cols = ["amount"] + BALANCE_COLS
    for col in scale_cols:
        df[f"raw_{col}"] = df[col]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(np.log1p(df[scale_cols].clip(lower=0)))

    return df, type_encoder, scaler


def main():
    df, type_encoder, scaler = load_and_clean(DATA_PATH)

    fraud_ratio = df["isFraud"].mean()
    print(f"Shape: {df.shape}")
    print(f"Fraud ratio: {fraud_ratio:.6f} ({df['isFraud'].sum()} fraud / {len(df)} total)")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(df, f)
    print(f"Saved cleaned dataframe to {OUTPUT_PATH}")

    # Keep the previous preprocessing once, for scoring the previous model.
    for path in (TYPE_ENCODER_PATH, SCALER_PATH):
        backup = path.replace(".pkl", "_v1_paysim_only.pkl")
        if os.path.exists(path) and not os.path.exists(backup) and not os.path.exists(PREPROCESS_PATH):
            shutil.copyfile(path, backup)
            print(f"Kept the previous {path} as {backup}")

    # Persisted so the backend can apply the IDENTICAL transform to raw
    # incoming transactions before scoring — without this, real-time
    # inference would silently see differently-scaled inputs than training.
    joblib.dump(type_encoder, TYPE_ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(PREPROCESS_PATH, "w") as f:
        json.dump({"money_transform": MONEY_TRANSFORM}, f)
    print(f"Saved {TYPE_ENCODER_PATH}, {SCALER_PATH} and {PREPROCESS_PATH}")


if __name__ == "__main__":
    main()
