"""
Load and clean the PaySim dataset.

Expects PS_20174392719_1491204439457_log.csv to be reachable at DATA_PATH
(e.g. Google Drive mounted in Colab: /content/drive/MyDrive/.../PS_...csv).

Fraud in PaySim only occurs in TRANSFER and CASH_OUT transactions, so every
other type is dropped to cut memory usage and training time.
"""

import pickle

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Update this path if your file lives elsewhere (e.g. Colab + Drive mount).
DATA_PATH = "/content/drive/MyDrive/SPL3/PS_20174392719_1491204439457_log.csv"
OUTPUT_PATH = "paysim_clean.pkl"
TYPE_ENCODER_PATH = "type_encoder.pkl"
SCALER_PATH = "scaler.pkl"

FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

BALANCE_COLS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def load_and_clean(path: str):
    df = pd.read_csv(path)

    # Fraud only happens on TRANSFER / CASH_OUT — drop everything else.
    df = df[df["type"].isin(FRAUD_TYPES)].reset_index(drop=True)

    # Label-encode the transaction type.
    type_encoder = LabelEncoder()
    df["type"] = type_encoder.fit_transform(df["type"])

    # Standard-scale amount and balance columns.
    scale_cols = ["amount"] + BALANCE_COLS
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df, type_encoder, scaler


def main():
    df, type_encoder, scaler = load_and_clean(DATA_PATH)

    fraud_ratio = df["isFraud"].mean()
    print(f"Shape: {df.shape}")
    print(f"Fraud ratio: {fraud_ratio:.6f} ({df['isFraud'].sum()} fraud / {len(df)} total)")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(df, f)
    print(f"Saved cleaned dataframe to {OUTPUT_PATH}")

    # Persisted so the backend can apply the IDENTICAL transform to raw
    # incoming transactions before scoring — without this, real-time
    # inference would silently see differently-scaled inputs than training.
    joblib.dump(type_encoder, TYPE_ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved {TYPE_ENCODER_PATH} and {SCALER_PATH}")


if __name__ == "__main__":
    main()
