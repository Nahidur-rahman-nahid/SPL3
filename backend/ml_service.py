"""
Loads the trained FraudGNN checkpoint once at startup and scores individual
transactions in real time (use_batch=False — RealTimeNet only, matching
architecture.md section 5: no full graph is ever loaded in production).

This is the Detection Engine (spec item 2) and ONLY the detection engine —
it answers "how likely is this fraud" and nothing else. Tiering
(ALLOW/REVIEW/BLOCK), explanations, and alert policy live in
decision_engine.py (spec item 4), which calls score() below. Keeping the
seam here means swapping the model checkpoint never touches alerting
policy, and retuning alerting policy never touches inference.
"""

import time

import joblib
import torch

from config import settings
from gnn_model import FraudGNN
from schemas import TransactionFeatures

# Must match 01_load_paysim.py exactly: scale_cols = ["amount"] + BALANCE_COLS
SCALE_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_type_encoder = None
_scaler = None
_tiers = {"low_max": settings.LOW_MAX, "review_max": settings.REVIEW_MAX, "high_max": settings.HIGH_MAX}


def load_model():
    global _model, _tiers, _type_encoder, _scaler

    checkpoint = torch.load(settings.MODEL_PATH, map_location=_device, weights_only=False)
    _model = FraudGNN(
        in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint["hidden_channels"],
        dropout=checkpoint["dropout"],
    ).to(_device)
    _model.load_state_dict(checkpoint["state_dict"])
    _model.eval()
    _tiers = checkpoint.get("tiers", _tiers)

    _type_encoder = joblib.load(settings.TYPE_ENCODER_PATH)
    _scaler = joblib.load(settings.SCALER_PATH)

    return _model


def current_tiers() -> dict:
    """Read-only access for decision_engine.py's tiering — tiers are owned
    here because they can be embedded in the model checkpoint itself."""
    return _tiers


def _features_to_vector(features: TransactionFeatures) -> list:
    """Applies the SAME LabelEncoder + StandardScaler fit during training
    (01_load_paysim.py) to a raw transaction, so real-time inference sees
    inputs on the identical scale the model was trained on."""
    type_encoded = float(_type_encoder.transform([features.type])[0])

    raw_scale_values = [[getattr(features, col) for col in SCALE_COLS]]
    scaled = _scaler.transform(raw_scale_values)[0]
    amount_s, old_orig_s, new_orig_s, old_dest_s, new_dest_s = scaled

    # Order must match FEATURE_COLS from 03_graph_builder.py / training.
    return [features.step, type_encoded, amount_s, old_orig_s, new_orig_s, old_dest_s, new_dest_s]


def _infer(target: TransactionFeatures, neighbours: list) -> float:
    if _model is None:
        raise RuntimeError("Model not loaded — call load_model() at startup")

    all_features = [target] + list(neighbours)
    x = torch.tensor([_features_to_vector(f) for f in all_features], dtype=torch.float, device=_device)

    n = len(all_features)
    if n > 1:
        # Star graph: target (node 0) connected to every neighbour, both
        # directions, so RealTimeNet's mean aggregation pools neighbour info
        # into the target's embedding.
        src = [0] * (n - 1) + list(range(1, n))
        dst = list(range(1, n)) + [0] * (n - 1)
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=_device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=_device)

    with torch.no_grad():
        probs = _model(x, edge_index, use_batch=False)
    return float(probs[0].item())


def score(target: TransactionFeatures, neighbours: list) -> tuple:
    """Pure inference: raw fraud probability (0-1) for `target` given its
    subgraph `neighbours`, plus how long that took. No tiering, no
    persistence, no side effects — safe to call repeatedly (decision_engine
    uses this for leave-one-out neighbour contribution scoring)."""
    start = time.perf_counter()
    fraud_probability = _infer(target, neighbours)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return fraud_probability, latency_ms
