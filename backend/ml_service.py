"""
Loads the trained FraudGNN checkpoint once at startup and scores individual
transactions in real time (use_batch=False — RealTimeNet only, matching
architecture.md section 5: no full graph is ever loaded in production).
"""

import time

import joblib
import torch

from config import settings
from gnn_model import FraudGNN
from schemas import ScoreRequest, TransactionFeatures

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


def _tier_decision(prob: float) -> tuple:
    if prob < _tiers["low_max"]:
        return "ALLOW", "LOW"
    if prob < _tiers["review_max"]:
        return "REVIEW", "MEDIUM"
    if prob < _tiers["high_max"]:
        return "BLOCK", "HIGH"
    return "BLOCK", "CRITICAL"


def score_transaction(request: ScoreRequest) -> dict:
    if _model is None:
        raise RuntimeError("Model not loaded — call load_model() at startup")

    start = time.perf_counter()

    all_features = [request.features] + list(request.neighbours)
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
    fraud_probability = float(probs[0].item())

    decision, risk_level = _tier_decision(fraud_probability)
    latency_ms = int((time.perf_counter() - start) * 1000)

    # Neighbour details for the frontend's subgraph visualization — this is
    # literally the same star-graph structure just fed to RealTimeNet above
    # (node 0 = target, nodes 1..N = neighbours), not a separate illustration.
    neighbour_details = [
        {"id": f"n{i}", "type": f.type, "amount": f.amount, "step": f.step}
        for i, f in enumerate(request.neighbours)
    ]

    explanation = {
        "neighbour_count": len(request.neighbours),
        "summary": (
            f"{len(request.neighbours)} related transaction(s) found for "
            f"{request.sender_account} / {request.receiver_account} within the lookup window."
            if request.neighbours
            else "No related transactions found in the lookup window — scored on this transaction's own features alone."
        ),
        "neighbours": neighbour_details,
    }

    return {
        "transaction_id": request.transaction_id,
        "fraud_probability": fraud_probability,
        "decision": decision,
        "risk_level": risk_level,
        "explanation": explanation,
        "neighbour_count": len(request.neighbours),
        "inference_latency_ms": latency_ms,
    }
