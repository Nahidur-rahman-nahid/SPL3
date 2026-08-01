"""
Alert & Decision Engine (spec item 4) — everything downstream of a raw
fraud probability from ml_service.score() (spec item 2, pure GNN
inference). This module owns:

  - tier_decision()   — turns a probability into ALLOW/REVIEW/BLOCK
  - build_explanation() — a REAL relationship-based explanation: which
    edge type (SAME_SENDER/SAME_RECEIVER) linked each neighbour, whether
    that neighbour was itself previously flagged, and a per-neighbour
    contribution score computed by leave-one-out re-inference (score with
    vs without that neighbour — RealTimeNet's SAGEConv has no built-in
    attention weights to read off directly, so this is the honest way to
    attribute "how much did this specific relationship move the score").
  - should_alert()   — a threshold independent of the tiering thresholds
    above, so alerting policy (who lands in the review queue) can diverge
    from and be tuned separately from decision tiering.

Deliberately a separate module from ml_service.py: swap the model
checkpoint and this file doesn't change; retune alerting policy and
ml_service.py doesn't change.
"""

from dataclasses import dataclass

from sqlalchemy.orm import Session

import ml_service
import models
from config import settings
from schemas import ScoreRequest, TransactionFeatures

# Leave-one-out contribution scoring costs one extra forward pass per
# neighbour — cheap for a handful of neighbours, not worth it past this
# many (bounds worst-case /score latency for a very chatty account).
MAX_CONTRIBUTION_NEIGHBOURS = 15


@dataclass
class NeighbourInfo:
    transaction_id: str
    edge_type: str  # SAME_SENDER, SAME_RECEIVER, SUPPLIED
    features: TransactionFeatures


def tier_decision(probability: float) -> tuple:
    tiers = ml_service.current_tiers()
    if probability < tiers["low_max"]:
        return "ALLOW", "LOW"
    if probability < tiers["review_max"]:
        return "REVIEW", "MEDIUM"
    if probability < tiers["high_max"]:
        return "BLOCK", "HIGH"
    return "BLOCK", "CRITICAL"


def should_alert(probability: float) -> bool:
    return probability >= settings.ALERT_THRESHOLD


def _prior_decisions(db: Session, transaction_ids: list) -> dict:
    if not transaction_ids:
        return {}
    rows = (
        db.query(models.FraudDecision.transaction_id, models.FraudDecision.decision)
        .filter(models.FraudDecision.transaction_id.in_(transaction_ids))
        .order_by(models.FraudDecision.decided_at.desc())
        .all()
    )
    prior = {}
    for txn_id, decision in rows:
        prior.setdefault(txn_id, decision)  # first hit wins = most recent, thanks to the ORDER BY
    return prior


def _contribution_scores(request: ScoreRequest, neighbours: list, base_probability: float) -> dict:
    all_features = [n.features for n in neighbours]
    contributions = {}
    for i, n in enumerate(neighbours[:MAX_CONTRIBUTION_NEIGHBOURS]):
        reduced = all_features[:i] + all_features[i + 1:]
        reduced_probability, _ = ml_service.score(request.features, reduced)
        contributions[n.transaction_id] = round(base_probability - reduced_probability, 4)
    return contributions


def _summary_sentence(neighbours: list, prior: dict) -> str:
    sender_edges = sum(1 for n in neighbours if n.edge_type == "SAME_SENDER")
    receiver_edges = sum(1 for n in neighbours if n.edge_type == "SAME_RECEIVER")
    flagged_count = sum(1 for v in prior.values() if v != "ALLOW")

    relation_bits = []
    if sender_edges:
        relation_bits.append(f"{sender_edges} via SAME_SENDER")
    if receiver_edges:
        relation_bits.append(f"{receiver_edges} via SAME_RECEIVER")
    relation_str = " and ".join(relation_bits) if relation_bits else f"{len(neighbours)} manually supplied"

    flagged_str = ""
    if flagged_count:
        was_were = "was" if flagged_count == 1 else "were"
        flagged_str = f", {flagged_count} of which {was_were} previously flagged (REVIEW/BLOCK)"

    return f"Linked to {len(neighbours)} related transaction(s) — {relation_str} — within the lookup window{flagged_str}."


def build_explanation(request: ScoreRequest, neighbours: list, base_probability: float, db: Session) -> dict:
    if not neighbours:
        return {
            "summary": "No related transactions found in the lookup window — scored on this transaction's own features alone.",
            "neighbour_count": 0,
            "neighbours": [],
        }

    real_ids = [n.transaction_id for n in neighbours if n.edge_type != "SUPPLIED"]
    prior = _prior_decisions(db, real_ids)
    contributions = _contribution_scores(request, neighbours, base_probability)

    neighbour_out = [
        {
            "transaction_id": n.transaction_id,
            "edge_type": n.edge_type,
            "type": n.features.type,
            "amount": n.features.amount,
            "step": n.features.step,
            "contribution": contributions.get(n.transaction_id),
            "prior_decision": prior.get(n.transaction_id),
        }
        for n in neighbours
    ]
    # Most-impactful relationship first — the point of computing this at all.
    neighbour_out.sort(key=lambda item: abs(item["contribution"] or 0), reverse=True)

    return {
        "summary": _summary_sentence(neighbours, prior),
        "neighbour_count": len(neighbours),
        "neighbours": neighbour_out,
    }
