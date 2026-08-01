"""
The single scoring pipeline — gather neighbours from Redis (tagging each
with the relationship it was found through), run inference via
ml_service.score() (Detection Engine, spec item 2), then tier/explain/
alert via decision_engine (Alert & Decision Engine, spec item 4), persist,
register back into Redis, and broadcast over WebSocket. Used by BOTH the
POST /score endpoint (manual/API-triggered scoring) and stream_simulator.py
(the live transaction feed), so there is exactly one code path for "what
happens when a transaction gets scored" — no risk of the two drifting apart.
"""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

import decision_engine
import ml_service
import models
import redis_client
import schemas
from decision_engine import NeighbourInfo
from websocket_manager import manager


def gather_neighbours(request: schemas.ScoreRequest) -> list:
    """Fetches recent transactions for both accounts from Redis, tagging
    each with the edge relation it was found through (SAME_SENDER vs
    SAME_RECEIVER — this is the real graph structure feeding the model,
    matching architecture.md's account-based edge types, not device/card
    linkage which this dataset has no fields for). Dedupes against the
    current transaction and each other, then appends whatever the caller
    explicitly supplied in the request."""
    seen_ids = {request.transaction_id}
    neighbours = []

    for account_id, edge_type in (
        (request.sender_account, "SAME_SENDER"),
        (request.receiver_account, "SAME_RECEIVER"),
    ):
        for entry in redis_client.get_recent_transactions(account_id):
            txn_id = entry.get("transaction_id")
            if not txn_id or txn_id in seen_ids:
                continue
            seen_ids.add(txn_id)
            try:
                features = schemas.TransactionFeatures(**{k: v for k, v in entry.items() if k != "transaction_id"})
            except Exception:
                continue  # skip malformed/legacy entries rather than fail the whole request
            neighbours.append(NeighbourInfo(transaction_id=txn_id, edge_type=edge_type, features=features))

    for i, f in enumerate(request.neighbours):
        neighbours.append(NeighbourInfo(transaction_id=f"supplied-{i}", edge_type="SUPPLIED", features=f))

    return neighbours


async def process_transaction(request: schemas.ScoreRequest, db: Session, source: str = "api") -> dict:
    neighbours = gather_neighbours(request)
    neighbour_features = [n.features for n in neighbours]

    fraud_probability, latency_ms = ml_service.score(request.features, neighbour_features)
    decision, risk_level = decision_engine.tier_decision(fraud_probability)
    explanation = decision_engine.build_explanation(request, neighbours, fraud_probability, db)

    existing = db.get(models.Transaction, request.transaction_id)
    if existing is None:
        f = request.features
        db.add(models.Transaction(
            transaction_id=request.transaction_id,
            sender_account=request.sender_account,
            receiver_account=request.receiver_account,
            amount=f.amount,
            transaction_type=f.type,
            step=int(f.step),
            old_balance_sender=f.oldbalanceOrg,
            new_balance_sender=f.newbalanceOrig,
            old_balance_receiver=f.oldbalanceDest,
            new_balance_receiver=f.newbalanceDest,
        ))

    decided_at = datetime.now(timezone.utc)
    decision_row = models.FraudDecision(
        transaction_id=request.transaction_id,
        fraud_probability=fraud_probability,
        decision=decision,
        risk_level=risk_level,
        explanation=explanation,
        neighbour_count=len(neighbours),
        inference_latency_ms=latency_ms,
        decided_at=decided_at,
    )
    db.add(decision_row)
    db.flush()  # populate decision_row.decision_id before using it below

    alert_id = None
    if decision_engine.should_alert(fraud_probability):
        alert = models.Alert(decision_id=decision_row.decision_id, pushed_at=decided_at)
        db.add(alert)
        db.flush()
        alert_id = alert.alert_id

    db.commit()

    redis_client.store_transaction(
        request.transaction_id, request.sender_account, request.receiver_account, request.features.model_dump()
    )

    await manager.broadcast({
        "type": "new_decision",
        "decision_id": decision_row.decision_id,
        "transaction_id": request.transaction_id,
        "sender_account": request.sender_account,
        "receiver_account": request.receiver_account,
        "amount": request.features.amount,
        "fraud_probability": fraud_probability,
        "decision": decision,
        "risk_level": risk_level,
        "explanation": explanation,
        "inference_latency_ms": latency_ms,
        "decided_at": decided_at.isoformat(),
        "alert_id": alert_id,
        "acknowledged_by": None,
        "is_false_positive": False,
        "escalated": False,
        "source": source,
    })

    return {
        "transaction_id": request.transaction_id,
        "fraud_probability": fraud_probability,
        "decision": decision,
        "risk_level": risk_level,
        "explanation": explanation,
        "neighbour_count": len(neighbours),
        "inference_latency_ms": latency_ms,
    }
