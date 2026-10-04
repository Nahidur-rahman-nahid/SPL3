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
from typing import Optional

from sqlalchemy.orm import Session

import decision_engine
import ml_service
import models
import redis_client
import schemas
from config import settings
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


def check_regulatory_limits(request: schemas.ScoreRequest, db: Session) -> Optional[dict]:
    """Bangladesh Bank daily limit check, run before scoring. Reserves the
    amount against the sender's daily total for its type; if that would
    exceed the limit, the result is marked violated and the caller hard-
    BLOCKs with risk_level="REGULATORY". Returns None when the check doesn't
    apply (disabled, no limit for this type) or Redis is unavailable — fail
    open, same policy as the rest of redis_client.

    A transaction_id that has been scored before is never counted twice: it
    keeps its earlier regulatory verdict and just reports current usage."""
    if not settings.REGULATORY_LIMITS_ENABLED:
        return None
    txn_type = request.features.type
    limit = settings.daily_limits.get(txn_type)
    if limit is None:
        return None

    amount = request.features.amount
    when = datetime.now(timezone.utc)
    check = {
        "rule": f"BB_DAILY_{txn_type}_LIMIT",
        "type": txn_type,
        "limit": limit,
        "attempted": amount,
        "violated": False,
        "reserved": False,
        "when": when,
    }

    previous = (
        db.query(models.FraudDecision.risk_level)
        .filter(models.FraudDecision.transaction_id == request.transaction_id)
        .order_by(models.FraudDecision.decided_at.desc())
        .first()
    )
    if previous is not None or amount <= 0:
        used = redis_client.get_daily_total(request.sender_account, txn_type, when)
        if used is None:
            return None
        check["used"] = used
        check["violated"] = previous is not None and previous.risk_level == "REGULATORY"
        return check

    reserved = redis_client.reserve_daily_amount(request.sender_account, txn_type, amount, limit, when)
    if reserved is None:
        return None
    accepted, total = reserved
    check.update(used=total, violated=not accepted, reserved=accepted)
    return check


def _release_reservation(request: schemas.ScoreRequest, limit_check: Optional[dict]) -> None:
    if limit_check and limit_check["reserved"]:
        redis_client.release_daily_amount(
            request.sender_account, limit_check["type"], limit_check["attempted"], limit_check["when"]
        )
        limit_check["reserved"] = False
        limit_check["used"] = round(limit_check["used"] - limit_check["attempted"], 2)


def _limit_usage(limit_check: Optional[dict]) -> Optional[dict]:
    if limit_check is None:
        return None
    return {
        "type": limit_check["type"],
        "period": "daily",
        "used": limit_check["used"],
        "limit": limit_check["limit"],
        "remaining": round(max(limit_check["limit"] - limit_check["used"], 0), 2),
    }


def _attach_regulatory_violation(explanation: dict, limit_check: dict) -> None:
    explanation["regulatory_violation"] = {
        "rule": limit_check["rule"],
        "limit": limit_check["limit"],
        "used_today": limit_check["used"],
        "attempted": limit_check["attempted"],
    }
    note = (
        f"Blocked by Bangladesh Bank daily {limit_check['type']} limit: BDT {limit_check['used']:,.0f} "
        f"already used today + BDT {limit_check['attempted']:,.0f} would exceed BDT {limit_check['limit']:,.0f}."
    )
    explanation["summary"] = f"{note} {explanation.get('summary', '')}".strip()


async def process_transaction(request: schemas.ScoreRequest, db: Session, source: str = "api") -> dict:
    # Rule-based checks (regulatory limits, night rule) apply to API-scored
    # transactions only, so the stream simulator's demo feed behaves exactly
    # as before.
    limit_check = check_regulatory_limits(request, db) if source == "api" else None
    try:
        return await _score_and_record(request, db, source, limit_check)
    except Exception:
        _release_reservation(request, limit_check)  # failed scoring must not eat into the limit
        raise


async def _score_and_record(
    request: schemas.ScoreRequest, db: Session, source: str, limit_check: Optional[dict]
) -> dict:
    neighbours = gather_neighbours(request)
    neighbour_features = [n.features for n in neighbours]

    # The model still runs on a regulatory block so fraud_probability stays a
    # real score (stats, alerting) rather than a placeholder.
    fraud_probability, latency_ms = ml_service.score(request.features, neighbour_features)
    decision, risk_level = decision_engine.tier_decision(fraud_probability)
    explanation = decision_engine.build_explanation(request, neighbours, fraud_probability, db)

    night_rule = None
    if limit_check and limit_check["violated"]:
        decision, risk_level = "BLOCK", "REGULATORY"
        _attach_regulatory_violation(explanation, limit_check)
    elif source == "api":
        decision, risk_level, night_rule = decision_engine.apply_night_rule(request, decision, risk_level)
        if night_rule:
            decision_engine.attach_rule_to_explanation(explanation, night_rule)

    if decision == "BLOCK":
        _release_reservation(request, limit_check)  # blocked money doesn't count toward the limit

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
    # A rule-forced REVIEW must reach the analyst queue even when the model
    # score alone is below ALERT_THRESHOLD.
    rule_forced_review = night_rule is not None and night_rule["changed_decision"]
    if decision_engine.should_alert(fraud_probability) or rule_forced_review:
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
        "limit_usage": _limit_usage(limit_check),
    }
