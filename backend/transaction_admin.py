"""
Admin-only deletion of scored transactions — for clearing out dummy/demo
data (stream simulator output, manual test scores from the score form).

Hard delete: a transaction, every FraudDecision for it, and every Alert on
those decisions go in one DB transaction. The foreign keys have no ON
DELETE CASCADE, so children are deleted first (alerts → decisions →
transactions). Afterwards Redis is cleaned up best-effort: deleted IDs are
dropped from the neighbour lists (so they stop appearing as "related
transactions" in new explanations), and amounts that counted toward
today's daily limit are taken back off.

"source" classifies a transaction by its ID: the stream simulator always
issues "stream-…" IDs (stream_simulator._next_id); everything else came in
through POST /score ("manual").
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

import models
import redis_client

STREAM_ID_PREFIX = "stream-"
CHUNK_SIZE = 500  # keeps each IN (...) list a reasonable size on Postgres


def _chunks(items: list, size: int = CHUNK_SIZE):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _to_naive_utc(value: datetime) -> datetime:
    # received_at is stored as "timestamp without time zone" holding UTC.
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc)
    return value.replace(tzinfo=None)


def _matching_query(db: Session, source: str, before: Optional[datetime]):
    query = db.query(models.Transaction.transaction_id)
    if source == "stream":
        query = query.filter(models.Transaction.transaction_id.like(f"{STREAM_ID_PREFIX}%"))
    elif source == "manual":
        query = query.filter(~models.Transaction.transaction_id.like(f"{STREAM_ID_PREFIX}%"))
    if before is not None:
        query = query.filter(models.Transaction.received_at < _to_naive_utc(before))
    return query


def count_matching(db: Session, source: str, before: Optional[datetime] = None) -> int:
    return _matching_query(db, source, before).count()


def _counted_daily_amounts(db: Session, transaction_ids: list) -> list:
    """Manual transactions whose FIRST decision wasn't a BLOCK — those are
    the ones scoring_pipeline counted toward the sender's daily limit
    (re-scores never count again, BLOCKs are released)."""
    first_decision = {}
    rows = (
        db.query(models.FraudDecision.transaction_id, models.FraudDecision.decision, models.FraudDecision.decided_at)
        .filter(models.FraudDecision.transaction_id.in_(transaction_ids))
        .order_by(models.FraudDecision.decided_at.asc())
        .all()
    )
    for txn_id, decision, decided_at in rows:
        first_decision.setdefault(txn_id, (decision, decided_at))
    return [
        (txn_id, decided_at)
        for txn_id, (decision, decided_at) in first_decision.items()
        if decision != "BLOCK" and not txn_id.startswith(STREAM_ID_PREFIX)
    ]


def delete_transactions(db: Session, transaction_ids: list) -> dict:
    """Deletes the given transactions (unknown IDs are ignored) and returns
    how many rows of each kind were removed."""
    unique_ids = list(dict.fromkeys(transaction_ids))
    totals = {"deleted_transactions": 0, "deleted_decisions": 0, "deleted_alerts": 0}
    redis_entries = []
    daily_refunds = []

    for chunk in _chunks(unique_ids):
        txns = (
            db.query(
                models.Transaction.transaction_id,
                models.Transaction.sender_account,
                models.Transaction.receiver_account,
                models.Transaction.transaction_type,
                models.Transaction.amount,
            )
            .filter(models.Transaction.transaction_id.in_(chunk))
            .all()
        )
        found = [t.transaction_id for t in txns]
        if not found:
            continue

        by_id = {t.transaction_id: t for t in txns}
        for txn_id, decided_at in _counted_daily_amounts(db, found):
            t = by_id[txn_id]
            daily_refunds.append((t.sender_account, t.transaction_type, t.amount, decided_at))
        redis_entries.extend((t.transaction_id, t.sender_account, t.receiver_account) for t in txns)

        decision_ids = select(models.FraudDecision.decision_id).where(models.FraudDecision.transaction_id.in_(found))
        totals["deleted_alerts"] += (
            db.query(models.Alert).filter(models.Alert.decision_id.in_(decision_ids)).delete(synchronize_session=False)
        )
        totals["deleted_decisions"] += (
            db.query(models.FraudDecision)
            .filter(models.FraudDecision.transaction_id.in_(found))
            .delete(synchronize_session=False)
        )
        totals["deleted_transactions"] += (
            db.query(models.Transaction)
            .filter(models.Transaction.transaction_id.in_(found))
            .delete(synchronize_session=False)
        )

    db.commit()

    redis_client.forget_transactions(redis_entries)
    for sender, txn_type, amount, decided_at in daily_refunds:
        when = decided_at if decided_at.tzinfo else decided_at.replace(tzinfo=timezone.utc)
        redis_client.forget_daily_amount(sender, txn_type, amount, when)

    return totals


def purge(db: Session, source: str, before: Optional[datetime] = None) -> dict:
    """Bulk version of delete_transactions() for the Data Cleanup page."""
    ids = [row.transaction_id for row in _matching_query(db, source, before).all()]
    return delete_transactions(db, ids)
