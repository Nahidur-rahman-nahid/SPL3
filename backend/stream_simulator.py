"""
Simulates a live transaction stream so the analyst dashboard has something
continuously flowing through it — standing in for the Kafka producer/
consumer architecture.md originally specified (see backend/README.md for
why: no Docker, no extra hosted-service signup, and this gets the actual
analyst-facing outcome — continuous real-time scoring — without it).

Swapping in real Kafka later is a drop-in replacement: point run_stream()'s
"where transactions come from" at a Kafka consumer instead of
generate_transaction() below. process_transaction() (scoring_pipeline.py)
doesn't change at all — it's already fully decoupled from how a transaction
arrived.

Reuses a small pool of RECURRING accounts (not a fresh random ID every
tick) so Redis's neighbour memory actually builds up realistically as the
stream runs, the same way real accounts transact repeatedly over time.
"""

import asyncio
import logging
import random
import time

from config import settings
from database import SessionLocal
from schemas import ScoreRequest, TransactionFeatures
from scoring_pipeline import process_transaction

logger = logging.getLogger("stream_simulator")

_enabled = settings.STREAM_ENABLED
_counter = 0

NORMAL_PAIRS = [
    ("C_CUSTOMER_100", "M_MERCHANT_200"),
    ("C_CUSTOMER_101", "M_MERCHANT_201"),
    ("C_CUSTOMER_102", "M_MERCHANT_202"),
    ("C_CUSTOMER_103", "M_MERCHANT_200"),
]

SUSPICIOUS_PAIRS = [
    ("C_VICTIM_301", "C_MULE_402"),
    ("C_VICTIM_305", "C_MULE_402"),
]

# Several different senders converging on one receiver -- the account-level
# shape of a ring, even though ring detection itself isn't built yet. This
# gives that future feature real accumulated Redis history to work with.
RING_SENDERS = ["C_VICTIM_501", "C_VICTIM_502", "C_VICTIM_503", "C_VICTIM_504"]
RING_RECEIVER = "C_MULE_602"


def _next_id() -> str:
    global _counter
    _counter += 1
    return f"stream-{int(time.time() * 1000)}-{_counter}"


def _normal_transaction() -> ScoreRequest:
    sender, receiver = random.choice(NORMAL_PAIRS)
    amount = round(random.uniform(500, 15000), 2)
    old_bal_sender = amount + random.uniform(5000, 30000)
    old_bal_receiver = random.uniform(1000, 20000)
    return ScoreRequest(
        transaction_id=_next_id(),
        sender_account=sender,
        receiver_account=receiver,
        features=TransactionFeatures(
            step=random.randint(1, 720),
            type=random.choice(["TRANSFER", "CASH_OUT"]),
            amount=amount,
            oldbalanceOrg=old_bal_sender,
            newbalanceOrig=old_bal_sender - amount,
            oldbalanceDest=old_bal_receiver,
            newbalanceDest=old_bal_receiver + amount,
        ),
        neighbours=[],
    )


def _drain_transaction(sender: str, receiver: str) -> ScoreRequest:
    """Full-balance-drain pattern (PaySim's actual fraud signature) —
    shared by both the suspicious-pair and ring generators below."""
    amount = round(random.uniform(80000, 250000), 2)
    return ScoreRequest(
        transaction_id=_next_id(),
        sender_account=sender,
        receiver_account=receiver,
        features=TransactionFeatures(
            step=random.randint(1, 720),
            type="TRANSFER",
            amount=amount,
            oldbalanceOrg=amount,
            newbalanceOrig=0,
            oldbalanceDest=0,
            newbalanceDest=amount,
        ),
        neighbours=[],
    )


def _suspicious_transaction() -> ScoreRequest:
    sender, receiver = random.choice(SUSPICIOUS_PAIRS)
    return _drain_transaction(sender, receiver)


def _ring_transaction() -> ScoreRequest:
    sender = random.choice(RING_SENDERS)
    return _drain_transaction(sender, RING_RECEIVER)


def generate_transaction() -> ScoreRequest:
    """Weighted random pick — mostly normal traffic, occasional suspicious
    activity, rare ring-building events. Roughly mirrors PaySim's own
    heavily-imbalanced fraud rate rather than flooding the demo with alerts."""
    roll = random.random()
    if roll < 0.80:
        return _normal_transaction()
    if roll < 0.93:
        return _suspicious_transaction()
    return _ring_transaction()


def is_enabled() -> bool:
    return _enabled


def set_enabled(value: bool) -> None:
    global _enabled
    _enabled = value


async def run_stream():
    logger.info("Transaction stream simulator started (interval=%.1fs)", settings.STREAM_INTERVAL_SECONDS)
    while True:
        await asyncio.sleep(settings.STREAM_INTERVAL_SECONDS)
        if not _enabled:
            continue
        db = SessionLocal()
        try:
            request = generate_transaction()
            await process_transaction(request, db, source="stream")
        except Exception as e:
            logger.warning("Stream simulator skipped one transaction due to an error: %s", e)
        finally:
            db.close()
