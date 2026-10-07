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

Everyday traffic comes from a pool of RECURRING customers whose balances
carry over from one transaction to the next, paying established merchants,
sending small amounts to relatives and cashing out at agents — so Redis's
neighbour memory builds up the way real accounts do. Fraud arrives in
groups, the way the GNN learned it (00_inject_fraud_patterns.py): drains
into a shared mule, and structuring bursts of near-limit transfers. Each
fraud group uses fresh accounts so it only ever links to itself.
"""

import asyncio
import logging
import random
import time
from collections import deque

from config import settings
from database import SessionLocal
from schemas import ScoreRequest, TransactionFeatures
from scoring_pipeline import process_transaction

logger = logging.getLogger("stream_simulator")

_enabled = settings.STREAM_ENABLED
_counter = 0

CUSTOMERS = [f"C_CUST_{i:03d}" for i in range(1, 61)]
MERCHANTS = [f"M_SHOP_{i:02d}" for i in range(1, 13)]
RELATIVES = [f"C_FAMILY_{i:02d}" for i in range(1, 21)]
AGENTS = [f"A_AGENT_{i:02d}" for i in range(1, 7)]

_balance: dict = {}
# Transactions of a fraud group still to be sent, one per tick.
_pending: deque = deque()
_ring = {"mule": None, "left": 0}


def _next_id() -> str:
    global _counter
    _counter += 1
    return f"stream-{int(time.time() * 1000)}-{_counter}"


def _current_step() -> int:
    """PaySim's `step` is the hour (1-743); use the current hour of a
    30-day cycle so transactions close in time also share a step."""
    return int(time.time() // 3600) % 720 + 1


def _opening_balance(account: str) -> float:
    if account.startswith("M_"):
        return random.uniform(200_000, 3_000_000)
    if account.startswith("A_"):
        return random.uniform(500_000, 5_000_000)
    if account.startswith("C_FAMILY"):
        return random.uniform(0, 15_000)
    return random.uniform(20_000, 250_000)


def _bal(account: str) -> float:
    if account not in _balance:
        _balance[account] = round(_opening_balance(account), 2)
    return _balance[account]


def _send(sender: str, receiver: str, amount: float, txn_type: str = "TRANSFER") -> ScoreRequest:
    """Moves money between two accounts, so balances chain from one
    transaction to the next like a real ledger."""
    old_org, old_dest = _bal(sender), _bal(receiver)
    amount = round(min(amount, old_org), 2)
    _balance[sender], _balance[receiver] = round(old_org - amount, 2), round(old_dest + amount, 2)
    return ScoreRequest(
        transaction_id=_next_id(),
        sender_account=sender,
        receiver_account=receiver,
        features=TransactionFeatures(
            step=_current_step(),
            type=txn_type,
            amount=amount,
            oldbalanceOrg=old_org,
            newbalanceOrig=_balance[sender],
            oldbalanceDest=old_dest,
            newbalanceDest=_balance[receiver],
        ),
        neighbours=[],
    )


def _customer() -> str:
    customer = random.choice(CUSTOMERS)
    if _bal(customer) < 5_000:  # salary day: keep everyday accounts funded
        _balance[customer] += random.uniform(30_000, 150_000)
    return customer


def _normal_transaction() -> ScoreRequest:
    customer = _customer()
    balance = _bal(customer)
    roll = random.random()
    if roll < 0.55:  # shop / merchant payment
        return _send(customer, random.choice(MERCHANTS), min(random.uniform(300, 12_000), balance * 0.10))
    if roll < 0.80:  # money for the same relative each time, as in real families
        relative = RELATIVES[CUSTOMERS.index(customer) % len(RELATIVES)]
        return _send(customer, relative, min(random.uniform(500, 10_000), balance * 0.15))
    return _send(customer, random.choice(AGENTS), balance * random.uniform(0.05, 0.30), "CASH_OUT")


def _fresh(prefix: str) -> str:
    return f"{prefix}_{_counter + 1}_{random.randint(1000, 9999)}"


def _drain_transaction() -> ScoreRequest:
    """A victim's whole balance moved into a mule shared by several victims
    (the mule hub / ring pattern); every few victims a new mule takes over."""
    if _ring["left"] <= 0:
        _ring["mule"], _ring["left"] = _fresh("C_MULE"), random.randint(4, 8)
        _balance[_ring["mule"]] = 0.0
    _ring["left"] -= 1
    victim = _fresh("C_VICTIM")
    _balance[victim] = round(random.uniform(40_000, 250_000), 2)
    return _send(victim, _ring["mule"], _balance[victim])


def _queue_structuring() -> None:
    """One sender splits a large sum into 3-5 transfers just under the BB
    50,000 limit to the same receiver, sent on consecutive ticks."""
    sender, receiver = _fresh("C_SPLIT"), _fresh("C_RECV")
    amounts = [random.uniform(45_000, 49_900) for _ in range(random.randint(3, 5))]
    _balance[sender] = round(sum(amounts) / random.uniform(0.45, 0.85), 2)
    _balance[receiver] = 0.0 if random.random() < 0.5 else round(random.uniform(5_000, 100_000), 2)
    _pending.extend((sender, receiver, amount) for amount in amounts)


def generate_transaction() -> ScoreRequest:
    """Mostly everyday traffic; fraud shows up as groups (a structuring
    burst is sent over the following ticks, interleaved with nothing else)."""
    if _pending:
        sender, receiver, amount = _pending.popleft()
        return _send(sender, receiver, amount)
    roll = random.random()
    if roll < 0.85:
        return _normal_transaction()
    if roll < 0.96:
        return _drain_transaction()
    _queue_structuring()
    return generate_transaction()


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
