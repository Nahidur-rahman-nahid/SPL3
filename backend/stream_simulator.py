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
fraud group uses fresh accounts so it only ever links to itself. A few
grey-zone pairs (see _queue_ambiguous) give the REVIEW queue cases the model
is honestly unsure about.
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


# PaySim's `step` (simulation hour, 1-743) is a raw model input, and the
# model's grey-zone scores shift with it: at step 100 the ambiguous pairs all
# BLOCK, at 300 drains drop to REVIEW. A fixed step keeps the stream's
# behaviour the same at any time of day; 200 is the best balance measured
# (and the step pattern_demo.py uses).
STREAM_STEP = 200


def _current_step() -> int:
    return STREAM_STEP


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


def _queue_ambiguous() -> None:
    """A pair of transactions the model is genuinely unsure about (it scores
    the second one mostly 40-75%, i.e. REVIEW), so the analyst queue gets
    real grey-zone cases, not only clear-cut fraud:
      - pooled money: two people each send almost their whole balance to the
        same small account (family pooling for a purchase, or a new mule?);
      - two new payees: someone sends about a quarter of their balance to
        each of two brand-new accounts (paying new suppliers, or a takeover?)."""
    if random.random() < 0.5:
        receiver = _fresh("C_POOL")
        _balance[receiver] = round(random.uniform(0, 20_000), 2)
        for _ in range(2):
            sender = _fresh("C_PAYER")
            _balance[sender] = round(random.uniform(40_000, 200_000), 2)
            _pending.append((sender, receiver, _balance[sender] * random.uniform(0.85, 0.95)))
    else:
        sender = _fresh("C_PAYER")
        balance = _balance[sender] = round(random.uniform(80_000, 300_000), 2)
        for _ in range(2):
            receiver = _fresh("C_NEWPAYEE")
            _balance[receiver] = 0.0
            amount = balance * random.uniform(0.28, 0.34)
            _pending.append((sender, receiver, amount))
            balance -= amount


def generate_transaction() -> ScoreRequest:
    """Mostly everyday traffic. Fraud (drains, structuring bursts) and
    grey-zone pairs arrive as groups sent over the following ticks. Measured
    over 800 transactions with the retrained model: 5% fraud, 13% grey-zone;
    decisions 90% ALLOW, 5% REVIEW, 5% BLOCK."""
    if _pending:
        sender, receiver, amount = _pending.popleft()
        return _send(sender, receiver, amount)
    roll = random.random()
    if roll < 0.87:
        return _normal_transaction()
    if roll < 0.91:
        return _drain_transaction()
    if roll < 0.92:
        _queue_structuring()
    else:
        _queue_ambiguous()
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
