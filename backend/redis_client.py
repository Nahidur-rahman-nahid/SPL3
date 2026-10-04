"""
Redis-backed live graph memory (architecture.md section 5): recent
transactions per account, keyed as "account:{account_id}", TTL 86400s (24h).
On /score, we look up the sender's and receiver's recent transactions to
assemble the real-time subgraph, then register the new transaction under
both accounts for future lookups.

Failures here degrade gracefully — a Redis hiccup should reduce the quality
of a score (fewer neighbours found), not take the whole API down. Every
call is wrapped so a connection error just logs and returns an empty/no-op
result instead of raising.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import redis

from config import settings

logger = logging.getLogger("redis_client")

TTL_SECONDS = 86400
MAX_PER_ACCOUNT = 50  # cap list length so a very active account can't grow unbounded

_client = None


def get_client():
    global _client
    if _client is None:
        _client = redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=2)
    return _client


def _key(account_id: str) -> str:
    return f"account:{account_id}"


def get_recent_transactions(account_id: str) -> list[dict]:
    try:
        raw_entries = get_client().lrange(_key(account_id), 0, MAX_PER_ACCOUNT - 1)
        return [json.loads(e) for e in raw_entries]
    except redis.RedisError as e:
        logger.warning("Redis lookup failed for %s: %s", account_id, e)
        return []


def store_transaction(transaction_id: str, sender_account: str, receiver_account: str, features: dict):
    entry = json.dumps({"transaction_id": transaction_id, **features})
    try:
        client = get_client()
        pipe = client.pipeline()
        for account_id in (sender_account, receiver_account):
            key = _key(account_id)
            pipe.lpush(key, entry)
            pipe.ltrim(key, 0, MAX_PER_ACCOUNT - 1)
            pipe.expire(key, TTL_SECONDS)
        pipe.execute()
    except redis.RedisError as e:
        logger.warning("Redis store failed for transaction %s: %s", transaction_id, e)


# ---------------------------------------------------------------- daily limits
#
# Per-account, per-type running totals for the current Bangladesh calendar
# day, keyed "daily:{account_id}:{TYPE}:{YYYYMMDD}". The date in the key is
# what resets the total at local midnight; the TTL only cleans up old keys,
# so it just has to outlive its day. Same fail-open policy as above: every
# function returns None on a Redis error so the caller can skip the check.

DAILY_TTL_SECONDS = 2 * 86400


def _local_date(when: Optional[datetime] = None) -> str:
    local_tz = timezone(timedelta(hours=settings.LOCAL_UTC_OFFSET_HOURS))
    when = when or datetime.now(timezone.utc)
    return when.astimezone(local_tz).strftime("%Y%m%d")


def _daily_key(account_id: str, txn_type: str, when: Optional[datetime] = None) -> str:
    return f"daily:{account_id}:{txn_type}:{_local_date(when)}"


def get_daily_total(account_id: str, txn_type: str, when: Optional[datetime] = None) -> Optional[float]:
    try:
        raw = get_client().get(_daily_key(account_id, txn_type, when))
        return round(float(raw), 2) if raw is not None else 0.0
    except redis.RedisError as e:
        logger.warning("Daily total lookup failed for %s: %s", account_id, e)
        return None


def reserve_daily_amount(
    account_id: str, txn_type: str, amount: float, limit: float, when: Optional[datetime] = None
) -> Optional[tuple]:
    """Adds `amount` to today's total and keeps it only if the new total
    stays within `limit`. Add-then-check (rather than read-then-add) means
    two concurrent requests can't both pass on the same remaining headroom.
    Returns (accepted, total_after) — total_after excludes `amount` when
    rejected — or None if Redis is unavailable."""
    key = _daily_key(account_id, txn_type, when)
    try:
        client = get_client()
        pipe = client.pipeline()
        pipe.incrbyfloat(key, amount)
        pipe.expire(key, DAILY_TTL_SECONDS)
        new_total = round(float(pipe.execute()[0]), 2)
        if new_total > limit:
            client.incrbyfloat(key, -amount)
            return False, round(new_total - amount, 2)
        return True, new_total
    except redis.RedisError as e:
        logger.warning("Daily total reserve failed for %s: %s", account_id, e)
        return None


def release_daily_amount(account_id: str, txn_type: str, amount: float, when: Optional[datetime] = None) -> None:
    """Undoes a reserve_daily_amount() — used when the transaction ends up
    BLOCKed (or scoring fails), so blocked money doesn't eat into the limit."""
    try:
        get_client().incrbyfloat(_daily_key(account_id, txn_type, when), -amount)
    except redis.RedisError as e:
        logger.warning("Daily total release failed for %s: %s", account_id, e)
