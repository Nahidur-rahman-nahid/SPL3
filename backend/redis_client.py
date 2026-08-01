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
