"""
Redis-backed failed-login lockout, keyed by "username+ip" so a brute force
against one account from one source gets slowed down without needing a
dedicated rate-limiting service.

Same graceful-degradation policy as redis_client.py: a Redis outage must not
lock everyone out or crash login, so every call fails open (treated as "not
locked out" / "attempt not recorded") on RedisError.
"""

import logging

import redis

from config import settings
from redis_client import get_client

logger = logging.getLogger("rate_limit")


def _key(username: str, ip: str) -> str:
    return f"login_attempts:{username}:{ip}"


def is_locked_out(username: str, ip: str) -> bool:
    try:
        attempts = get_client().get(_key(username, ip))
        return attempts is not None and int(attempts) >= settings.LOGIN_MAX_ATTEMPTS
    except redis.RedisError as e:
        logger.warning("Rate-limit check failed, failing open: %s", e)
        return False


def record_failed_attempt(username: str, ip: str) -> None:
    try:
        client = get_client()
        key = _key(username, ip)
        pipe = client.pipeline()
        pipe.incr(key)
        pipe.expire(key, int(settings.LOGIN_LOCKOUT_MINUTES * 60))
        pipe.execute()
    except redis.RedisError as e:
        logger.warning("Failed-attempt recording failed: %s", e)


def clear_attempts(username: str, ip: str) -> None:
    try:
        get_client().delete(_key(username, ip))
    except redis.RedisError as e:
        logger.warning("Clearing login attempts failed: %s", e)
