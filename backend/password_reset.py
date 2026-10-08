"""Forgot-password flow: email a one-time reset link, then set a new password
with it.

  - The answer to "send me a link" never reveals whether the address
    belongs to an account (main.py always replies the same way).
  - Only the SHA-256 of the token is stored, like refresh tokens; the raw
    token exists only in the emailed link.
  - A link works once, for PASSWORD_RESET_TOKEN_MINUTES; asking again
    voids the older links of that account.
  - Using it signs the account out everywhere (all refresh tokens revoked).
  - At most PASSWORD_RESET_MAX_REQUESTS emails per address per window
    (Redis; fails open like the login limit).
"""

import hashlib
import html as html_lib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import redis
from sqlalchemy import func
from sqlalchemy.orm import Session

import models
from auth import hash_password
from config import settings
from redis_client import get_client

logger = logging.getLogger("password_reset")


def _hash(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode()).hexdigest()


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _within_rate_limit(email: str) -> bool:
    key = f"password_reset:{email}"
    try:
        client = get_client()
        count = client.incr(key)
        if count == 1:  # first request of the window starts the clock
            client.expire(key, int(settings.PASSWORD_RESET_WINDOW_MINUTES * 60))
        return count <= settings.PASSWORD_RESET_MAX_REQUESTS
    except redis.RedisError as e:
        logger.warning("Password-reset rate limit check failed, failing open: %s", e)
        return True


def reset_email(user: models.User, link: str) -> tuple[str, str, str]:
    minutes = settings.PASSWORD_RESET_TOKEN_MINUTES
    subject = "Reset your FraudScope password"
    text = (
        f"Hello {user.username},\n\n"
        f"Someone asked to reset the password of your FraudScope account. "
        f"Open this link to choose a new one (valid for {minutes} minutes, one use only):\n\n"
        f"{link}\n\n"
        "If you did not ask for this, ignore this email; your password stays the same."
    )
    html = (
        f"<p>Hello {html_lib.escape(user.username)},</p>"
        "<p>Someone asked to reset the password of your FraudScope account.</p>"
        f'<p><a href="{link}" style="display:inline-block;padding:10px 18px;background:#4f46e5;'
        'color:#ffffff;text-decoration:none;border-radius:6px">Choose a new password</a></p>'
        f"<p>The link is valid for {minutes} minutes and works once.</p>"
        "<p>If you did not ask for this, ignore this email; your password stays the same.</p>"
    )
    return subject, text, html


def request_reset(db: Session, email: str) -> list[tuple[str, str, str, str]]:
    """Creates a reset link for every active account with this email and
    returns the emails to send as (to, subject, text, html). Returns an
    empty list for an unknown address or when rate-limited."""
    email = email.strip().lower()
    if not email or not _within_rate_limit(email):
        return []

    users = (
        db.query(models.User)
        .filter(func.lower(models.User.email) == email, models.User.is_active.is_(True))
        .all()
    )
    now = datetime.now(timezone.utc)
    messages = []
    for user in users:
        # A new link voids the older, unused ones.
        db.query(models.PasswordResetToken).filter(
            models.PasswordResetToken.user_id == user.id,
            models.PasswordResetToken.used_at.is_(None),
        ).update({"used_at": now})

        raw_token = secrets.token_urlsafe(32)
        db.add(models.PasswordResetToken(
            user_id=user.id,
            token_hash=_hash(raw_token),
            expires_at=now + timedelta(minutes=settings.PASSWORD_RESET_TOKEN_MINUTES),
        ))
        link = f"{settings.FRONTEND_BASE_URL.rstrip('/')}/reset-password?token={quote(raw_token)}"
        messages.append((user.email, *reset_email(user, link)))
    db.commit()
    return messages


def reset_password(db: Session, raw_token: str, new_password: str) -> bool:
    """Sets the new password if the token is valid, unused and not expired.
    Signs the account out of every session."""
    record = (
        db.query(models.PasswordResetToken)
        .filter(models.PasswordResetToken.token_hash == _hash(raw_token))
        .first()
    )
    now = datetime.now(timezone.utc)
    if record is None or record.used_at is not None or _as_utc(record.expires_at) < now:
        return False
    user = db.get(models.User, record.user_id)
    if user is None or not user.is_active:
        return False

    user.hashed_password = hash_password(new_password)
    user.must_change_password = False
    record.used_at = now
    db.query(models.RefreshToken).filter(
        models.RefreshToken.user_id == user.id,
        models.RefreshToken.revoked_at.is_(None),
    ).update({"revoked_at": now})
    db.commit()
    return True
