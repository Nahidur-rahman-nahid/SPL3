"""
Cookie-based session auth backed by the users/refresh_tokens tables (see
models.py). Two tokens are issued at login:

  - access_token: a short-lived (ACCESS_TOKEN_EXPIRE_MINUTES) JWT, carried in
    an httpOnly cookie, verified statelessly (no DB hit needed to check its
    signature/expiry) but re-checked against the DB in get_current_user so a
    deactivated account is locked out within one access-token lifetime.
  - refresh_token: a long-lived (REFRESH_TOKEN_EXPIRE_DAYS), opaque random
    string, also httpOnly-cookied. Only its SHA-256 hash is stored
    (models.RefreshToken.token_hash) — same principle as password hashing,
    though a fast hash suffices here since the raw token already has 256
    bits of entropy (unlike a human-chosen password). Rotated on every use;
    presenting an already-rotated/unknown token revokes its whole family
    (see rotate_refresh_token) as a theft signal.

A third, non-httpOnly csrf_token cookie is set alongside these so the
frontend can echo it back as an X-CSRF-Token header on mutating requests
(double-submit CSRF defense — needed once a cross-domain prod deploy forces
SameSite=None, at which point SameSite alone no longer blocks CSRF).
"""

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import Cookie, Depends, HTTPException, Request, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

import models
from config import settings
from database import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ACCESS_TOKEN_COOKIE = "access_token"
REFRESH_TOKEN_COOKIE = "refresh_token"
CSRF_COOKIE = "csrf_token"


# ---------------------------------------------------------------- passwords

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)


def generate_temp_password() -> str:
    """Human-typeable one-time password for admin-created accounts / resets
    (decision: admin-driven reset, no email flow to deliver a link)."""
    return secrets.token_urlsafe(9)


# --------------------------------------------------------- access token JWT

def create_access_token(user: models.User) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user.id), "role": user.role, "type": "access", "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None
    if payload.get("type") != "access":
        return None
    return payload


def _load_user_from_access_token(token: str, db: Session) -> models.User | None:
    payload = decode_access_token(token)
    if payload is None:
        return None
    user_id = payload.get("sub")
    if user_id is None:
        return None
    user = db.get(models.User, int(user_id))
    if user is None or not user.is_active:
        return None
    return user


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    access_token: str | None = Cookie(default=None),
) -> models.User:
    if access_token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user = _load_user_from_access_token(access_token, db)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    return user


def require_role(*roles: str):
    def dependency(user: models.User = Depends(get_current_user)) -> models.User:
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return dependency


# ------------------------------------------------------- WebSocket variant
# /ws/alerts can't use FastAPI's Depends()-based Cookie extraction the same
# way a normal HTTP route can, so this reads straight from the handshake's
# cookie jar (websocket.cookies), which the browser populates the same way
# it does for a plain fetch — see main.py's ws_alerts for the Origin check
# that has to accompany this (CORSMiddleware doesn't cover WebSocket routes).
def load_user_from_cookies(cookies: dict, db: Session) -> models.User | None:
    token = cookies.get(ACCESS_TOKEN_COOKIE)
    if token is None:
        return None
    return _load_user_from_access_token(token, db)


# ------------------------------------------------------------ refresh token

def _hash_refresh_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode()).hexdigest()


def issue_refresh_token(db: Session, user: models.User, family_id: str | None = None) -> tuple[str, models.RefreshToken]:
    """family_id is omitted on login (starts a new family) and passed through
    on rotation (stays in the same family so reuse detection can bulk-revoke
    it later)."""
    raw_token = secrets.token_urlsafe(48)
    record = models.RefreshToken(
        user_id=user.id,
        token_hash=_hash_refresh_token(raw_token),
        family_id=family_id or str(uuid.uuid4()),
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return raw_token, record


class RefreshTokenReuseError(Exception):
    """Raised when an already-rotated or unknown refresh token is presented
    — treated as a theft signal; the caller should force a re-login."""


def rotate_refresh_token(db: Session, raw_token: str) -> tuple[str, models.User]:
    token_hash = _hash_refresh_token(raw_token)
    record = db.query(models.RefreshToken).filter(models.RefreshToken.token_hash == token_hash).first()

    if record is None:
        raise RefreshTokenReuseError("Unknown refresh token")

    now = datetime.now(timezone.utc)
    expires_at = record.expires_at if record.expires_at.tzinfo else record.expires_at.replace(tzinfo=timezone.utc)

    if record.revoked_at is not None:
        # Already-used token presented again: revoke the whole family.
        _revoke_family(db, record.family_id)
        raise RefreshTokenReuseError("Refresh token reuse detected")

    if expires_at < now:
        raise RefreshTokenReuseError("Refresh token expired")

    user = db.get(models.User, record.user_id)
    if user is None or not user.is_active:
        raise RefreshTokenReuseError("User no longer active")

    record.revoked_at = now
    db.commit()

    new_raw_token, _ = issue_refresh_token(db, user, family_id=record.family_id)
    return new_raw_token, user


def _revoke_family(db: Session, family_id: str) -> None:
    db.query(models.RefreshToken).filter(
        models.RefreshToken.family_id == family_id,
        models.RefreshToken.revoked_at.is_(None),
    ).update({"revoked_at": datetime.now(timezone.utc)})
    db.commit()


def revoke_refresh_token(db: Session, raw_token: str) -> None:
    token_hash = _hash_refresh_token(raw_token)
    record = db.query(models.RefreshToken).filter(models.RefreshToken.token_hash == token_hash).first()
    if record is not None:
        _revoke_family(db, record.family_id)


# ------------------------------------------------------------------- CSRF

def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def verify_csrf(csrf_cookie: str | None, csrf_header: str | None) -> bool:
    if not csrf_cookie or not csrf_header:
        return False
    return secrets.compare_digest(csrf_cookie, csrf_header)


# ----------------------------------------------------------- cookie helpers

def set_session_cookies(response, access_token: str, refresh_token: str, csrf_token: str) -> None:
    common = dict(
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        path="/",
    )
    response.set_cookie(
        ACCESS_TOKEN_COOKIE, access_token,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60, **common,
    )
    response.set_cookie(
        REFRESH_TOKEN_COOKIE, refresh_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60, **common,
    )
    # Not httpOnly — the frontend must be able to read this one to echo it
    # back as a header.
    response.set_cookie(
        CSRF_COOKIE, csrf_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        httponly=False,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        path="/",
    )


def clear_session_cookies(response) -> None:
    for name in (ACCESS_TOKEN_COOKIE, REFRESH_TOKEN_COOKIE, CSRF_COOKIE):
        response.delete_cookie(name, path="/")


def require_csrf(request: Request, csrf_token: str | None = Cookie(default=None)) -> None:
    header_token = request.headers.get("x-csrf-token")
    if not verify_csrf(csrf_token, header_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF check failed")
