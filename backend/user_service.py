"""DB-backed user CRUD — the real replacement for auth.py's old in-memory
_USERS dict. No public signup: accounts are only created by an existing
ADMIN via create_user (see main.py's POST /auth/users), plus the one-time
bootstrap_admin seed below for the very first account."""

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

import models
from auth import generate_temp_password, hash_password, verify_password
from config import settings

logger = logging.getLogger("user_service")


def get_user_by_username(db: Session, username: str) -> models.User | None:
    return db.query(models.User).filter(models.User.username == username).first()


def get_user_by_id(db: Session, user_id: int) -> models.User | None:
    return db.get(models.User, user_id)


def list_users(db: Session) -> list[models.User]:
    return db.query(models.User).order_by(models.User.created_at).all()


def create_user(db: Session, username: str, email: str | None, role: str, created_by_id: int) -> tuple[models.User, str]:
    temp_password = generate_temp_password()
    user = models.User(
        username=username,
        email=email,
        hashed_password=hash_password(temp_password),
        role=role,
        is_active=True,
        must_change_password=True,
        created_by_id=created_by_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user, temp_password


def update_user(db: Session, user: models.User, role: str | None = None, is_active: bool | None = None) -> models.User:
    if role is not None:
        user.role = role
    if is_active is not None:
        user.is_active = is_active
    db.commit()
    db.refresh(user)
    return user


def reset_password(db: Session, user: models.User) -> str:
    temp_password = generate_temp_password()
    user.hashed_password = hash_password(temp_password)
    user.must_change_password = True
    db.commit()
    return temp_password


def change_password(db: Session, user: models.User, current_password: str, new_password: str) -> bool:
    if not verify_password(current_password, user.hashed_password):
        return False
    user.hashed_password = hash_password(new_password)
    user.must_change_password = False
    db.commit()
    return True


def authenticate(db: Session, username: str, password: str) -> models.User | None:
    user = get_user_by_username(db, username)
    if user is None or not user.is_active:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    return user


def bootstrap_admin(db: Session) -> None:
    """Seeds exactly one real admin account from env vars, only if the users
    table is empty. Every account after this one is created via the admin
    API, not env vars — this just breaks the chicken-and-egg problem of
    "who creates the first admin"."""
    if db.query(models.User).first() is not None:
        return

    password = settings.BOOTSTRAP_ADMIN_PASSWORD
    if password == "change-me-on-first-login":
        logger.warning(
            "Bootstrap admin '%s' is being created with the DEFAULT password. "
            "Set BOOTSTRAP_ADMIN_PASSWORD before deploying anywhere real, "
            "and change it immediately after first login.",
            settings.BOOTSTRAP_ADMIN_USERNAME,
        )

    admin = models.User(
        username=settings.BOOTSTRAP_ADMIN_USERNAME,
        email=settings.BOOTSTRAP_ADMIN_EMAIL,
        hashed_password=hash_password(password),
        role="ADMIN",
        is_active=True,
        must_change_password=True,
    )
    db.add(admin)
    db.commit()
    logger.info("Bootstrap admin account '%s' created.", settings.BOOTSTRAP_ADMIN_USERNAME)
