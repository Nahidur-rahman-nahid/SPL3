"""
Minimal JWT auth — two roles (ADMIN, ANALYST), credentials from config (env
vars in production). No user registration/CRUD; that's out of scope for a
demo with two known roles. Swap in a real users table later if needed.
"""

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

_USERS = {
    settings.ADMIN_USERNAME: {
        "username": settings.ADMIN_USERNAME,
        "hashed_password": pwd_context.hash(settings.ADMIN_PASSWORD),
        "role": "ADMIN",
    },
    settings.ANALYST_USERNAME: {
        "username": settings.ANALYST_USERNAME,
        "hashed_password": pwd_context.hash(settings.ANALYST_PASSWORD),
        "role": "ANALYST",
    },
}


def authenticate_user(username: str, password: str):
    user = _USERS.get(username)
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_user_from_token(token: str) -> dict | None:
    """Shared by the HTTP dependency below and the WebSocket endpoint (which
    can't use FastAPI's Depends() the same way — browsers can't set custom
    headers on a WebSocket, so the token arrives as a query param there
    instead of an Authorization header)."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username = payload.get("sub")
    except JWTError:
        return None
    if username is None or username not in _USERS:
        return None
    return _USERS[username]


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    user = decode_user_from_token(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
