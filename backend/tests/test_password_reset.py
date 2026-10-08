"""Forgot password: emailed one-time reset link (password_reset.py,
email_service.py, POST /auth/forgot-password, POST /auth/reset-password).
Real auth, SQLite, fakeredis; emails are captured instead of sent."""

import logging
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

import auth
import database
import email_service
import main
import models
from config import settings

OLD_PW = "OldPass#123"
NEW_PW = "NewPass#456"


@pytest.fixture
def outbox(monkeypatch):
    sent = []
    monkeypatch.setattr(email_service, "send_email", lambda to, subject, text, html=None: sent.append(
        {"to": to, "subject": subject, "text": text, "html": html}))
    return sent


@pytest.fixture
def client(db, outbox):
    def override_db():
        yield db

    main.app.dependency_overrides[database.get_db] = override_db
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def make_user(db, username="rahim", email="rahim@example.com", active=True, role="ANALYST", password=OLD_PW):
    user = models.User(username=username, email=email, hashed_password=auth.hash_password(password),
                       role=role, is_active=active, must_change_password=False)
    db.add(user)
    db.commit()
    return user


def forgot(client, email):
    return client.post("/auth/forgot-password", json={"email": email})


def token_from(mail):
    return unquote(re.search(r"reset-password\?token=(\S+)", mail["text"]).group(1))


def reset(client, token, password=NEW_PW):
    return client.post("/auth/reset-password", json={"token": token, "new_password": password})


def login(username, password):
    return TestClient(main.app).post("/auth/login", json={"username": username, "password": password}).status_code


# PR-01
def test_known_email_gets_a_reset_link(client, db, outbox):
    make_user(db)
    res = forgot(client, "Rahim@Example.com ")  # case and spaces don't matter
    assert res.status_code == 200
    assert len(outbox) == 1 and outbox[0]["to"] == "rahim@example.com"
    assert f"{settings.FRONTEND_BASE_URL}/reset-password?token=" in outbox[0]["text"]
    assert "rahim" in outbox[0]["text"] and outbox[0]["html"]


# PR-02
def test_unknown_or_inactive_email_gets_the_same_reply_and_no_mail(client, db, outbox):
    make_user(db, "gone", "gone@example.com", active=False)
    known = make_user(db)
    replies = [forgot(client, e) for e in ("nobody@example.com", "gone@example.com", known.email)]
    assert {r.status_code for r in replies} == {200}
    assert replies[0].json() == replies[1].json() == replies[2].json()
    assert [m["to"] for m in outbox] == ["rahim@example.com"]


# PR-03
def test_only_the_token_hash_is_stored(client, db, outbox):
    make_user(db)
    forgot(client, "rahim@example.com")
    token = token_from(outbox[0])
    stored = db.query(models.PasswordResetToken).one()
    assert stored.token_hash != token and len(stored.token_hash) == 64


# PR-04
def test_reset_sets_new_password(client, db, outbox):
    make_user(db)
    forgot(client, "rahim@example.com")
    assert reset(client, token_from(outbox[0])).status_code == 200
    assert login("rahim", NEW_PW) == 200
    assert login("rahim", OLD_PW) == 401


# PR-05
def test_link_works_only_once(client, db, outbox):
    make_user(db)
    forgot(client, "rahim@example.com")
    token = token_from(outbox[0])
    assert reset(client, token).status_code == 200
    assert reset(client, token, "Another#789").status_code == 400
    assert login("rahim", NEW_PW) == 200


# PR-06
def test_expired_link_is_refused(client, db, outbox):
    make_user(db)
    forgot(client, "rahim@example.com")
    record = db.query(models.PasswordResetToken).one()
    record.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()
    assert reset(client, token_from(outbox[0])).status_code == 400
    assert login("rahim", OLD_PW) == 200


# PR-07
def test_new_request_voids_the_older_link(client, db, outbox):
    make_user(db)
    forgot(client, "rahim@example.com")
    forgot(client, "rahim@example.com")
    first, second = token_from(outbox[0]), token_from(outbox[1])
    assert reset(client, first).status_code == 400
    assert reset(client, second).status_code == 200


# PR-08
def test_invalid_token_and_short_password(client, db, outbox):
    make_user(db)
    assert reset(client, "made-up-token-value").status_code == 400
    forgot(client, "rahim@example.com")
    assert reset(client, token_from(outbox[0]), "short").status_code == 422


# PR-09
def test_reset_signs_out_every_session(client, db, outbox):
    make_user(db)
    session = TestClient(main.app)
    session.post("/auth/login", json={"username": "rahim", "password": OLD_PW})
    forgot(client, "rahim@example.com")
    reset(client, token_from(outbox[0]))
    headers = {"X-CSRF-Token": session.cookies.get("csrf_token")}
    assert session.post("/auth/refresh", headers=headers).status_code == 401


# PR-10
def test_reset_clears_forced_password_change(client, db, outbox):
    user = make_user(db)
    user.must_change_password = True
    db.commit()
    forgot(client, "rahim@example.com")
    reset(client, token_from(outbox[0]))
    db.refresh(user)
    assert user.must_change_password is False


# PR-11
def test_rate_limit_per_address(client, db, outbox):
    make_user(db)
    replies = [forgot(client, "rahim@example.com") for _ in range(settings.PASSWORD_RESET_MAX_REQUESTS + 2)]
    assert {r.status_code for r in replies} == {200}  # same reply, so the limit isn't visible either
    assert len(outbox) == settings.PASSWORD_RESET_MAX_REQUESTS


# PR-12
def test_redis_down_fails_open(client, db, outbox, redis_down):
    make_user(db)
    forgot(client, "rahim@example.com")
    assert len(outbox) == 1


# PR-13
def test_admin_sets_and_clears_user_email(client, db):
    make_user(db, "boss", "boss@example.com", role="ADMIN")
    target = make_user(db, "karim", None)
    client.post("/auth/login", json={"username": "boss", "password": OLD_PW})
    headers = {"X-CSRF-Token": client.cookies.get("csrf_token")}
    res = client.patch(f"/auth/users/{target.id}", headers=headers, json={"email": "karim@example.com"})
    assert res.status_code == 200 and res.json()["email"] == "karim@example.com"
    assert client.patch(f"/auth/users/{target.id}", headers=headers, json={"email": "not-an-email"}).status_code == 422
    assert client.patch(f"/auth/users/{target.id}", headers=headers, json={"email": ""}).json()["email"] is None


# PR-14
def test_without_smtp_the_mail_goes_to_the_log(monkeypatch, caplog):
    monkeypatch.setattr(settings, "SMTP_HOST", "")
    with caplog.at_level(logging.WARNING, logger="email_service"):
        email_service.send_email("x@example.com", "Subject", "the link")
    assert "the link" in caplog.text and "not sent" in caplog.text
