"""REST + WebSocket API over HTTP with the REAL authentication: cookie
sessions, CSRF header, refresh rotation, roles and the login rate limit are
not overridden here (unlike the other API test files). SQLite DB, fakeredis,
fake model."""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import auth
import database
import main
import models
from config import settings

ADMIN_PW = "AdminPass#123"
ANALYST_PW = "AnalystPass#123"
ORIGIN = settings.cors_origins_list[0]

TXN = {
    "transaction_id": "api-1",
    "sender_account": "C_API_S",
    "receiver_account": "C_API_R",
    "features": {
        "step": 1, "type": "TRANSFER", "amount": 5000, "oldbalanceOrg": 25000,
        "newbalanceOrig": 20000, "oldbalanceDest": 10000, "newbalanceDest": 15000,
    },
    "transaction_time": "2026-10-04T12:00:00",
}


def make_user(db, username, password, role="ANALYST", active=True, must_change=False):
    user = models.User(
        username=username, hashed_password=auth.hash_password(password), role=role,
        is_active=active, must_change_password=must_change,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def client(db, model):
    def override_db():
        yield db

    main.app.dependency_overrides[database.get_db] = override_db
    make_user(db, "admin", ADMIN_PW, role="ADMIN")
    make_user(db, "analyst", ANALYST_PW)
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def login(client, username="admin", password=ADMIN_PW):
    return client.post("/auth/login", json={"username": username, "password": password})


def csrf(client):
    return {"X-CSRF-Token": client.cookies.get("csrf_token")}


def as_analyst(db):
    analyst = TestClient(main.app)
    assert login(analyst, "analyst", ANALYST_PW).status_code == 200
    return analyst


# --------------------------------------------------------------- health

# A01
def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


# --------------------------------------------------------- POST /auth/login

# A02
def test_login_sets_session_cookies(client):
    res = login(client)
    assert res.status_code == 200 and res.json()["username"] == "admin"
    cookies = res.headers.get_list("set-cookie")
    access = next(c for c in cookies if c.startswith("access_token="))
    csrf_cookie = next(c for c in cookies if c.startswith("csrf_token="))
    assert "httponly" in access.lower()
    assert any(c.startswith("refresh_token=") and "httponly" in c.lower() for c in cookies)
    assert "httponly" not in csrf_cookie.lower()  # the frontend must read it to echo it back


# A03
def test_wrong_password_and_unknown_user_look_the_same(client):
    wrong = login(client, "admin", "nope")
    unknown = login(client, "ghost", "nope")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


# A04
def test_deactivated_user_cannot_log_in(client, db):
    make_user(db, "gone", "GonePass#123", active=False)
    assert login(client, "gone", "GonePass#123").status_code == 401


# A05
def test_lockout_after_five_failures_even_with_right_password(client):
    for _ in range(5):
        assert login(client, "admin", "nope").status_code == 401
    assert login(client, "admin", ADMIN_PW).status_code == 429


# A06
def test_successful_login_resets_failure_count(client):
    for _ in range(4):
        login(client, "admin", "nope")
    assert login(client).status_code == 200
    codes = [login(client, "admin", "nope").status_code for _ in range(4)]
    assert codes == [401] * 4


# ------------------------------------------------------------ GET /auth/me

# A07
def test_me_requires_a_valid_session(client):
    assert client.get("/auth/me").status_code == 401
    client.cookies.set("access_token", "not-a-jwt")
    assert client.get("/auth/me").status_code == 401
    client.cookies.clear()
    login(client)
    me = client.get("/auth/me")
    assert me.status_code == 200 and me.json()["username"] == "admin" and me.json()["role"] == "ADMIN"


# -------------------------------------------------------- POST /auth/refresh

# A08
def test_refresh_rotates_the_refresh_token(client):
    login(client)
    old = client.cookies.get("refresh_token")
    res = client.post("/auth/refresh", headers=csrf(client))
    assert res.status_code == 200
    assert client.cookies.get("refresh_token") != old


# A09
def test_reused_refresh_token_revokes_the_whole_session(client):
    login(client)
    stolen = client.cookies.get("refresh_token")
    assert client.post("/auth/refresh", headers=csrf(client)).status_code == 200
    current = client.cookies.get("refresh_token")

    attacker = TestClient(main.app)
    attacker.cookies.set("refresh_token", stolen)
    attacker.cookies.set("csrf_token", "x")
    assert attacker.post("/auth/refresh", headers={"X-CSRF-Token": "x"}).status_code == 401
    # the legitimate session's newest token is revoked too
    client.cookies.set("refresh_token", current)
    assert client.post("/auth/refresh", headers=csrf(client)).status_code == 401


# A10
def test_refresh_needs_csrf_header(client):
    login(client)
    assert client.post("/auth/refresh").status_code == 403


# --------------------------------------------------------- POST /auth/logout

# A11
def test_logout_clears_session_and_revokes_refresh(client):
    login(client)
    refresh = client.cookies.get("refresh_token")
    assert client.post("/auth/logout", headers=csrf(client)).status_code == 200
    assert client.get("/auth/me").status_code == 401
    client.cookies.set("refresh_token", refresh)
    client.cookies.set("csrf_token", "x")
    assert client.post("/auth/refresh", headers={"X-CSRF-Token": "x"}).status_code == 401


# ------------------------------------------------- POST /auth/change-password

# A12
def test_change_password(client, db):
    make_user(db, "newbie", "TempPass#123", must_change=True)
    login(client, "newbie", "TempPass#123")
    bad = client.post("/auth/change-password", headers=csrf(client),
                      json={"current_password": "wrong", "new_password": "NewPass#456"})
    assert bad.status_code == 400
    ok = client.post("/auth/change-password", headers=csrf(client),
                     json={"current_password": "TempPass#123", "new_password": "NewPass#456"})
    assert ok.status_code == 200
    assert client.get("/auth/me").json()["must_change_password"] is False
    assert login(TestClient(main.app), "newbie", "NewPass#456").status_code == 200
    assert login(TestClient(main.app), "newbie", "TempPass#123").status_code == 401


# A13
def test_new_password_minimum_length(client):
    login(client)
    res = client.post("/auth/change-password", headers=csrf(client),
                      json={"current_password": ADMIN_PW, "new_password": "short"})
    assert res.status_code == 422


# ---------------------------------------------------- user management (ADMIN)

# A14
def test_admin_creates_user_with_temp_password(client):
    login(client)
    res = client.post("/auth/users", headers=csrf(client), json={"username": "ana2", "role": "ANALYST"})
    assert res.status_code == 200
    body = res.json()
    assert body["user"]["must_change_password"] is True and len(body["temp_password"]) >= 8
    assert login(TestClient(main.app), "ana2", body["temp_password"]).json()["must_change_password"] is True


# A15
def test_create_user_rejects_duplicates_and_bad_input(client):
    login(client)
    assert client.post("/auth/users", headers=csrf(client), json={"username": "analyst", "role": "ANALYST"}).status_code == 409
    assert client.post("/auth/users", headers=csrf(client), json={"username": "ab", "role": "ANALYST"}).status_code == 422
    assert client.post("/auth/users", headers=csrf(client), json={"username": "boss", "role": "ROOT"}).status_code == 422


# A16
def test_analyst_cannot_manage_users(client, db):
    analyst = as_analyst(db)
    assert analyst.get("/auth/users").status_code == 403
    assert analyst.post("/auth/users", headers=csrf(analyst), json={"username": "x123", "role": "ADMIN"}).status_code == 403


# A17
def test_admin_lists_users(client):
    login(client)
    names = {u["username"] for u in client.get("/auth/users").json()}
    assert {"admin", "analyst"} <= names


# A18
def test_deactivate_and_change_role(client, db):
    login(client)
    analyst_id = db.query(models.User).filter_by(username="analyst").one().id
    res = client.patch(f"/auth/users/{analyst_id}", headers=csrf(client), json={"is_active": False})
    assert res.status_code == 200 and res.json()["is_active"] is False
    assert login(TestClient(main.app), "analyst", ANALYST_PW).status_code == 401
    res = client.patch(f"/auth/users/{analyst_id}", headers=csrf(client), json={"is_active": True, "role": "ADMIN"})
    assert res.json()["role"] == "ADMIN"
    assert client.patch("/auth/users/9999", headers=csrf(client), json={"is_active": False}).status_code == 404


# A19
def test_admin_resets_password(client, db):
    login(client)
    analyst_id = db.query(models.User).filter_by(username="analyst").one().id
    res = client.post(f"/auth/users/{analyst_id}/reset-password", headers=csrf(client))
    temp = res.json()["temp_password"]
    assert login(TestClient(main.app), "analyst", ANALYST_PW).status_code == 401
    assert login(TestClient(main.app), "analyst", temp).json()["must_change_password"] is True


# ----------------------------------------------------------------- POST /score

# A20
def test_score_with_session_and_csrf(client):
    login(client)
    res = client.post("/score", headers=csrf(client), json=TXN)
    assert res.status_code == 200
    assert {"fraud_probability", "decision", "risk_level", "explanation", "neighbour_count", "inference_latency_ms"} <= res.json().keys()


# A21
def test_score_security_and_validation(client):
    assert client.post("/score", json=TXN).status_code in (401, 403)  # no session
    login(client)
    assert client.post("/score", json=TXN).status_code == 403  # no CSRF header
    bad = {**TXN, "features": {**TXN["features"], "type": "PAYMENT"}}
    assert client.post("/score", headers=csrf(client), json=bad).status_code == 422


# ------------------------------------------------------- results, stats, alerts

# A22
def test_results_and_stats(client, model):
    login(client)
    model.probability = 0.6  # REVIEW, raises an alert
    client.post("/score", headers=csrf(client), json=TXN)
    results = client.get("/api/fraud/results").json()
    assert results[0]["transaction_id"] == "api-1" and results[0]["alert_id"] is not None
    assert results[0]["decided_at"].endswith("Z")
    assert client.get("/api/fraud/results", params={"only_alerts": True}).json()[0]["decision"] == "REVIEW"
    stats = client.get("/api/fraud/stats").json()
    assert stats["total_scored"] == 1 and stats["review_count"] == 1 and stats["pending_alerts"] == 1


# A23
def test_acknowledge_uses_the_session_user(client, model, db):
    login(client)
    model.probability = 0.6
    client.post("/score", headers=csrf(client), json=TXN)
    alert_id = client.get("/api/fraud/results").json()[0]["alert_id"]
    analyst = as_analyst(db)
    res = analyst.post(f"/api/fraud/alerts/{alert_id}/acknowledge", headers=csrf(analyst),
                       json={"is_false_positive": True, "acknowledged_by": "someone-else"})
    assert res.status_code == 200
    row = client.get("/api/fraud/results").json()[0]
    assert row["acknowledged_by"] == "analyst" and row["is_false_positive"] is True
    assert client.get("/api/fraud/stats").json()["pending_alerts"] == 0
    assert client.post("/api/fraud/alerts/9999/acknowledge", headers=csrf(client), json={}).status_code == 404


# ------------------------------------------------------------ stream control

# A24
def test_stream_status_and_admin_only_toggle(client, db):
    analyst = as_analyst(db)
    before = analyst.get("/api/stream/status").json()["enabled"]
    assert analyst.post("/api/stream/toggle", headers=csrf(analyst)).status_code == 403
    login(client)
    assert client.post("/api/stream/toggle", headers=csrf(client)).json()["enabled"] is (not before)
    client.post("/api/stream/toggle", headers=csrf(client))  # restore


# ------------------------------------------------------------- WS /ws/alerts

# A25
def test_websocket_pushes_new_decisions(client):
    login(client)
    with client.websocket_connect("/ws/alerts", headers={"origin": ORIGIN}) as ws:
        client.post("/score", headers=csrf(client), json=TXN)
        msg = ws.receive_json()
    assert msg["type"] == "new_decision" and msg["transaction_id"] == "api-1"


# A26
def test_websocket_rejects_bad_origin_and_no_session(client):
    login(client)
    with pytest.raises(WebSocketDisconnect) as bad_origin:
        with client.websocket_connect("/ws/alerts", headers={"origin": "http://evil.example"}) as ws:
            ws.receive_text()
    assert bad_origin.value.code == 4403

    anonymous = TestClient(main.app)
    with pytest.raises(WebSocketDisconnect) as no_session:
        with anonymous.websocket_connect("/ws/alerts", headers={"origin": ORIGIN}) as ws:
            ws.receive_text()
    assert no_session.value.code == 4401
