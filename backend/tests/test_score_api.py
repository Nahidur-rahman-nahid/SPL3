"""POST /score and GET /api/fraud/results over HTTP (auth/CSRF dependencies
overridden; startup hooks — model load, stream simulator — are not run)."""

import pytest
from fastapi.testclient import TestClient

import auth
import database
import main
import models
from conftest import DAYTIME, NIGHT

BASE_PAYLOAD = {
    "sender_account": "C_API_1",
    "receiver_account": "M_API_1",
    "features": {
        "step": 1, "type": "TRANSFER", "amount": 20000, "oldbalanceOrg": 21000,
        "newbalanceOrig": 1000, "oldbalanceDest": 0, "newbalanceDest": 20000,
    },
    "neighbours": [],
}


def payload(txn_id, amount=20000, transaction_time=DAYTIME, **extra):
    body = {**BASE_PAYLOAD, "transaction_id": txn_id, "features": {**BASE_PAYLOAD["features"], "amount": amount}}
    if transaction_time is not None:
        body["transaction_time"] = transaction_time
    body.update(extra)
    return body


@pytest.fixture
def client(db, model):
    def override_db():
        yield db

    main.app.dependency_overrides[database.get_db] = override_db
    main.app.dependency_overrides[auth.get_current_user] = lambda: models.User(id=1, username="tester", role="ANALYST")
    main.app.dependency_overrides[auth.require_csrf] = lambda: None
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


# API-01
def test_score_returns_limit_usage(client):
    res = client.post("/score", json=payload("api-1"))
    assert res.status_code == 200
    assert res.json()["limit_usage"] == {"type": "TRANSFER", "period": "daily", "used": 20000.0, "limit": 50000.0, "remaining": 30000.0}


# API-02
def test_score_regulatory_block_over_http(client):
    client.post("/score", json=payload("api-2a", amount=40000))
    body = client.post("/score", json=payload("api-2b", amount=20000)).json()
    assert (body["decision"], body["risk_level"]) == ("BLOCK", "REGULATORY")
    assert body["limit_usage"]["remaining"] == 10000.0


# API-03
def test_night_rule_over_http(client):
    body = client.post("/score", json=payload("api-3", amount=15000, transaction_time=NIGHT)).json()
    assert body["decision"] == "REVIEW"
    assert body["explanation"]["rules_triggered"][0]["rule"] == "NIGHT_HIGH_VALUE"


# API-04
def test_invalid_transaction_time_returns_422(client):
    assert client.post("/score", json=payload("api-4", transaction_time="31-31-2026")).status_code == 422


# API-05
def test_legacy_payload_without_new_fields_still_works(client):
    body = client.post("/score", json=payload("api-5", amount=5000, transaction_time=None)).json()
    for field in ("transaction_id", "fraud_probability", "decision", "risk_level", "explanation", "neighbour_count", "inference_latency_ms"):
        assert field in body


# API-06
def test_results_filter_by_regulatory_risk_level(client):
    client.post("/score", json=payload("api-6a", amount=60000))
    client.post("/score", json=payload("api-6b", amount=1000))
    rows = client.get("/api/fraud/results", params={"risk_level": "regulatory"}).json()
    assert [r["transaction_id"] for r in rows] == ["api-6a"]


# API-07
def test_unsupported_type_still_rejected(client):
    body = payload("api-7")
    body["features"]["type"] = "PAYMENT"
    assert client.post("/score", json=body).status_code == 422


# API-08
def test_score_still_requires_authentication(db, model):
    main.app.dependency_overrides.clear()
    res = TestClient(main.app).post("/score", json=payload("api-8"))
    assert res.status_code in (401, 403)


# API-09
def test_result_timestamps_carry_utc_offset(client):
    """The DB returns naive UTC datetimes; without an offset the browser
    would show them as local time (6 hours early in Dhaka)."""
    from datetime import datetime, timedelta, timezone

    assert client.post("/score", json=payload("api-tz-1")).status_code == 200
    stamp = client.get("/api/fraud/results").json()[0]["decided_at"]
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert parsed.utcoffset() == timedelta(0)
    assert abs(datetime.now(timezone.utc) - parsed) < timedelta(minutes=1)
