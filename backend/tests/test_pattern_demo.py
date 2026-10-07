"""Fraud Patterns demo — pattern_demo.py + /api/patterns endpoints
(real SQLite DB, fakeredis, fake model)."""

import pytest
from fastapi.testclient import TestClient

import auth
import database
import main
import models
import pattern_demo
import redis_client
import transaction_admin

NUMERIC = ["type", "amount", "share_of_balance", "receiver_was_empty"]


@pytest.fixture
def client(db, model):
    def override_db():
        yield db

    main.app.dependency_overrides[database.get_db] = override_db
    main.app.dependency_overrides[auth.get_current_user] = lambda: models.User(id=1, username="tester", role="ANALYST")
    main.app.dependency_overrides[auth.require_csrf] = lambda: None
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def run(client, pattern, variant="fraud"):
    res = client.post(f"/api/patterns/{pattern}/run", params={"variant": variant})
    assert res.status_code == 200, res.text
    return res.json()


# PD-01
def test_list_patterns(client):
    body = client.get("/api/patterns").json()
    assert [p["id"] for p in body["patterns"]] == list(pattern_demo.PATTERNS)
    assert len(body["patterns"]) == 6
    assert "model" in body


# PD-02
def test_structuring_neighbours_build_up_step_by_step(client):
    body = run(client, "STRUCTURING")
    assert [s["neighbour_count"] for s in body["steps"]] == [0, 1, 2, 3]
    assert [s["amount"] for s in body["steps"]] == [48_200, 49_100, 47_500, 46_900]
    assert body["redis_available"] is True


# PD-03
def test_isolated_lookalike_is_identical_but_has_no_neighbours(client):
    fraud = run(client, "STRUCTURING", "fraud")["steps"]
    twin = run(client, "STRUCTURING", "lookalike")["steps"]
    assert [{k: s[k] for k in NUMERIC} for s in fraud] == [{k: s[k] for k in NUMERIC} for s in twin]
    assert all(s["neighbour_count"] == 0 for s in twin)


# PD-04
@pytest.mark.parametrize("pattern", list(pattern_demo.PATTERNS))
@pytest.mark.parametrize("variant", ["fraud", "lookalike"])
def test_every_pattern_and_variant_runs(client, pattern, variant):
    body = run(client, pattern, variant)
    assert body["steps"], "a run must score at least one transaction"
    for s in body["steps"]:
        assert 0 < s["amount"]
        assert 0.0 <= s["probability"] <= 1.0 and 0.0 <= s["alone_probability"] <= 1.0
        assert s["decision"] in {"ALLOW", "REVIEW", "BLOCK"}


# PD-05
@pytest.mark.parametrize("pattern", ["MULE_HUB", "AGENT_COLLUSION", "MULE_NETWORK", "SMURF_FANOUT", "ACCOUNT_TAKEOVER"])
def test_fraud_group_gets_linked(client, pattern):
    steps = run(client, pattern)["steps"]
    assert steps[-1]["neighbour_count"] >= 1
    assert max(s["neighbour_count"] for s in steps) >= 2


# PD-06
def test_runs_never_share_accounts(client):
    first = run(client, "MULE_HUB")
    second = run(client, "MULE_HUB")
    assert first["run_id"] != second["run_id"]
    assert second["steps"][0]["neighbour_count"] == 0  # nothing carried over from the first run
    accounts = lambda b: {s["sender"] for s in b["steps"]} | {s["receiver"] for s in b["steps"]}
    assert not accounts(first) & accounts(second)


# PD-07
def test_bb_rules_are_skipped(client, db):
    body = run(client, "STRUCTURING")  # 191,700 in one day from one sender — over the 50,000 limit
    assert all(s["risk_level"] != "REGULATORY" for s in body["steps"])
    sender = body["steps"][0]["sender"]
    assert redis_client.get_daily_total(sender, "TRANSFER") == 0.0


# PD-08
def test_transactions_are_stored_and_removable_as_manual_data(client, db):
    body = run(client, "AGENT_COLLUSION")
    ids = [s["transaction_id"] for s in body["steps"]]
    assert all(i.startswith("pattern-agent_collusion-fraud-") for i in ids)
    assert db.query(models.Transaction).count() == len(ids)
    assert transaction_admin.count_matching(db, "manual") == len(ids)


# PD-09
def test_alone_probability_comes_from_model_without_neighbours(client, model):
    model.probability = 0.33
    steps = run(client, "STRUCTURING")["steps"]
    assert all(s["alone_probability"] == 0.33 for s in steps)


# PD-10
def test_balances_chain_like_a_real_account(client):
    steps = run(client, "STRUCTURING")["steps"]
    shares = [s["share_of_balance"] for s in steps]
    assert shares == sorted(shares)  # same sender, shrinking balance -> each transfer is a bigger share


# PD-11
def test_unknown_pattern_and_bad_variant(client):
    assert client.post("/api/patterns/NOPE/run").status_code == 404
    assert client.post("/api/patterns/STRUCTURING/run", params={"variant": "other"}).status_code == 422


# PD-12
def test_requires_login_and_csrf(db, model):
    main.app.dependency_overrides.clear()
    anonymous = TestClient(main.app)
    assert anonymous.get("/api/patterns").status_code == 401
    assert anonymous.post("/api/patterns/STRUCTURING/run").status_code in (401, 403)
