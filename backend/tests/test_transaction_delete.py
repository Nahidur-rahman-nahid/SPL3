"""Admin deletion of transactions — transaction_admin.py + the three
/api/transactions endpoints (real SQLite DB, fakeredis, fake model)."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import auth
import database
import main
import models
import redis_client
import transaction_admin
from conftest import make_request
from scoring_pipeline import gather_neighbours, process_transaction


def score(db, txn_id, amount=5000.0, sender="C_SENDER_1", receiver="C_RECEIVER_1", source=None):
    source = source or ("stream" if txn_id.startswith("stream-") else "api")
    request = make_request(amount=amount, sender=sender, receiver=receiver, transaction_id=txn_id)
    return asyncio.run(process_transaction(request, db, source=source))


def counts(db):
    return (
        db.query(models.Transaction).count(),
        db.query(models.FraudDecision).count(),
        db.query(models.Alert).count(),
    )


def neighbour_ids(account="C_SENDER_1"):
    return {e["transaction_id"] for e in redis_client.get_recent_transactions(account)}


# ------------------------------------------------------------ delete by IDs

# DEL-01
def test_delete_single_removes_transaction_decision_and_alert(db, model):
    model.probability = 0.6  # REVIEW → alert created
    score(db, "demo-1")
    assert counts(db) == (1, 1, 1)
    result = transaction_admin.delete_transactions(db, ["demo-1"])
    assert result == {"deleted_transactions": 1, "deleted_decisions": 1, "deleted_alerts": 1}
    assert counts(db) == (0, 0, 0)


# DEL-02
def test_delete_multiple(db, model):
    for i in range(5):
        score(db, f"demo-{i}")
    result = transaction_admin.delete_transactions(db, ["demo-0", "demo-2", "demo-4"])
    assert result["deleted_transactions"] == 3
    remaining = {t.transaction_id for t in db.query(models.Transaction).all()}
    assert remaining == {"demo-1", "demo-3"}


# DEL-03
def test_unknown_ids_ignored(db, model):
    score(db, "demo-1")
    assert transaction_admin.delete_transactions(db, ["nope-1", "nope-2"]) == {
        "deleted_transactions": 0, "deleted_decisions": 0, "deleted_alerts": 0,
    }
    assert transaction_admin.delete_transactions(db, ["nope-3", "demo-1"])["deleted_transactions"] == 1


# DEL-04
def test_duplicate_ids_counted_once(db, model):
    score(db, "demo-1")
    assert transaction_admin.delete_transactions(db, ["demo-1", "demo-1", "demo-1"])["deleted_transactions"] == 1


# DEL-05
def test_rescored_transaction_all_decisions_removed(db, model):
    score(db, "demo-1")
    score(db, "demo-1")
    assert counts(db) == (1, 2, 0)
    assert transaction_admin.delete_transactions(db, ["demo-1"])["deleted_decisions"] == 2
    assert counts(db) == (0, 0, 0)


# DEL-06
def test_other_transactions_and_their_alerts_untouched(db, model):
    model.probability = 0.6
    score(db, "demo-1")
    score(db, "demo-2")
    transaction_admin.delete_transactions(db, ["demo-1"])
    assert counts(db) == (1, 1, 1)
    assert db.query(models.FraudDecision).one().transaction_id == "demo-2"


# DEL-07
def test_deleted_transaction_no_longer_a_neighbour(db, model):
    score(db, "demo-1")
    score(db, "demo-2")
    assert neighbour_ids() == {"demo-1", "demo-2"}
    transaction_admin.delete_transactions(db, ["demo-1"])
    assert neighbour_ids() == {"demo-2"}
    assert neighbour_ids("C_RECEIVER_1") == {"demo-2"}
    later = make_request(transaction_id="demo-3")
    assert [n.transaction_id for n in gather_neighbours(later)] == ["demo-2"]


# DEL-08
def test_daily_limit_refunded_for_counted_manual_transaction(db, model):
    score(db, "demo-1", amount=30000)
    score(db, "demo-2", amount=15000)
    assert redis_client.get_daily_total("C_SENDER_1", "TRANSFER") == 45000.0
    transaction_admin.delete_transactions(db, ["demo-1"])
    assert redis_client.get_daily_total("C_SENDER_1", "TRANSFER") == 15000.0


# DEL-09
def test_no_refund_for_model_blocked_or_stream_transactions(db, model):
    score(db, "demo-ok", amount=20000)
    model.probability = 0.95  # BLOCK → never counted
    score(db, "demo-blocked", amount=10000)
    model.probability = 0.05
    score(db, "stream-1", amount=10000)  # stream → never counted
    transaction_admin.delete_transactions(db, ["demo-blocked", "stream-1"])
    assert redis_client.get_daily_total("C_SENDER_1", "TRANSFER") == 20000.0


# DEL-10
def test_refund_never_negative_and_never_recreates_expired_day(db, model):
    score(db, "demo-1", amount=20000)
    redis_client.get_client().set(redis_client._daily_key("C_SENDER_1", "TRANSFER"), 5000)  # e.g. partly reset
    transaction_admin.delete_transactions(db, ["demo-1"])
    assert redis_client.get_daily_total("C_SENDER_1", "TRANSFER") == 0.0

    score(db, "demo-2", amount=20000)
    redis_client.get_client().delete(redis_client._daily_key("C_SENDER_1", "TRANSFER"))  # day expired
    transaction_admin.delete_transactions(db, ["demo-2"])
    assert not redis_client.get_client().exists(redis_client._daily_key("C_SENDER_1", "TRANSFER"))


# DEL-11
def test_redis_down_database_delete_still_succeeds(db, model, fake_redis):
    score(db, "demo-1")
    fake_redis.connected = False
    assert transaction_admin.delete_transactions(db, ["demo-1"])["deleted_transactions"] == 1
    assert counts(db) == (0, 0, 0)


# DEL-12
def test_large_batch_spanning_multiple_chunks(db, model):
    now = datetime.now(timezone.utc)
    db.add_all(
        models.Transaction(
            transaction_id=f"stream-bulk-{i}", sender_account="S", receiver_account="R", amount=1.0,
            transaction_type="TRANSFER", step=1, received_at=now,
        )
        for i in range(1200)
    )
    db.commit()
    assert transaction_admin.purge(db, "stream")["deleted_transactions"] == 1200
    assert db.query(models.Transaction).count() == 0


# ----------------------------------------------------------------- purge

@pytest.fixture
def mixed_data(db, model):
    for i in range(3):
        score(db, f"stream-{i}", sender="SIM_A", receiver="SIM_B")
    for i in range(2):
        score(db, f"demo-{i}")
    return db


# PUR-01
@pytest.mark.parametrize("source,expected", [("stream", 3), ("manual", 2), ("all", 5)])
def test_preview_counts_by_source(mixed_data, source, expected):
    assert transaction_admin.count_matching(mixed_data, source) == expected


# PUR-02
def test_purge_stream_keeps_manual(mixed_data):
    assert transaction_admin.purge(mixed_data, "stream")["deleted_transactions"] == 3
    assert {t.transaction_id for t in mixed_data.query(models.Transaction).all()} == {"demo-0", "demo-1"}


# PUR-03
def test_purge_manual_keeps_stream(mixed_data):
    assert transaction_admin.purge(mixed_data, "manual")["deleted_transactions"] == 2
    assert all(t.transaction_id.startswith("stream-") for t in mixed_data.query(models.Transaction).all())


# PUR-04
def test_purge_all(mixed_data):
    assert transaction_admin.purge(mixed_data, "all")["deleted_transactions"] == 5
    assert counts(mixed_data) == (0, 0, 0)


# PUR-05
def test_purge_before_only_removes_older(db, model):
    score(db, "demo-old")
    old = db.get(models.Transaction, "demo-old")
    old.received_at = datetime.now(timezone.utc) - timedelta(days=3)
    db.commit()
    score(db, "demo-new")
    cutoff = datetime.now(timezone(timedelta(hours=6))) - timedelta(days=1)  # aware, Dhaka offset
    assert transaction_admin.count_matching(db, "all", cutoff) == 1
    assert transaction_admin.purge(db, "all", cutoff)["deleted_transactions"] == 1
    assert db.get(models.Transaction, "demo-new") is not None


# PUR-06
def test_purge_with_nothing_matching(db, model):
    score(db, "demo-1")
    assert transaction_admin.purge(db, "stream") == {"deleted_transactions": 0, "deleted_decisions": 0, "deleted_alerts": 0}


# ------------------------------------------------------------------- API

@pytest.fixture
def as_role(db, model):
    def make(role="ADMIN", csrf=True):
        def override_db():
            yield db

        main.app.dependency_overrides.clear()
        main.app.dependency_overrides[database.get_db] = override_db
        main.app.dependency_overrides[auth.get_current_user] = lambda: models.User(id=1, username="tester", role=role)
        if csrf:
            main.app.dependency_overrides[auth.require_csrf] = lambda: None
        return TestClient(main.app)

    yield make
    main.app.dependency_overrides.clear()


def api_delete(client, ids):
    return client.request("DELETE", "/api/transactions", json={"transaction_ids": ids})


# API-D-01
def test_api_delete_as_admin(as_role, db, model):
    score(db, "demo-1")
    res = api_delete(as_role("ADMIN"), ["demo-1"])
    assert res.status_code == 200
    assert res.json() == {"deleted_transactions": 1, "deleted_decisions": 1, "deleted_alerts": 0}


# API-D-02
def test_api_delete_forbidden_for_analyst(as_role, db, model):
    score(db, "demo-1")
    assert api_delete(as_role("ANALYST"), ["demo-1"]).status_code == 403
    assert db.query(models.Transaction).count() == 1


# API-D-03
def test_api_delete_requires_csrf(as_role, db, model):
    score(db, "demo-1")
    assert api_delete(as_role("ADMIN", csrf=False), ["demo-1"]).status_code == 403
    assert db.query(models.Transaction).count() == 1


# API-D-04
@pytest.mark.parametrize("ids", [[], [f"x-{i}" for i in range(501)]])
def test_api_delete_rejects_empty_or_oversized_list(as_role, ids):
    assert api_delete(as_role("ADMIN"), ids).status_code == 422


# API-D-05
def test_api_purge_preview(as_role, mixed_data):
    client = as_role("ADMIN")
    assert client.get("/api/transactions/purge-preview", params={"source": "stream"}).json() == {"count": 3}
    assert client.get("/api/transactions/purge-preview", params={"source": "bogus"}).status_code == 422
    assert as_role("ANALYST").get("/api/transactions/purge-preview", params={"source": "all"}).status_code == 403


# API-D-06
def test_api_purge_and_broadcasts(as_role, mixed_data, monkeypatch):
    sent = []

    async def capture(message):
        sent.append(message)

    monkeypatch.setattr(main.manager, "broadcast", capture)
    client = as_role("ADMIN")
    assert client.post("/api/transactions/purge", json={"source": "stream"}).json()["deleted_transactions"] == 3
    api_delete(client, ["demo-0"])
    assert sent == [{"type": "transactions_purged"}, {"type": "transactions_deleted", "transaction_ids": ["demo-0"]}]


# API-D-07
def test_api_purge_invalid_source_and_analyst(as_role, mixed_data):
    assert as_role("ADMIN").post("/api/transactions/purge", json={"source": "everything"}).status_code == 422
    assert as_role("ANALYST").post("/api/transactions/purge", json={"source": "all"}).status_code == 403
    assert mixed_data.query(models.Transaction).count() == 5


# API-D-08
def test_api_delete_requires_login(db, model):
    main.app.dependency_overrides.clear()
    assert api_delete(TestClient(main.app), ["demo-1"]).status_code in (401, 403)


# API-D-09
def test_deleted_rows_gone_from_results_and_stats(as_role, db, model):
    model.probability = 0.6
    score(db, "demo-1")
    score(db, "demo-2")
    client = as_role("ADMIN")
    api_delete(client, ["demo-1"])
    assert [r["transaction_id"] for r in client.get("/api/fraud/results").json()] == ["demo-2"]
    assert client.get("/api/fraud/stats").json()["total_scored"] == 1
