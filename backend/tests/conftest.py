"""
Shared test fixtures. Tests never touch the real Postgres/Upstash from .env:
DATABASE_URL is pointed at a throwaway SQLite file and Redis is replaced by
an in-process fakeredis before any backend module is imported (environment
variables take precedence over .env in pydantic-settings).
"""

import os
import sys
import tempfile
from pathlib import Path

_TEST_DB = Path(tempfile.gettempdir()) / "spl3_test_fraud.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB.as_posix()}"
os.environ["REDIS_URL"] = "redis://localhost:6399/0"  # never contacted — fakeredis is injected
os.environ["STREAM_ENABLED"] = "False"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fakeredis  # noqa: E402
import pytest  # noqa: E402

import ml_service  # noqa: E402
import redis_client  # noqa: E402
from database import Base, SessionLocal, engine  # noqa: E402
from schemas import ScoreRequest  # noqa: E402

DAYTIME = "2026-10-04T12:00:00"  # outside the night-rule window
NIGHT = "2026-10-04T23:30:00"


@pytest.fixture(autouse=True)
def fake_redis():
    server = fakeredis.FakeServer()
    redis_client._client = fakeredis.FakeRedis(server=server, decode_responses=True)
    yield server
    redis_client._client = None


@pytest.fixture
def redis_down(fake_redis):
    fake_redis.connected = False
    return fake_redis


@pytest.fixture
def db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


class FakeModel:
    """Stands in for ml_service.score so each test controls the GNN output."""

    def __init__(self):
        self.probability = 0.05  # ALLOW / LOW by default
        self.calls = 0
        self.raise_error = False

    def score(self, target, neighbours):
        self.calls += 1
        if self.raise_error:
            raise RuntimeError("simulated inference failure")
        return self.probability, 3


@pytest.fixture
def model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(ml_service, "score", fake.score)
    return fake


def make_request(
    amount=5000.0,
    txn_type="TRANSFER",
    sender="C_SENDER_1",
    receiver="C_RECEIVER_1",
    transaction_id=None,
    transaction_time=DAYTIME,
):
    make_request.counter += 1
    payload = {
        "transaction_id": transaction_id or f"test-{make_request.counter}",
        "sender_account": sender,
        "receiver_account": receiver,
        "features": {
            "step": 1,
            "type": txn_type,
            "amount": amount,
            "oldbalanceOrg": amount + 1000,
            "newbalanceOrig": 1000,
            "oldbalanceDest": 0,
            "newbalanceDest": amount,
        },
        "neighbours": [],
    }
    if transaction_time is not None:
        payload["transaction_time"] = transaction_time
    return ScoreRequest(**payload)


make_request.counter = 0
