"""End-to-end with the REAL FraudGNN checkpoint (no fake model) — confirms
the new rules work on top of actual inference. Skipped if ./model files or
torch are unavailable."""

import asyncio
from pathlib import Path

import pytest

from config import settings
from conftest import NIGHT, make_request

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "model" / "FraudGNN_main.pt").exists(),
    reason="model checkpoint not present",
)


@pytest.fixture(scope="module")
def real_model():
    import os

    import ml_service

    os.chdir(Path(__file__).resolve().parent.parent)  # MODEL_PATH etc. are relative to backend/
    ml_service.load_model()
    return ml_service


def score(request, db, source="api"):
    from scoring_pipeline import process_transaction

    return asyncio.run(process_transaction(request, db, source=source))


# SM-01
def test_real_model_returns_valid_probability_with_usage(db, real_model):
    result = score(make_request(amount=5000), db)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["limit_usage"]["limit"] == settings.DAILY_LIMIT_TRANSFER


# SM-02
def test_real_model_regulatory_block(db, real_model):
    result = score(make_request(amount=60000), db)
    assert (result["decision"], result["risk_level"]) == ("BLOCK", "REGULATORY")
    assert 0.0 <= result["fraud_probability"] <= 1.0


# SM-03
def test_real_model_night_rule_never_lowers_decision(db, real_model):
    day = score(make_request(amount=15000, sender="S_DAY"), db, source="stream")
    night = score(make_request(amount=15000, sender="S_NIGHT", transaction_time=NIGHT), db)
    order = {"ALLOW": 0, "REVIEW": 1, "BLOCK": 2}
    assert order[night["decision"]] >= max(order[day["decision"]], order["REVIEW"])
