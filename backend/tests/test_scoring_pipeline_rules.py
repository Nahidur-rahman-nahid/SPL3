"""Regulatory limits + night rule inside scoring_pipeline.process_transaction()
(real SQLite DB, fakeredis, controllable fake model)."""

import asyncio

import pytest

import models
import redis_client
from config import settings
from conftest import NIGHT, make_request
from scoring_pipeline import process_transaction


def score(request, db, source="api"):
    return asyncio.run(process_transaction(request, db, source=source))


def used_today(account="C_SENDER_1", txn_type="TRANSFER"):
    return redis_client.get_daily_total(account, txn_type)


# RL-01
def test_first_transaction_allowed_and_usage_reported(db, model):
    result = score(make_request(amount=20000), db)
    assert result["decision"] == "ALLOW"
    assert result["limit_usage"] == {"type": "TRANSFER", "period": "daily", "used": 20000.0, "limit": 50000, "remaining": 30000.0}


# RL-02
def test_cumulative_up_to_exact_limit_allowed(db, model):
    score(make_request(amount=30000), db)
    result = score(make_request(amount=20000), db)
    assert result["decision"] == "ALLOW"
    assert result["limit_usage"]["used"] == 50000.0
    assert result["limit_usage"]["remaining"] == 0


# RL-03
def test_exceeding_daily_transfer_limit_blocks_regulatory(db, model):
    score(make_request(amount=42000), db)
    result = score(make_request(amount=15000), db)
    assert (result["decision"], result["risk_level"]) == ("BLOCK", "REGULATORY")
    violation = result["explanation"]["regulatory_violation"]
    assert violation == {"rule": "BB_DAILY_TRANSFER_LIMIT", "limit": 50000, "used_today": 42000.0, "attempted": 15000.0}
    assert result["explanation"]["summary"].startswith("Blocked by Bangladesh Bank daily TRANSFER limit")
    assert result["limit_usage"]["used"] == 42000.0  # blocked amount not counted
    assert used_today() == 42000.0


# RL-04
def test_cash_out_limit_is_30000(db, model):
    assert score(make_request(amount=30000, txn_type="CASH_OUT"), db)["decision"] == "ALLOW"
    result = score(make_request(amount=0.01, txn_type="CASH_OUT"), db)
    assert result["risk_level"] == "REGULATORY"
    assert result["limit_usage"]["limit"] == 30000


# RL-05
def test_single_transaction_over_limit_blocked_with_no_history(db, model):
    result = score(make_request(amount=50000.01), db)
    assert result["risk_level"] == "REGULATORY"
    assert used_today() == 0.0


# RL-06
def test_small_transaction_within_remaining_allowed_after_block(db, model):
    score(make_request(amount=45000), db)
    assert score(make_request(amount=10000), db)["risk_level"] == "REGULATORY"
    result = score(make_request(amount=5000), db)
    assert result["decision"] == "ALLOW"
    assert result["limit_usage"]["used"] == 50000.0


# RL-07
def test_model_block_is_not_counted_toward_limit(db, model):
    model.probability = 0.95  # model → BLOCK CRITICAL
    result = score(make_request(amount=40000), db)
    assert (result["decision"], result["risk_level"]) == ("BLOCK", "CRITICAL")
    assert result["limit_usage"]["used"] == 0.0
    assert used_today() == 0.0


# RL-08
def test_model_review_is_counted(db, model):
    model.probability = 0.5  # REVIEW MEDIUM
    result = score(make_request(amount=40000), db)
    assert result["decision"] == "REVIEW"
    assert used_today() == 40000.0


# RL-09
def test_night_rule_review_is_counted(db, model):
    result = score(make_request(amount=15000, transaction_time=NIGHT), db)
    assert result["decision"] == "REVIEW"
    assert used_today() == 15000.0


# RL-10
def test_model_still_runs_on_regulatory_block_and_row_persisted(db, model):
    model.probability = 0.12
    result = score(make_request(amount=60000, transaction_id="reg-1"), db)
    assert model.calls >= 1
    assert result["fraud_probability"] == 0.12
    row = db.query(models.FraudDecision).filter_by(transaction_id="reg-1").one()
    assert (row.decision, row.risk_level, row.fraud_probability) == ("BLOCK", "REGULATORY", 0.12)
    assert db.get(models.Transaction, "reg-1") is not None


# RL-11
@pytest.mark.parametrize("probability,expect_alert", [(0.05, False), (0.6, True)])
def test_alert_on_regulatory_block_follows_model_score(db, model, probability, expect_alert):
    model.probability = probability
    result = score(make_request(amount=60000, transaction_id=f"reg-alert-{probability}"), db)
    assert result["risk_level"] == "REGULATORY"
    decision = db.query(models.FraudDecision).filter_by(transaction_id=f"reg-alert-{probability}").one()
    assert (len(decision.alerts) == 1) is expect_alert


# RL-12
def test_stream_source_skips_limit_check(db, model):
    result = score(make_request(amount=200000), db, source="stream")
    assert result["decision"] == "ALLOW"
    assert result["limit_usage"] is None
    assert used_today() == 0.0


# RL-13
def test_stream_source_skips_night_rule(db, model):
    result = score(make_request(amount=15000, transaction_time=NIGHT), db, source="stream")
    assert result["decision"] == "ALLOW"
    assert "rules_triggered" not in result["explanation"]


# RL-14
def test_limits_disabled(db, model, monkeypatch):
    monkeypatch.setattr(settings, "REGULATORY_LIMITS_ENABLED", False)
    result = score(make_request(amount=200000), db)
    assert result["decision"] == "ALLOW"
    assert result["limit_usage"] is None


# RL-15
def test_redis_down_fails_open(db, model, redis_down):
    result = score(make_request(amount=200000), db)
    assert result["decision"] == "ALLOW"
    assert result["limit_usage"] is None


# RL-16
def test_rescoring_allowed_transaction_not_double_counted(db, model):
    score(make_request(amount=30000, transaction_id="dup-1"), db)
    result = score(make_request(amount=30000, transaction_id="dup-1"), db)
    assert result["decision"] == "ALLOW"
    assert used_today() == 30000.0


# RL-17
def test_rescoring_regulatory_blocked_transaction_stays_blocked(db, model):
    score(make_request(amount=45000), db)
    score(make_request(amount=10000, transaction_id="dup-2"), db)
    result = score(make_request(amount=10000, transaction_id="dup-2"), db)
    assert result["risk_level"] == "REGULATORY"
    assert used_today() == 45000.0


# RL-18
@pytest.mark.parametrize("amount", [0, -5000])
def test_zero_or_negative_amount_never_changes_total(db, model, amount):
    score(make_request(amount=20000), db)
    result = score(make_request(amount=amount), db)
    assert result["limit_usage"]["used"] == 20000.0
    assert used_today() == 20000.0


# RL-19
def test_scoring_failure_releases_reservation(db, model):
    model.raise_error = True
    with pytest.raises(RuntimeError):
        score(make_request(amount=20000), db)
    assert used_today() == 0.0


# RL-20
def test_limit_applies_to_sender_only(db, model):
    score(make_request(amount=50000, sender="A", receiver="B"), db)
    assert used_today("B") == 0.0
    result = score(make_request(amount=50000, sender="B", receiver="A"), db)
    assert result["decision"] == "ALLOW"


# RL-21
def test_regulatory_block_takes_precedence_over_night_rule(db, model):
    result = score(make_request(amount=60000, transaction_time=NIGHT), db)
    assert result["risk_level"] == "REGULATORY"
    assert "rules_triggered" not in result["explanation"]


# RL-22
def test_night_rule_forces_alert_even_with_low_model_score(db, model):
    model.probability = 0.05
    score(make_request(amount=15000, transaction_time=NIGHT, transaction_id="night-1"), db)
    decision = db.query(models.FraudDecision).filter_by(transaction_id="night-1").one()
    assert (decision.decision, decision.risk_level) == ("REVIEW", "MEDIUM")
    assert len(decision.alerts) == 1
    assert decision.explanation["rules_triggered"][0]["rule"] == "NIGHT_HIGH_VALUE"


# RL-23
def test_daytime_low_score_has_no_alert(db, model):
    score(make_request(amount=15000, transaction_id="day-1"), db)
    decision = db.query(models.FraudDecision).filter_by(transaction_id="day-1").one()
    assert decision.decision == "ALLOW"
    assert decision.alerts == []


# RL-24
def test_custom_limits_from_config(db, model, monkeypatch):
    monkeypatch.setattr(settings, "DAILY_LIMIT_TRANSFER", 1000)
    assert score(make_request(amount=1001), db)["risk_level"] == "REGULATORY"


# RL-25
def test_regulatory_decision_is_broadcast(db, model, monkeypatch):
    import scoring_pipeline

    sent = []

    async def capture(message):
        sent.append(message)

    monkeypatch.setattr(scoring_pipeline.manager, "broadcast", capture)
    score(make_request(amount=60000), db)
    assert sent[0]["decision"] == "BLOCK"
    assert sent[0]["risk_level"] == "REGULATORY"
