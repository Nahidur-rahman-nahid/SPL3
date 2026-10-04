"""Night-time high-value transfer rule — decision_engine.apply_night_rule()."""

from datetime import datetime as real_datetime, timezone

import pytest
from pydantic import ValidationError

import decision_engine
from config import settings
from conftest import make_request

NIGHT_TIME = "2026-10-04T23:30:00"


def _apply(request, decision="ALLOW", risk_level="LOW"):
    return decision_engine.apply_night_rule(request, decision, risk_level)


# NR-01
def test_night_high_value_allow_becomes_review():
    decision, risk, info = _apply(make_request(amount=15000, transaction_time="2026-10-04T23:30:00"))
    assert (decision, risk) == ("REVIEW", "MEDIUM")
    assert info["rule"] == "NIGHT_HIGH_VALUE"
    assert info["changed_decision"] is True
    assert info["local_hour"] == 23


# NR-02
@pytest.mark.parametrize("hour", [23, 0, 1, 2, 3])
def test_every_night_hour_triggers(hour):
    decision, _, info = _apply(make_request(amount=15000, transaction_time=f"2026-10-04T{hour:02d}:15:00"))
    assert decision == "REVIEW"
    assert info is not None


# NR-03
@pytest.mark.parametrize("time_str", ["2026-10-04T22:59:59", "2026-10-04T04:00:00", "2026-10-04T12:00:00", "2026-10-04T18:30:00"])
def test_outside_night_window_no_trigger(time_str):
    decision, risk, info = _apply(make_request(amount=50000, transaction_time=time_str))
    assert (decision, risk, info) == ("ALLOW", "LOW", None)


# NR-04
def test_window_edges_inclusive():
    assert _apply(make_request(amount=15000, transaction_time="2026-10-04T23:00:00"))[0] == "REVIEW"
    assert _apply(make_request(amount=15000, transaction_time="2026-10-04T03:59:59"))[0] == "REVIEW"


# NR-05
def test_amount_exactly_threshold_does_not_trigger():
    decision, _, info = _apply(make_request(amount=10000, transaction_time="2026-10-04T01:00:00"))
    assert decision == "ALLOW"
    assert info is None


# NR-06
def test_amount_just_above_threshold_triggers():
    decision, _, info = _apply(make_request(amount=10000.01, transaction_time="2026-10-04T01:00:00"))
    assert decision == "REVIEW"
    assert info is not None


# NR-07
def test_low_amount_at_night_no_trigger():
    assert _apply(make_request(amount=500, transaction_time="2026-10-04T02:00:00"))[2] is None


# NR-08
def test_existing_review_is_kept_and_rule_recorded():
    decision, risk, info = _apply(make_request(amount=15000, transaction_time=NIGHT_TIME), "REVIEW", "MEDIUM")
    assert (decision, risk) == ("REVIEW", "MEDIUM")
    assert info["changed_decision"] is False


# NR-09
@pytest.mark.parametrize("risk", ["HIGH", "CRITICAL"])
def test_block_is_never_downgraded(risk):
    decision, new_risk, info = _apply(make_request(amount=15000, transaction_time=NIGHT_TIME), "BLOCK", risk)
    assert (decision, new_risk) == ("BLOCK", risk)
    assert info["changed_decision"] is False


# NR-10
def test_utc_time_converted_to_bangladesh_time():
    # 20:00 UTC == 02:00 Asia/Dhaka (UTC+6) → night
    decision, _, info = _apply(make_request(amount=15000, transaction_time="2026-10-04T20:00:00Z"))
    assert decision == "REVIEW"
    assert info["local_hour"] == 2
    # 12:00 UTC == 18:00 Dhaka → day
    assert _apply(make_request(amount=15000, transaction_time="2026-10-04T12:00:00Z"))[2] is None


# NR-11
def test_explicit_plus6_offset():
    assert _apply(make_request(amount=15000, transaction_time="2026-10-04T23:10:00+06:00"))[0] == "REVIEW"


# NR-12
def test_naive_time_treated_as_local():
    # naive 02:00 is 02:00 local, NOT 02:00 UTC (which would be 08:00 local)
    assert _apply(make_request(amount=15000, transaction_time="2026-10-04T02:00:00"))[0] == "REVIEW"


class _FrozenDatetime(real_datetime):
    frozen_utc = None

    @classmethod
    def now(cls, tz=None):
        return cls.frozen_utc.astimezone(tz) if tz else cls.frozen_utc.replace(tzinfo=None)


# NR-13
@pytest.mark.parametrize("utc_hour,expected", [(18, "REVIEW"), (21, "REVIEW"), (6, "ALLOW")])
def test_missing_transaction_time_uses_current_bangladesh_time(monkeypatch, utc_hour, expected):
    # 18:00 UTC = 00:00 Dhaka, 21:00 UTC = 03:00 Dhaka, 06:00 UTC = 12:00 Dhaka
    _FrozenDatetime.frozen_utc = real_datetime(2026, 10, 4, utc_hour, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(decision_engine, "datetime", _FrozenDatetime)
    assert _apply(make_request(amount=15000, transaction_time=None))[0] == expected


# NR-14
def test_rule_disabled(monkeypatch):
    monkeypatch.setattr(settings, "NIGHT_RULE_ENABLED", False)
    assert _apply(make_request(amount=50000, transaction_time=NIGHT_TIME)) == ("ALLOW", "LOW", None)


# NR-15
def test_custom_hours_and_threshold(monkeypatch):
    monkeypatch.setattr(settings, "NIGHT_RULE_HOURS", "22")
    monkeypatch.setattr(settings, "NIGHT_RULE_MIN_AMOUNT", 1000)
    assert _apply(make_request(amount=1500, transaction_time="2026-10-04T22:30:00"))[0] == "REVIEW"
    assert _apply(make_request(amount=1500, transaction_time="2026-10-04T23:30:00"))[0] == "ALLOW"


# NR-16
def test_applies_to_cash_out_too():
    assert _apply(make_request(amount=15000, txn_type="CASH_OUT", transaction_time=NIGHT_TIME))[0] == "REVIEW"


# NR-17
def test_explanation_summary_and_rules_list_when_changed():
    _, _, info = _apply(make_request(amount=15000, transaction_time=NIGHT_TIME))
    explanation = {"summary": "Linked to 2 related transaction(s)."}
    decision_engine.attach_rule_to_explanation(explanation, info)
    assert explanation["summary"].startswith("Held for REVIEW by night-time rule: BDT 15,000 at 23:00")
    assert explanation["summary"].endswith("Linked to 2 related transaction(s).")
    assert explanation["rules_triggered"] == [info]


# NR-18
def test_explanation_summary_untouched_when_not_changed():
    _, _, info = _apply(make_request(amount=15000, transaction_time=NIGHT_TIME), "BLOCK", "CRITICAL")
    explanation = {"summary": "original"}
    decision_engine.attach_rule_to_explanation(explanation, info)
    assert explanation["summary"] == "original"
    assert explanation["rules_triggered"][0]["changed_decision"] is False


# NR-19
def test_request_without_transaction_time_is_valid():
    assert make_request(transaction_time=None).transaction_time is None


# NR-20
def test_invalid_transaction_time_rejected():
    with pytest.raises(ValidationError):
        make_request(transaction_time="not-a-date")

