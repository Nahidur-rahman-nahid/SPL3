"""Daily limit counters — redis_client.get/reserve/release_daily_amount()."""

import threading
from datetime import datetime, timezone

import redis_client

NOON_UTC = datetime(2026, 10, 4, 6, 0, tzinfo=timezone.utc)  # 12:00 in Dhaka


# RC-01
def test_empty_total_is_zero():
    assert redis_client.get_daily_total("C1", "TRANSFER", NOON_UTC) == 0.0


# RC-02
def test_reserve_within_limit_accepted():
    assert redis_client.reserve_daily_amount("C1", "TRANSFER", 20000, 50000, NOON_UTC) == (True, 20000.0)
    assert redis_client.get_daily_total("C1", "TRANSFER", NOON_UTC) == 20000.0


# RC-03
def test_reserves_accumulate():
    redis_client.reserve_daily_amount("C1", "TRANSFER", 20000, 50000, NOON_UTC)
    assert redis_client.reserve_daily_amount("C1", "TRANSFER", 15000, 50000, NOON_UTC) == (True, 35000.0)


# RC-04
def test_reaching_limit_exactly_is_accepted():
    redis_client.reserve_daily_amount("C1", "TRANSFER", 30000, 50000, NOON_UTC)
    assert redis_client.reserve_daily_amount("C1", "TRANSFER", 20000, 50000, NOON_UTC) == (True, 50000.0)


# RC-05
def test_exceeding_limit_rejected_and_rolled_back():
    redis_client.reserve_daily_amount("C1", "TRANSFER", 40000, 50000, NOON_UTC)
    assert redis_client.reserve_daily_amount("C1", "TRANSFER", 10000.01, 50000, NOON_UTC) == (False, 40000.0)
    assert redis_client.get_daily_total("C1", "TRANSFER", NOON_UTC) == 40000.0


# RC-06
def test_single_amount_over_limit_rejected_on_empty_day():
    assert redis_client.reserve_daily_amount("C1", "CASH_OUT", 30001, 30000, NOON_UTC) == (False, 0.0)
    assert redis_client.get_daily_total("C1", "CASH_OUT", NOON_UTC) == 0.0


# RC-07
def test_release_reduces_total():
    redis_client.reserve_daily_amount("C1", "TRANSFER", 20000, 50000, NOON_UTC)
    redis_client.release_daily_amount("C1", "TRANSFER", 20000, NOON_UTC)
    assert redis_client.get_daily_total("C1", "TRANSFER", NOON_UTC) == 0.0


# RC-08
def test_key_format_and_ttl():
    redis_client.reserve_daily_amount("C1", "TRANSFER", 100, 50000, NOON_UTC)
    client = redis_client.get_client()
    assert client.exists("daily:C1:TRANSFER:20261004")
    assert 0 < client.ttl("daily:C1:TRANSFER:20261004") <= redis_client.DAILY_TTL_SECONDS


# RC-09
def test_types_counted_separately():
    redis_client.reserve_daily_amount("C1", "TRANSFER", 45000, 50000, NOON_UTC)
    assert redis_client.reserve_daily_amount("C1", "CASH_OUT", 25000, 30000, NOON_UTC) == (True, 25000.0)


# RC-10
def test_accounts_counted_separately():
    redis_client.reserve_daily_amount("C1", "TRANSFER", 50000, 50000, NOON_UTC)
    assert redis_client.reserve_daily_amount("C2", "TRANSFER", 50000, 50000, NOON_UTC) == (True, 50000.0)


# RC-11
def test_total_resets_at_bangladesh_midnight_not_utc_midnight():
    before_midnight = datetime(2026, 10, 4, 17, 59, tzinfo=timezone.utc)  # 23:59 Dhaka, Oct 4
    after_midnight = datetime(2026, 10, 4, 18, 1, tzinfo=timezone.utc)    # 00:01 Dhaka, Oct 5 (still Oct 4 in UTC)
    redis_client.reserve_daily_amount("C1", "TRANSFER", 50000, 50000, before_midnight)
    assert redis_client.reserve_daily_amount("C1", "TRANSFER", 50000, 50000, after_midnight) == (True, 50000.0)
    assert redis_client.get_client().exists("daily:C1:TRANSFER:20261005")


# RC-12
def test_float_amounts_rounded_to_paisa():
    for _ in range(3):
        redis_client.reserve_daily_amount("C1", "TRANSFER", 0.1, 50000, NOON_UTC)
    assert redis_client.get_daily_total("C1", "TRANSFER", NOON_UTC) == 0.3


# RC-13
def test_redis_down_fails_open(redis_down):
    assert redis_client.get_daily_total("C1", "TRANSFER", NOON_UTC) is None
    assert redis_client.reserve_daily_amount("C1", "TRANSFER", 1000, 50000, NOON_UTC) is None
    redis_client.release_daily_amount("C1", "TRANSFER", 1000, NOON_UTC)  # must not raise


# RC-14
def test_concurrent_reserves_never_exceed_limit():
    results = []
    lock = threading.Lock()

    def worker():
        result = redis_client.reserve_daily_amount("C1", "TRANSFER", 5000, 50000, NOON_UTC)
        with lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    accepted = [r for r in results if r[0]]
    assert len(accepted) == 10
    assert redis_client.get_daily_total("C1", "TRANSFER", NOON_UTC) == 50000.0
