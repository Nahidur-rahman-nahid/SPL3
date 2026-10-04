# Test Cases: Regulatory Daily Limits & Night-Time High-Value Rule

**Project:** GNN-based Real-Time Fraud Detection for Mobile Financial Services (SPL3)
**Features under test:**
1. **Bangladesh Bank daily limits:** a transaction that pushes the sender's daily total for its type over the limit is hard-blocked (`decision=BLOCK`, `risk_level=REGULATORY`). Limits: CASH_OUT BDT 30,000/day, TRANSFER (P2P send money) BDT 50,000/day.
2. **Night-time high-value rule:** a transaction over BDT 10,000 between 23:00 and 03:59 Bangladesh time (UTC+6) is held for `REVIEW`, whatever the GNN score.

**Test date:** 2026-10-05
**Result:** **82 / 82 passed** (70 test cases; some cases run with several inputs)

---

## 1. Test Environment

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.12.5 |
| Test framework | pytest 9.1.1 |
| Database | SQLite (temporary test file; the production database is never touched) |
| Redis | fakeredis 2.37 (in-memory Redis replacement) |
| ML model | Fake model with a fixed fraud probability for rule tests; the **real** `FraudGNN_main.pt` checkpoint for smoke tests (Suite 5) |
| HTTP | FastAPI `TestClient`; login and CSRF checks bypassed except in API-08 |

**How to run**

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests -v
```

## 2. Business Rules Being Verified

| Rule | Description |
|---|---|
| BR-1 | A sender's same-type total for the current Bangladesh calendar day may not exceed the daily limit (CASH_OUT 30,000; TRANSFER 50,000). A transaction that reaches the limit **exactly** is allowed. |
| BR-2 | A limit breach gives `BLOCK / REGULATORY`. The GNN still runs, so a real fraud probability is stored. |
| BR-3 | Only money that goes through counts: ALLOW and REVIEW are counted; any BLOCK (model or regulatory) is not. |
| BR-4 | Daily totals reset at **midnight Bangladesh time** (UTC+6), not midnight UTC. |
| BR-5 | A transaction over BDT 10,000 between 23:00 and 03:59 local time is raised from ALLOW to REVIEW and sent to the analyst alert queue. |
| BR-6 | The night rule never lowers a decision (an existing REVIEW or BLOCK is kept). |
| BR-7 | A regulatory block takes priority over the night rule. |
| BR-8 | Both rules apply to `POST /score` only. The live stream simulator is unaffected. |
| BR-9 | If Redis is unavailable, the limit check is skipped (fail open) and scoring continues. |
| BR-10 | Both features can be switched off by configuration (`REGULATORY_LIMITS_ENABLED`, `NIGHT_RULE_ENABLED`). |

## 3. Summary

| Suite | File | Cases | Executed | Passed |
|---|---|---|---|---|
| 1. Night-time rule (unit) | `tests/test_night_rule.py` | 20 | 30 | 30 |
| 2. Daily limit counters (unit) | `tests/test_daily_limit_redis.py` | 14 | 14 | 14 |
| 3. Scoring pipeline integration | `tests/test_scoring_pipeline_rules.py` | 25 | 27 | 27 |
| 4. REST API | `tests/test_score_api.py` | 8 | 8 | 8 |
| 5. Real-model smoke | `tests/test_real_model_smoke.py` | 3 | 3 | 3 |
| **Total** | | **70** | **82** | **82** |

---

## Suite 1: Night-Time High-Value Rule (Unit)

Unless stated otherwise: model decision = ALLOW/LOW, type = TRANSFER.

| ID | Test case | Input | Expected result | Status |
|---|---|---|---|---|
| NR-01 | High-value transaction at night | BDT 15,000 at 23:30 | REVIEW / MEDIUM, rule `NIGHT_HIGH_VALUE` recorded, `changed_decision=true` | Pass |
| NR-02 | Every hour in the night window triggers | BDT 15,000 at 23:15, 00:15, 01:15, 02:15, 03:15 | REVIEW for each of the 5 hours | Pass |
| NR-03 | Times outside the window don't trigger | BDT 50,000 at 22:59:59, 04:00, 12:00, 18:30 | Stays ALLOW, rule not triggered | Pass |
| NR-04 | Window edges are included | BDT 15,000 at 23:00:00 and 03:59:59 | REVIEW | Pass |
| NR-05 | Amount exactly at threshold | BDT 10,000 at 01:00 | ALLOW (rule needs amount > 10,000) | Pass |
| NR-06 | Amount just above threshold | BDT 10,000.01 at 01:00 | REVIEW | Pass |
| NR-07 | Low amount at night | BDT 500 at 02:00 | ALLOW, not triggered | Pass |
| NR-08 | Model already said REVIEW | BDT 15,000 at 23:30, model REVIEW | Stays REVIEW; rule recorded with `changed_decision=false` | Pass |
| NR-09 | BLOCK never lowered | BDT 15,000 at 23:30, model BLOCK HIGH / CRITICAL | Stays BLOCK with the same risk level | Pass |
| NR-10 | UTC time converted to Bangladesh time | `20:00Z` (= 02:00 BD); `12:00Z` (= 18:00 BD) | First → REVIEW (local hour 2); second → not triggered | Pass |
| NR-11 | Explicit +06:00 offset | `23:10+06:00` | REVIEW | Pass |
| NR-12 | Time without timezone is treated as local | `02:00` (no timezone) | REVIEW (read as 02:00 BD, not 02:00 UTC) | Pass |
| NR-13 | No `transaction_time` → current BD time used | Clock set to 18:00Z / 21:00Z / 06:00Z | REVIEW / REVIEW / ALLOW (00:00, 03:00, 12:00 BD) | Pass |
| NR-14 | Rule switched off | `NIGHT_RULE_ENABLED=False`, BDT 50,000 at 23:30 | ALLOW, not triggered | Pass |
| NR-15 | Custom hours and threshold | hours=`22`, min=1,000; BDT 1,500 at 22:30 and 23:30 | 22:30 → REVIEW; 23:30 → ALLOW | Pass |
| NR-16 | Applies to CASH_OUT | CASH_OUT BDT 15,000 at 23:30 | REVIEW | Pass |
| NR-17 | Explanation text when decision changed | Rule raised ALLOW to REVIEW | Summary starts "Held for REVIEW by night-time rule: BDT 15,000 at 23:00…", original summary kept after it; `rules_triggered` lists the rule | Pass |
| NR-18 | Explanation when decision not changed | Rule triggered on a BLOCK | Summary unchanged; `rules_triggered` present with `changed_decision=false` | Pass |
| NR-19 | Request without `transaction_time` (backward compatible) | Old-style request | Valid; `transaction_time = None` | Pass |
| NR-20 | Invalid `transaction_time` | `"not-a-date"` | Rejected with a validation error | Pass |

## Suite 2: Daily Limit Counters in Redis (Unit)

Time is fixed at 12:00 Bangladesh time on 2026-10-04 unless stated otherwise.

| ID | Test case | Input | Expected result | Status |
|---|---|---|---|---|
| RC-01 | Account with no transactions today | Fresh account | Total = 0.0 | Pass |
| RC-02 | Reserve within limit | 20,000 against 50,000 | Accepted, total 20,000 | Pass |
| RC-03 | Totals add up | 20,000 then 15,000 | Accepted, total 35,000 | Pass |
| RC-04 | Reaching the limit exactly | 30,000 then 20,000 (limit 50,000) | Accepted, total 50,000 | Pass |
| RC-05 | Going over the limit | 40,000 then 10,000.01 | Rejected; total stays 40,000 (rolled back) | Pass |
| RC-06 | Single amount over limit, no history | CASH_OUT 30,001 against 30,000 | Rejected; total 0 | Pass |
| RC-07 | Release undoes a reserve | Reserve 20,000, then release 20,000 | Total 0 | Pass |
| RC-08 | Key format and expiry | Reserve 100 | Key `daily:C1:TRANSFER:20261004` exists; 0 < TTL ≤ 48h | Pass |
| RC-09 | Types counted separately | TRANSFER 45,000 then CASH_OUT 25,000 | CASH_OUT accepted (own total 25,000) | Pass |
| RC-10 | Accounts counted separately | C1 50,000, then C2 50,000 | C2 accepted | Pass |
| RC-11 | Reset at Bangladesh midnight | 50,000 at 23:59 BD on Oct 4, then 50,000 at 00:01 BD on Oct 5 (still Oct 4 in UTC) | Second accepted under key `…:20261005` | Pass |
| RC-12 | Decimal rounding | 0.1 × 3 | Total exactly 0.30 | Pass |
| RC-13 | Redis unavailable | Redis connection down | get → None, reserve → None, release raises no error (fail open) | Pass |
| RC-14 | Simultaneous requests | 25 threads at once, each reserving 5,000 against 50,000 | Exactly 10 accepted; total exactly 50,000 (never over the limit) | Pass |

## Suite 3: Scoring Pipeline Integration

Real database (SQLite) and fakeredis, through `process_transaction()`. Model probability 0.05 (ALLOW) and daytime unless stated. Sender `C_SENDER_1`, TRANSFER.

| ID | Test case | Input | Expected result | Status |
|---|---|---|---|---|
| RL-01 | First transaction of the day | 20,000 | ALLOW; `limit_usage` = {TRANSFER, daily, used 20,000, limit 50,000, remaining 30,000} | Pass |
| RL-02 | Up to the limit exactly | 30,000 then 20,000 | ALLOW; used 50,000, remaining 0 | Pass |
| RL-03 | Over the TRANSFER limit | 42,000 then 15,000 | BLOCK / REGULATORY; violation {BB_DAILY_TRANSFER_LIMIT, limit 50,000, used today 42,000, attempted 15,000}; summary starts "Blocked by Bangladesh Bank daily TRANSFER limit"; total stays 42,000 | Pass |
| RL-04 | CASH_OUT limit is 30,000 | CASH_OUT 30,000 then 0.01 | First ALLOW; second REGULATORY; limit shown as 30,000 | Pass |
| RL-05 | Single transaction over limit | 50,000.01 | REGULATORY; total 0 | Pass |
| RL-06 | Smaller transaction still allowed after a block | 45,000 → 10,000 (blocked) → 5,000 | Last one ALLOW; used 50,000 | Pass |
| RL-07 | Model BLOCK not counted | Probability 0.95, 40,000 | BLOCK / CRITICAL; used 0 (amount released) | Pass |
| RL-08 | Model REVIEW counted | Probability 0.5, 40,000 | REVIEW; total 40,000 | Pass |
| RL-09 | Night-rule REVIEW counted | 15,000 at 23:30 | REVIEW; total 15,000 | Pass |
| RL-10 | Model still runs on a regulatory block; row saved | Probability 0.12, 60,000 | Model called; stored row = BLOCK / REGULATORY / 0.12; transaction row exists | Pass |
| RL-11 | Alert on regulatory block follows the model score | 60,000 with probability 0.05, and with 0.6 | No alert at 0.05; 1 alert at 0.6 (existing alert threshold 0.40) | Pass |
| RL-12 | Stream simulator skips the limit check | `source="stream"`, 200,000 | ALLOW (model decides); `limit_usage = null`; total unchanged | Pass |
| RL-13 | Stream simulator skips the night rule | `source="stream"`, 15,000 at 23:30 | ALLOW; no `rules_triggered` | Pass |
| RL-14 | Limits switched off | `REGULATORY_LIMITS_ENABLED=False`, 200,000 | ALLOW; `limit_usage = null` | Pass |
| RL-15 | Redis down → fail open | Redis offline, 200,000 | Scored normally (ALLOW); `limit_usage = null` | Pass |
| RL-16 | Same transaction scored again (allowed) | `dup-1` 30,000 sent twice | Both ALLOW; total 30,000 (not 60,000) | Pass |
| RL-17 | Same transaction scored again (blocked) | 45,000, then `dup-2` 10,000 twice | `dup-2` stays REGULATORY both times; total 45,000 | Pass |
| RL-18 | Zero or negative amount | 20,000, then 0 / −5,000 | Total stays 20,000 (can't lower the counter) | Pass |
| RL-19 | Scoring failure after reserving | Model raises an error, 20,000 | Error passed to the caller; reservation released (total 0) | Pass |
| RL-20 | Limit applies to the sender only | A→B 50,000, then B→A 50,000 | B's total unaffected by receiving; B→A ALLOW | Pass |
| RL-21 | Regulatory block beats the night rule | 60,000 at 23:30 | REGULATORY; night rule not applied | Pass |
| RL-22 | Night rule creates an alert despite a low model score | Probability 0.05, 15,000 at 23:30 | REVIEW / MEDIUM saved; 1 alert created; explanation lists `NIGHT_HIGH_VALUE` | Pass |
| RL-23 | Daytime, low score: no alert (unchanged behaviour) | 15,000 at 12:00 | ALLOW; no alert | Pass |
| RL-24 | Limit taken from config | `DAILY_LIMIT_TRANSFER=1000`, 1,001 | REGULATORY | Pass |
| RL-25 | Regulatory decision sent to the live dashboard | 60,000 | WebSocket message with decision BLOCK, risk REGULATORY | Pass |

## Suite 4: REST API (`POST /score`, `GET /api/fraud/results`)

| ID | Test case | Input | Expected result | Status |
|---|---|---|---|---|
| API-01 | Response includes limit usage | POST 20,000 | HTTP 200; `limit_usage` used 20,000, remaining 30,000 | Pass |
| API-02 | Regulatory block over HTTP | POST 40,000, then 20,000 | Second: BLOCK / REGULATORY; remaining 10,000 | Pass |
| API-03 | Night rule over HTTP | POST 15,000 with `transaction_time` 23:30 | REVIEW; `rules_triggered[0].rule = NIGHT_HIGH_VALUE` | Pass |
| API-04 | Invalid date | `transaction_time = "31-31-2026"` | HTTP 422 | Pass |
| API-05 | Old client payload still works | No `transaction_time` | HTTP 200; all original response fields present | Pass |
| API-06 | Results can be filtered by REGULATORY | Score one blocked and one normal; GET `?risk_level=regulatory` | Only the blocked transaction returned | Pass |
| API-07 | Unsupported type still rejected | type = PAYMENT | HTTP 422 (existing validation unchanged) | Pass |
| API-08 | Login still required | POST without a session | HTTP 401/403 | Pass |

## Suite 5: Smoke Tests with the Real GNN Model

Uses the real `FraudGNN_main.pt`, `scaler.pkl` and `type_encoder.pkl`.

| ID | Test case | Input | Expected result | Status |
|---|---|---|---|---|
| SM-01 | Real inference plus limit usage | 5,000 TRANSFER | Probability between 0 and 1; `limit_usage.limit = 50,000` | Pass |
| SM-02 | Regulatory block with the real model | 60,000 TRANSFER | BLOCK / REGULATORY; real probability between 0 and 1 stored | Pass |
| SM-03 | Night rule never lowers the real model's decision | 15,000: once in daytime (stream), once at 23:30 (API) | Night decision is at least REVIEW and at least the daytime decision | Pass |

---

## 4. Checking That the Tests Can Fail

To make sure the tests detect real faults, the suite was run again with each feature switched off:

| Run | Result |
|---|---|
| `REGULATORY_LIMITS_ENABLED=False` | 24 failed, 58 passed. The limit tests fail as expected. |
| `NIGHT_RULE_ENABLED=False` | 24 failed, 58 passed. The night-rule tests fail as expected. |
| Both on (normal) | **82 passed** |

## 5. Known Limitations / Not Covered

- The limit date comes from the **server clock**, not the client's `transaction_time`. This is deliberate, so a caller can't back-date a transaction to get around the limit. The night rule uses `transaction_time` because it is about when the transaction happened.
- If a transaction is scored again after being blocked **by the model** (not by the limit), it was never counted, so it is not counted on the re-score either.
- The monthly P2P limit (BDT 300,000) and the maximum wallet balance (BDT 500,000) are not implemented.
- The frontend display of `limit_usage` in `ScoreForm.js` was checked by code review only (no automated UI test).
- Tests used fakeredis, not the hosted Upstash Redis.
