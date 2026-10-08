"""Stream simulator — realistic everyday traffic + grouped fraud
(stream_simulator.py). Pure generator tests: nothing is scored."""

import random

import pytest

import stream_simulator as sim


@pytest.fixture(autouse=True)
def fresh_state():
    sim._balance.clear()
    sim._pending.clear()
    sim._ring.update(mule=None, left=0)
    random.seed(7)
    yield


def generate(n):
    return [sim.generate_transaction() for _ in range(n)]


# SS-01
def test_every_transaction_balances():
    for req in generate(500):
        f = req.features
        assert f.amount > 0
        assert f.newbalanceOrig == pytest.approx(f.oldbalanceOrg - f.amount, abs=0.02)
        assert f.newbalanceDest == pytest.approx(f.oldbalanceDest + f.amount, abs=0.02)
        assert min(f.newbalanceOrig, f.oldbalanceDest) >= 0


# SS-02
def test_balances_chain_between_transactions():
    last_after = {}
    for req in generate(500):
        f = req.features
        if req.sender_account in last_after and not req.sender_account.startswith("C_CUST"):
            assert f.oldbalanceOrg == pytest.approx(last_after[req.sender_account])
        if req.sender_account in last_after and req.sender_account.startswith("C_CUST"):
            assert f.oldbalanceOrg >= last_after[req.sender_account] - 0.01  # may be topped up (salary)
        last_after[req.sender_account] = f.newbalanceOrig
        last_after[req.receiver_account] = f.newbalanceDest


# SS-03
def test_structuring_burst_is_consecutive_and_under_limit():
    sim._queue_structuring()
    burst = [sim.generate_transaction() for _ in range(len(sim._pending))]
    assert 3 <= len(burst) <= 5
    assert len({(r.sender_account, r.receiver_account) for r in burst}) == 1
    assert all(45_000 <= r.features.amount < 50_000 for r in burst)


# SS-04
def test_drains_share_a_mule_and_empty_the_victim():
    drains = [sim._drain_transaction() for _ in range(12)]
    assert all(d.features.newbalanceOrig == 0 for d in drains)
    mules = [d.receiver_account for d in drains]
    assert len(set(mules)) < len(mules)  # several victims per mule
    assert len({d.sender_account for d in drains}) == len(drains)  # every victim is fresh


# SS-05
def test_everyday_traffic_is_the_majority_and_uses_recurring_accounts():
    reqs = generate(1000)
    everyday = [r for r in reqs if r.sender_account.startswith("C_CUST")]
    assert len(everyday) > 0.7 * len(reqs)
    assert {r.sender_account for r in everyday} <= set(sim.CUSTOMERS)
    family = [r for r in everyday if r.receiver_account.startswith("C_FAMILY")]
    # each customer always pays the same relative
    assert all(len({r.receiver_account for r in family if r.sender_account == c}) <= 1 for c in sim.CUSTOMERS)


# SS-06
def test_ambiguous_pairs():
    pooled = new_payees = 0
    for _ in range(40):
        sim._pending.clear()
        sim._queue_ambiguous()
        pair = [sim.generate_transaction() for _ in range(2)]
        assert not sim._pending
        if pair[0].receiver_account == pair[1].receiver_account:  # pooled money
            pooled += 1
            assert pair[0].sender_account != pair[1].sender_account
            assert all(0.85 <= r.features.amount / r.features.oldbalanceOrg <= 0.95 for r in pair)
        else:  # one payer, two brand-new payees
            new_payees += 1
            assert pair[0].sender_account == pair[1].sender_account
            assert all(r.features.oldbalanceDest == 0 for r in pair)
            assert all(0.28 <= r.features.amount / r.features.oldbalanceOrg <= 0.34 for r in pair)
    assert pooled and new_payees


# SS-07
def test_step_is_fixed():
    assert {r.features.step for r in generate(50)} == {sim.STREAM_STEP}
