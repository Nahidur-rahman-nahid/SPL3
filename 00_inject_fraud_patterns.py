"""
Inject graph-dependent fraud patterns — and a legitimate look-alike for
each — into PaySim, so the GNN has to learn that WHO a transaction is
connected to matters, not just the transaction's own columns.

Why this is needed (measured on PaySim's TRANSFER + CASH_OUT rows):
  - a legit transaction's receiver is shared with other transactions 95.5%
    of the time (median 9 transactions per receiver) vs 62.2% for fraud
    (median 3) — so the trained model learned "more neighbours = more
    normal", and connecting a transaction to known fraud LOWERED its score;
  - senders almost never repeat (0.1-0.2%), so sender-side patterns like
    structuring or a takeover burst don't exist in PaySim at all.

Every fraud pattern below puts fraud next to fraud (so neighbours push the
score UP). Two kinds of legitimate twin keep the model honest:
  - group twins with the same overall shape (payroll vs smurfing, busy
    agent vs collusive agent, ...), so "being in a group" isn't fraud;
  - isolated twins: an exact copy of every row of the five graph-only
    patterns, same numbers, but sent between unrelated accounts with no
    neighbours. Row by row, fraud and twin are identical — a tabular model
    (XGBoost) cannot tell them apart; only the graph can.
All rows of one instance fall inside 03_graph_builder.py's 24-step window,
so they really become connected in the graph.

Patterns (see INJECTED_PATTERNS.md for the full list and how to verify
the retrained GNN learned each one):
  fraud: MULE_HUB, STRUCTURING, SMURF_FANOUT, AGENT_COLLUSION,
         ACCOUNT_TAKEOVER, MULE_NETWORK
  legit group twins: LEGIT_PAYROLL, LEGIT_BUSY_AGENT, LEGIT_BILL_PAYMENTS,
         LEGIT_SHARED_PAYEES (+ the owner's normal history in ACCOUNT_TAKEOVER)
  legit small-account groups: LEGIT_FAMILY_SUPPORT, LEGIT_FAMILY_FANOUT,
         LEGIT_SMALL_SHOP
  legit isolated twins: LEGIT_NEAR_LIMIT_SINGLE, LEGIT_SINGLE_NEW_PAYEE,
         LEGIT_SINGLE_FULL_CASHOUT, LEGIT_SINGLE_LARGE_TRANSFER,
         LEGIT_SINGLE_PARTIAL_TRANSFER

Output (Dataset/, never modifies the original CSV):
  PS_augmented.csv       original PaySim + injected rows, same columns
  injected_manifest.csv  one line per injected row: pattern, instance,
                         role, isFraud, holdout (20% of instances, for a
                         strict "never seen in training" evaluation)

Run locally:  python 00_inject_fraud_patterns.py      (deterministic, seed 42)
Run in Colab (CSV already in the working folder, writes next to it):
      python 00_inject_fraud_patterns.py --src PS_20174392719_1491204439457_log.csv --out-dir .
01_load_paysim.py then reads PS_augmented.csv + injected_manifest.csv.
"""

import argparse
import os
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

SEED = 42
DEFAULT_SRC = "Dataset/PS_20174392719_1491204439457_log.csv"
DEFAULT_OUT_DIR = "Dataset"

MAX_STEP = 743      # PaySim covers steps 1..743 (hours)
HOLDOUT_SHARE = 0.2

# Instances per pattern — sized so the injected fraud (~4.8k rows) stays a
# minority of the fraud class (PaySim has 8,213 fraud TRANSFER/CASH_OUT
# rows) and the whole injection is well under 1% of all rows.
COUNTS = {
    "MULE_HUB": 70,
    "STRUCTURING": 175,
    "SMURF_FANOUT": 70,
    "AGENT_COLLUSION": 45,
    "ACCOUNT_TAKEOVER": 110,
    "MULE_NETWORK": 55,
    # Group twins are sized to roughly match the fraud groups row for row,
    # so "belongs to a connected injected group" can't become the new
    # shortcut (with half as many, the model flagged 83-100% of them).
    "LEGIT_PAYROLL": 100,
    "LEGIT_BUSY_AGENT": 80,
    "LEGIT_BILL_PAYMENTS": 330,
    "LEGIT_SHARED_PAYEES": 110,
    # Everyday groups around SMALL-balance accounts. Without these, every
    # small or empty account receiving several transfers in the training
    # data was a mule, so the model flagged a customer paying the same
    # relative or corner shop twice (75% of the time in the live stream).
    "LEGIT_FAMILY_SUPPORT": 150,
    "LEGIT_FAMILY_FANOUT": 130,
    "LEGIT_SMALL_SHOP": 60,
}

# Graph-only fraud pattern -> its isolated twin (one exact, unconnected copy
# of every fraud row). MULE_HUB has none: its rows are full drains, the
# classic PaySim fraud signature, which should stay detectable row by row.
ISOLATED_TWINS = {
    "STRUCTURING": "LEGIT_NEAR_LIMIT_SINGLE",
    "SMURF_FANOUT": "LEGIT_SINGLE_NEW_PAYEE",
    "AGENT_COLLUSION": "LEGIT_SINGLE_FULL_CASHOUT",
    "ACCOUNT_TAKEOVER": "LEGIT_SINGLE_LARGE_TRANSFER",
    "MULE_NETWORK": "LEGIT_SINGLE_PARTIAL_TRANSFER",
}

COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]

rng = np.random.default_rng(SEED)


def u(lo, hi):
    return float(rng.uniform(lo, hi))


def n_between(lo, hi):
    """Integer in [lo, hi] inclusive."""
    return int(rng.integers(lo, hi + 1))


def base_step(span):
    """A start step leaving room for `span` more steps before MAX_STEP."""
    return int(rng.integers(1, MAX_STEP - span + 1))


class Ledger:
    """Account balances plus the rows written so far. PaySim account ids
    are at most 11 characters ("C" + 10 digits); injected ones are "C" +
    11 digits, so they can never collide with a real PaySim account."""

    def __init__(self):
        self.balance = {}
        self.rows = []
        self.meta = []
        self._next_id = int(rng.integers(10_000_000_000, 20_000_000_000))

    def _new_name(self):
        name = f"C{self._next_id:011d}"
        self._next_id += int(rng.integers(1, 9_973))
        return name

    def open(self, balance):
        name = self._new_name()
        self.balance[name] = round(balance, 2)
        return name

    def clone_isolated(self, source_pattern, twin_pattern):
        """Copy every fraud row of `source_pattern` with identical numbers
        (step, type, amount, all four balances) but a brand-new sender and
        receiver each, labelled legit. Each copy is its own one-row instance
        with no neighbours; it inherits the holdout flag of the instance it
        was copied from, so a held-out pattern keeps a held-out twin."""
        count = 0
        for row, (pattern, _, _, is_fraud, holdout) in list(zip(self.rows, self.meta)):
            if pattern != source_pattern or is_fraud != 1:
                continue
            count += 1
            step, txn_type, amount, _, old_org, new_org, _, old_dest, new_dest, _, flagged = row
            self.rows.append((step, txn_type, amount, self._new_name(), old_org, new_org,
                              self._new_name(), old_dest, new_dest, 0, flagged))
            self.meta.append((twin_pattern, f"{twin_pattern}-{count:04d}", "isolated_lookalike", 0, holdout))

    def emit(self, events, pattern, instance_id, holdout):
        """events: (step, type, sender, receiver, amount, isFraud, role).
        Applied in step order so every balance chains correctly."""
        for step, txn_type, sender, receiver, amount, is_fraud, role in sorted(events, key=lambda e: e[0]):
            amount = round(min(amount, self.balance[sender]), 2)
            old_org, old_dest = self.balance[sender], self.balance[receiver]
            new_org, new_dest = round(old_org - amount, 2), round(old_dest + amount, 2)
            self.balance[sender], self.balance[receiver] = new_org, new_dest
            self.rows.append((step, txn_type, amount, sender, old_org, new_org, receiver, old_dest, new_dest, is_fraud, 0))
            self.meta.append((pattern, instance_id, role, is_fraud, holdout))


ledger = Ledger()
instance_counter = defaultdict(int)


def new_instance(pattern):
    instance_counter[pattern] += 1
    return f"{pattern}-{instance_counter[pattern]:04d}", bool(rng.random() < HOLDOUT_SHARE)


def split_weights(n, spread=0.3):
    w = rng.uniform(1 - spread, 1 + spread, n)
    return w / w.sum()


# ------------------------------------------------------------------ fraud

def mule_hub():
    """8-15 victims each (mostly fully) drained into ONE fresh mule within
    6 hours; the mule then cashes everything out at an agent. Fan-in:
    linked by SAME_RECEIVER, every neighbour is itself a drain."""
    instance_id, holdout = new_instance("MULE_HUB")
    mule, agent = ledger.open(0.0), ledger.open(u(50_000, 2_000_000))
    b = base_step(10)
    events, total = [], 0.0
    for step in sorted(b + rng.integers(0, 7, n_between(8, 15))):
        victim = ledger.open(u(20_000, 300_000))
        frac = 1.0 if rng.random() < 0.85 else u(0.70, 0.95)
        amount = ledger.balance[victim] * frac
        total += amount
        events.append((int(step), "TRANSFER", victim, mule, amount, 1, "victim_transfer"))
    last = max(e[0] for e in events)
    n_out = n_between(1, 2)
    for i, part in enumerate(split_weights(n_out)):
        amount = total * part if i < n_out - 1 else total  # last one empties the mule
        events.append((last + 1 + i, "CASH_OUT", mule, agent, amount, 1, "mule_cashout"))
    ledger.emit(events, "MULE_HUB", instance_id, holdout)


def structuring():
    """One sender makes 3-6 transfers of BDT 40,000-49,999 within 2 hours —
    each just under the BB 50,000 daily limit — while keeping a healthy
    balance. One such transfer alone is normal (see LEGIT_NEAR_LIMIT_SINGLE);
    only the SAME_SENDER group reveals it."""
    instance_id, holdout = new_instance("STRUCTURING")
    n = n_between(3, 6)
    amounts = [u(40_000, 49_999) for _ in range(n)]
    sender = ledger.open(sum(amounts) / u(0.40, 0.85))
    receivers = [ledger.open(0.0 if rng.random() < 0.5 else u(5_000, 300_000)) for _ in range(n_between(1, 2))]
    b = base_step(3)
    events = [
        (int(step), "TRANSFER", sender, receivers[i % len(receivers)], amt, 1, "near_limit_split")
        for i, (step, amt) in enumerate(zip(sorted(b + rng.integers(0, 3, n)), amounts))
    ]
    ledger.emit(events, "STRUCTURING", instance_id, holdout)


def smurf_fanout():
    """One source spreads 60-95% of its balance over 8-20 FRESH accounts
    within 3 hours (placement/layering). Each transfer is a modest share of
    the balance; the fan-out to brand-new accounts is the signal. Twin:
    LEGIT_PAYROLL."""
    instance_id, holdout = new_instance("SMURF_FANOUT")
    source = ledger.open(u(150_000, 1_500_000))
    n = n_between(8, 20)
    total = ledger.balance[source] * u(0.60, 0.95)
    b = base_step(4)
    events = [
        (int(step), "TRANSFER", source, ledger.open(0.0), total * w, 1, "fanout_to_fresh")
        for step, w in zip(b + rng.integers(0, 4, n), split_weights(n))
    ]
    ledger.emit(events, "SMURF_FANOUT", instance_id, holdout)


def agent_collusion():
    """10-25 near-empty, recently-funded accounts each cash out (almost)
    everything at the SAME agent within 6 hours. A small full-balance
    cash-out alone is common and legit; many of them converging on one
    agent is the signal. Twin: LEGIT_BUSY_AGENT."""
    instance_id, holdout = new_instance("AGENT_COLLUSION")
    agent = ledger.open(u(0, 50_000))
    b = base_step(7)
    events = []
    for step in b + rng.integers(0, 7, n_between(10, 25)):
        customer = ledger.open(u(5_000, 30_000))
        frac = 1.0 if rng.random() < 0.6 else u(0.90, 0.99)
        events.append((int(step), "CASH_OUT", customer, agent, ledger.balance[customer] * frac, 1, "collusive_cashout"))
    ledger.emit(events, "AGENT_COLLUSION", instance_id, holdout)


def account_takeover():
    """An established account makes 2-4 small normal payments (legit,
    label 0), then within ~1 hour sends 80-100% of what's left to 3-5
    brand-new accounts (label 1). Each burst transfer is only 15-35% of the
    balance; the sudden SAME_SENDER burst is the signal."""
    instance_id, holdout = new_instance("ACCOUNT_TAKEOVER")
    owner = ledger.open(u(80_000, 800_000))
    b = base_step(20)
    events, spent = [], 0.0
    for step in sorted(b + rng.integers(0, 12, n_between(2, 4))):
        amount = ledger.balance[owner] * u(0.01, 0.08)
        spent += amount
        events.append((int(step), "TRANSFER", owner, ledger.open(u(5_000, 500_000)), amount, 0, "owner_history"))
    remaining = ledger.balance[owner] - spent
    burst_step = b + n_between(13, 19)
    total = remaining * u(0.80, 1.0)
    m = n_between(3, 5)
    for i, w in enumerate(split_weights(m)):
        events.append((burst_step + int(i >= 3), "TRANSFER", owner, ledger.open(0.0), total * w, 1, "takeover_burst"))
    ledger.emit(events, "ACCOUNT_TAKEOVER", instance_id, holdout)


def mule_network():
    """A coordinated ring: 5-9 victims each send 15-40% of their balance to
    2-3 mules picked from a shared set of 3-5 fresh mules, within 4 hours.
    No victim is drained, so single rows look like normal partial
    transfers; the dense victim<->mule structure (SAME_SENDER +
    SAME_RECEIVER, 2 hops) is the signal. Twin: LEGIT_SHARED_PAYEES."""
    instance_id, holdout = new_instance("MULE_NETWORK")
    mules = [ledger.open(0.0) for _ in range(n_between(3, 5))]
    b = base_step(5)
    events = []
    for _ in range(n_between(5, 9)):
        victim = ledger.open(u(30_000, 400_000))
        targets = rng.choice(len(mules), size=min(n_between(2, 3), len(mules)), replace=False)
        for t in targets:
            events.append((int(b + rng.integers(0, 5)), "TRANSFER", victim, mules[t],
                           ledger.balance[victim] * u(0.15, 0.30), 1, "ring_transfer"))
    ledger.emit(events, "MULE_NETWORK", instance_id, holdout)


# ---------------------------------------------------- legit group twins

def legit_payroll():
    """Twin of SMURF_FANOUT: an employer pays 8-20 ESTABLISHED employee
    accounts similar salaries in one go and keeps most of its balance."""
    instance_id, holdout = new_instance("LEGIT_PAYROLL")
    n = n_between(8, 20)
    salary = u(15_000, 60_000)
    amounts = [salary * u(0.9, 1.1) for _ in range(n)]
    employer = ledger.open(sum(amounts) / u(0.10, 0.30))
    b = base_step(2)
    events = [
        (int(b + rng.integers(0, 2)), "TRANSFER", employer, ledger.open(u(2_000, 80_000)), amt, 0, "salary_payment")
        for amt in amounts
    ]
    ledger.emit(events, "LEGIT_PAYROLL", instance_id, holdout)


def legit_busy_agent():
    """Twin of AGENT_COLLUSION: a busy agent serving 10-25 established
    customers who each withdraw a small part of their balance over a day."""
    instance_id, holdout = new_instance("LEGIT_BUSY_AGENT")
    agent = ledger.open(u(200_000, 5_000_000))
    b = base_step(21)
    events = []
    for step in b + rng.integers(0, 21, n_between(10, 25)):
        customer = ledger.open(u(20_000, 500_000))
        events.append((int(step), "CASH_OUT", customer, agent, ledger.balance[customer] * u(0.05, 0.40), 0, "routine_cashout"))
    ledger.emit(events, "LEGIT_BUSY_AGENT", instance_id, holdout)


def legit_bill_payments():
    """Twin of ACCOUNT_TAKEOVER: 2-4 ordinary payments spread over a day to
    established payees, small shares of the balance."""
    instance_id, holdout = new_instance("LEGIT_BILL_PAYMENTS")
    sender = ledger.open(u(50_000, 800_000))
    b = base_step(21)
    events = [
        (int(step), "TRANSFER", sender, ledger.open(u(5_000, 500_000)), ledger.balance[sender] * u(0.01, 0.10), 0, "bill_payment")
        for step in b + rng.integers(0, 21, n_between(2, 4))
    ]
    ledger.emit(events, "LEGIT_BILL_PAYMENTS", instance_id, holdout)


def legit_shared_payees():
    """Twin of MULE_NETWORK: 5-9 people paying 1-2 shared ESTABLISHED payees
    (landlord, school) small shares of their balance over a day."""
    instance_id, holdout = new_instance("LEGIT_SHARED_PAYEES")
    payees = [ledger.open(u(50_000, 2_000_000)) for _ in range(n_between(2, 3))]
    b = base_step(21)
    events = []
    for _ in range(n_between(5, 9)):
        sender = ledger.open(u(20_000, 400_000))
        for t in rng.choice(len(payees), size=n_between(1, 2), replace=False):
            events.append((int(b + rng.integers(0, 21)), "TRANSFER", sender, payees[t],
                           ledger.balance[sender] * u(0.02, 0.15), 0, "shared_payee_payment"))
    ledger.emit(events, "LEGIT_SHARED_PAYEES", instance_id, holdout)


def small_payment(balance, lo_share, hi_share):
    """BDT 500-15,000, but never more than hi_share of the balance."""
    return max(min(u(500, 15_000), balance * u(lo_share, hi_share)), min(500.0, balance * hi_share))


def legit_family_support():
    """Twin of STRUCTURING / MULE_HUB on the small-account side: someone
    sends a low-balance relative (often empty) 2-5 small amounts over a
    day. Same sender, same receiver, repeated — but small sums, a small
    share of the sender's balance, nowhere near the BB limit."""
    instance_id, holdout = new_instance("LEGIT_FAMILY_SUPPORT")
    sender = ledger.open(u(15_000, 200_000))
    relative = ledger.open(0.0 if rng.random() < 0.3 else u(200, 15_000))
    b = base_step(21)
    events = [
        (int(step), "TRANSFER", sender, relative, small_payment(ledger.balance[sender], 0.02, 0.15), 0, "family_transfer")
        for step in b + rng.integers(0, 21, n_between(2, 5))
    ]
    ledger.emit(events, "LEGIT_FAMILY_SUPPORT", instance_id, holdout)


def legit_family_fanout():
    """Twin of SMURF_FANOUT: one person sends 2-5 low-balance relatives a
    small amount each (Eid salami, monthly support) within a few hours —
    5-25% of the balance in total, not 60-95% to 8-20 brand-new accounts."""
    instance_id, holdout = new_instance("LEGIT_FAMILY_FANOUT")
    sender = ledger.open(u(20_000, 300_000))
    b = base_step(6)
    events = []
    for step in b + rng.integers(0, 6, n_between(2, 5)):
        relative = ledger.open(0.0 if rng.random() < 0.3 else u(200, 20_000))
        events.append((int(step), "TRANSFER", sender, relative, small_payment(ledger.balance[sender], 0.02, 0.06), 0, "family_gift"))
    ledger.emit(events, "LEGIT_FAMILY_FANOUT", instance_id, holdout)


def legit_small_shop():
    """Twin of MULE_HUB: 5-15 customers pay one small shop (low balance)
    over a day, each a small part of their own balance — many senders into
    one small account, but nobody is drained."""
    instance_id, holdout = new_instance("LEGIT_SMALL_SHOP")
    shop = ledger.open(u(2_000, 30_000))
    b = base_step(21)
    events = []
    for step in b + rng.integers(0, 21, n_between(5, 15)):
        customer = ledger.open(u(5_000, 200_000))
        events.append((int(step), "TRANSFER", customer, shop, small_payment(ledger.balance[customer], 0.01, 0.10), 0, "shop_payment"))
    ledger.emit(events, "LEGIT_SMALL_SHOP", instance_id, holdout)


GENERATORS = {
    "MULE_HUB": mule_hub,
    "STRUCTURING": structuring,
    "SMURF_FANOUT": smurf_fanout,
    "AGENT_COLLUSION": agent_collusion,
    "ACCOUNT_TAKEOVER": account_takeover,
    "MULE_NETWORK": mule_network,
    "LEGIT_PAYROLL": legit_payroll,
    "LEGIT_BUSY_AGENT": legit_busy_agent,
    "LEGIT_BILL_PAYMENTS": legit_bill_payments,
    "LEGIT_SHARED_PAYEES": legit_shared_payees,
    "LEGIT_FAMILY_SUPPORT": legit_family_support,
    "LEGIT_FAMILY_FANOUT": legit_family_fanout,
    "LEGIT_SMALL_SHOP": legit_small_shop,
}


def validate(df):
    assert df["step"].between(1, MAX_STEP).all(), "step out of range"
    assert (df["amount"] > 0).all(), "non-positive amount"
    assert (df[["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]] >= -0.01).all().all(), "negative balance"
    assert np.allclose(df["oldbalanceOrg"] - df["amount"], df["newbalanceOrig"], atol=0.02), "sender balance doesn't add up"
    assert np.allclose(df["oldbalanceDest"] + df["amount"], df["newbalanceDest"], atol=0.02), "receiver balance doesn't add up"
    assert df["nameOrig"].str.len().eq(12).all() and df["nameDest"].str.len().eq(12).all(), "id could collide with PaySim"
    assert df["type"].isin(["TRANSFER", "CASH_OUT"]).all()


def main():
    parser = argparse.ArgumentParser(description="Inject graph fraud patterns into PaySim.")
    parser.add_argument("--src", default=DEFAULT_SRC, help="original PaySim CSV")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="where PS_augmented.csv + injected_manifest.csv go")
    args = parser.parse_args()
    out_path = os.path.join(args.out_dir, "PS_augmented.csv")
    manifest_path = os.path.join(args.out_dir, "injected_manifest.csv")

    for pattern, count in COUNTS.items():
        for _ in range(count):
            GENERATORS[pattern]()
    for source_pattern, twin_pattern in ISOLATED_TWINS.items():
        ledger.clone_isolated(source_pattern, twin_pattern)

    injected = pd.DataFrame(ledger.rows, columns=COLUMNS)
    meta = pd.DataFrame(ledger.meta, columns=["pattern", "instance_id", "role", "isFraud", "holdout"])
    validate(injected)

    shutil.copyfile(args.src, out_path)
    with open(out_path, "rb+") as f:  # make sure the copy ends with a newline before appending
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            f.write(b"\n")
    injected.to_csv(out_path, mode="a", header=False, index=False)

    manifest = pd.concat([meta.drop(columns="isFraud"), injected], axis=1)
    manifest.insert(0, "inject_id", range(1, len(manifest) + 1))
    manifest.to_csv(manifest_path, index=False)

    summary = manifest.groupby("pattern", sort=False).agg(
        instances=("instance_id", "nunique"),
        rows=("inject_id", "size"),
        fraud_rows=("isFraud", "sum"),
    )
    summary["legit_rows"] = summary["rows"] - summary["fraud_rows"]
    summary["holdout_instances"] = manifest[manifest["holdout"]].groupby("pattern")["instance_id"].nunique()
    print(summary[["instances", "rows", "fraud_rows", "legit_rows", "holdout_instances"]].to_string())
    print(f"\nInjected {len(injected):,} rows ({injected.isFraud.sum():,} fraud, {(injected.isFraud == 0).sum():,} legit)")
    print(f"Wrote {out_path} and {manifest_path}")


if __name__ == "__main__":
    main()
