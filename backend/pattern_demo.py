"""
Fraud Patterns demo: replays each injected fraud pattern (and its
legitimate look-alike) through the REAL scoring pipeline, one transaction
at a time, so the dashboard can show the GNN's score climbing as a fraud
group forms while the look-alike stays low.

The scenarios mirror 00_inject_fraud_patterns.py (see INJECTED_PATTERNS.md)
with fixed, readable example numbers instead of random ones. For the five
graph-only patterns the look-alike is the isolated twin: the exact same
transactions (amounts, balances, timing) sent between unrelated accounts —
row by row identical, so any difference in score comes from the graph.

Every run uses fresh account ids, so Redis neighbour history from one run
never leaks into another. Runs go through process_transaction() with
source="pattern_demo": like the stream simulator, that skips the
Bangladesh Bank daily limit and night rule, which would otherwise block
e.g. structuring on the 50,000 limit before the GNN is even consulted.
Transactions are stored like any other ("pattern-…" ids) and can be
removed with Data Cleanup -> manually scored data.
"""

import secrets
from dataclasses import dataclass

from sqlalchemy.orm import Session

import ml_service
import redis_client
from schemas import ScoreRequest, TransactionFeatures
from scoring_pipeline import process_transaction

DEMO_STEP = 200  # any mid-simulation hour; all transactions of a run share it


@dataclass
class DemoTxn:
    sender: str
    receiver: str
    features: TransactionFeatures
    note: str


class _Ledger:
    """Fresh accounts + running balances for one run, so every row's
    old/new balances chain exactly like a real account's would."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.balance = {}

    def account(self, label: str, balance: float) -> str:
        name = f"C_DEMO_{self.run_id}_{label}"
        self.balance[name] = round(balance, 2)
        return name

    def send(self, sender: str, receiver: str, amount: float, note: str, txn_type: str = "TRANSFER") -> DemoTxn:
        amount = round(min(amount, self.balance[sender]), 2)
        old_org, old_dest = self.balance[sender], self.balance[receiver]
        self.balance[sender] = round(old_org - amount, 2)
        self.balance[receiver] = round(old_dest + amount, 2)
        return DemoTxn(sender, receiver, TransactionFeatures(
            step=DEMO_STEP, type=txn_type, amount=amount,
            oldbalanceOrg=old_org, newbalanceOrig=self.balance[sender],
            oldbalanceDest=old_dest, newbalanceDest=self.balance[receiver],
        ), note)


# ------------------------------------------------------------- scenarios

def _mule_hub(L):
    mule = L.account("MULE", 0)
    return [L.send(L.account(f"VICTIM{i}", bal), mule, bal, f"victim {i} drained into the mule")
            for i, bal in enumerate([120_000, 95_000, 210_000, 160_000, 75_000, 180_000, 140_000, 250_000], 1)]


def _shared_payees(L):
    landlord = L.account("LANDLORD", 1_500_000)
    return [L.send(L.account(f"TENANT{i}", bal), landlord, 12_000, f"tenant {i} pays rent")
            for i, bal in enumerate([150_000, 220_000, 180_000, 310_000, 260_000, 400_000, 175_000, 290_000], 1)]


def _structuring(L):
    sender, receiver = L.account("SENDER", 400_000), L.account("RECEIVER", 90_000)
    return [L.send(sender, receiver, amt, f"near-limit transfer {i} of 4")
            for i, amt in enumerate([48_200, 49_100, 47_500, 46_900], 1)]


def _smurf_fanout(L):
    source = L.account("SOURCE", 600_000)
    amounts = [52_000, 47_500, 49_800, 55_200, 46_100, 51_300, 48_700, 53_900, 44_600, 50_900]
    return [L.send(source, L.account(f"NEW{i}", 0), amt, f"to brand-new account {i}")
            for i, amt in enumerate(amounts, 1)]


def _agent_collusion(L):
    agent = L.account("AGENT", 20_000)
    balances = [8_000, 15_500, 12_200, 21_000, 9_800, 18_400, 11_100, 24_600, 13_300, 16_900, 10_400, 19_700]
    return [L.send(L.account(f"CUST{i}", bal), agent, bal, f"near-empty account {i} cashes out everything", "CASH_OUT")
            for i, bal in enumerate(balances, 1)]


def _account_takeover(L):
    owner = L.account("OWNER", 300_000)
    rows = [L.send(owner, L.account(f"PAYEE{i}", bal), amt, f"owner's normal payment {i}")
            for i, (amt, bal) in enumerate([(6_000, 80_000), (9_000, 150_000), (4_500, 60_000)], 1)]
    remaining = L.balance[owner]
    rows += [L.send(owner, L.account(f"NEW{i}", 0), remaining * share, f"takeover: to brand-new account {i}")
             for i, share in enumerate([0.24, 0.26, 0.22, 0.23], 1)]
    return rows


def _mule_network(L):
    mules = [L.account(f"MULE{i}", 0) for i in range(1, 5)]
    plan = [(150_000, (0, 1)), (220_000, (1, 2)), (90_000, (0, 3)), (300_000, (2, 3)), (180_000, (0, 2)), (260_000, (1, 3))]
    rows = []
    for v, (bal, targets) in enumerate(plan, 1):
        victim = L.account(f"VICTIM{v}", bal)
        rows += [L.send(victim, mules[t], bal * 0.22, f"victim {v} pays mule {t + 1}") for t in targets]
    return rows


def _isolated_copy(build):
    """The same transactions, each between two brand-new accounts that
    carry the exact same balances — identical rows, no neighbours."""
    def lookalike(L):
        rows = build(_Ledger(L.run_id + "X"))
        out = []
        for i, r in enumerate(rows, 1):
            f = r.features
            sender = L.account(f"ALONE{i}_FROM", f.oldbalanceOrg)
            receiver = L.account(f"ALONE{i}_TO", f.oldbalanceDest)
            out.append(L.send(sender, receiver, f.amount, f"same as fraud row {i}, unrelated accounts", f.type))
        return out
    return lookalike


PATTERNS = {
    "MULE_HUB": {
        "title": "Mule hub",
        "summary": "Many victims are fully drained into one fresh mule account within hours.",
        "signal": "Many SAME_RECEIVER neighbours, all of them drains into a brand-new account.",
        "fraud": _mule_hub,
        "lookalike": _shared_payees,
        "lookalike_title": "Tenants paying one landlord",
        "lookalike_summary": "8 people pay the same established landlord a normal rent — a legit fan-in.",
    },
    "STRUCTURING": {
        "title": "Structuring",
        "summary": "One sender splits a large sum into transfers just under the BDT 50,000 limit.",
        "signal": "Several near-limit transfers from the SAME_SENDER in a short time.",
        "fraud": _structuring,
        "lookalike": _isolated_copy(_structuring),
        "lookalike_title": "Same transfers, unrelated people",
        "lookalike_summary": "Exactly the same 4 transfers, each by a different person — one rent payment each.",
    },
    "SMURF_FANOUT": {
        "title": "Smurfing fan-out",
        "summary": "One account spreads most of its balance over many brand-new accounts.",
        "signal": "SAME_SENDER fan-out to accounts that had no money before.",
        "fraud": _smurf_fanout,
        "lookalike": _isolated_copy(_smurf_fanout),
        "lookalike_title": "Same transfers, unrelated people",
        "lookalike_summary": "Exactly the same 10 transfers, each by a different person to a different new payee.",
    },
    "AGENT_COLLUSION": {
        "title": "Agent collusion",
        "summary": "Many near-empty accounts all cash out everything at the same agent.",
        "signal": "Many SAME_RECEIVER cash-outs converging on one agent.",
        "fraud": _agent_collusion,
        "lookalike": _isolated_copy(_agent_collusion),
        "lookalike_title": "Same cash-outs, different agents",
        "lookalike_summary": "Exactly the same 12 cash-outs, each at a different agent.",
    },
    "ACCOUNT_TAKEOVER": {
        "title": "Account takeover",
        "summary": "After a few normal payments, the account suddenly sends almost everything to new accounts.",
        "signal": "A sudden SAME_SENDER burst to brand-new accounts after normal history.",
        "fraud": _account_takeover,
        "lookalike": _isolated_copy(_account_takeover),
        "lookalike_title": "Same payments, unrelated people",
        "lookalike_summary": "Exactly the same 7 payments, each by a different person.",
    },
    "MULE_NETWORK": {
        "title": "Mule network",
        "summary": "Victims each pay two mules from a shared set of fresh mule accounts.",
        "signal": "Dense SAME_SENDER + SAME_RECEIVER web between victims and mules (2 hops).",
        "fraud": _mule_network,
        "lookalike": _isolated_copy(_mule_network),
        "lookalike_title": "Same transfers, unrelated people",
        "lookalike_summary": "Exactly the same 12 transfers, each between two unrelated accounts.",
    },
}


def list_patterns() -> dict:
    return {
        "model": ml_service.model_info(),
        "patterns": [
            {"id": pid, "title": p["title"], "summary": p["summary"], "signal": p["signal"],
             "lookalike_title": p["lookalike_title"], "lookalike_summary": p["lookalike_summary"]}
            for pid, p in PATTERNS.items()
        ],
    }


def _redis_available() -> bool:
    try:
        return bool(redis_client.get_client().ping())
    except Exception:
        return False


async def run_pattern(pattern_id: str, variant: str, db: Session) -> dict:
    pattern = PATTERNS[pattern_id]
    run_id = secrets.token_hex(3).upper()
    txns = pattern[variant](_Ledger(run_id))

    steps = []
    for i, t in enumerate(txns, 1):
        request = ScoreRequest(
            transaction_id=f"pattern-{pattern_id.lower()}-{variant}-{run_id}-{i}",
            sender_account=t.sender, receiver_account=t.receiver, features=t.features,
        )
        result = await process_transaction(request, db, source="pattern_demo")
        alone, _ = ml_service.score(t.features, [])
        f = t.features
        steps.append({
            "index": i,
            "transaction_id": request.transaction_id,
            "sender": t.sender,
            "receiver": t.receiver,
            "type": f.type,
            "amount": f.amount,
            "share_of_balance": round(f.amount / f.oldbalanceOrg, 4) if f.oldbalanceOrg else None,
            "receiver_was_empty": f.oldbalanceDest == 0,
            "note": t.note,
            "neighbour_count": result["neighbour_count"],
            "probability": result["fraud_probability"],
            "alone_probability": alone,
            "decision": result["decision"],
            "risk_level": result["risk_level"],
        })

    return {
        "pattern_id": pattern_id,
        "variant": variant,
        "run_id": run_id,
        "redis_available": _redis_available(),
        "steps": steps,
        "flagged": sum(s["decision"] != "ALLOW" for s in steps),
        "mean_probability": sum(s["probability"] for s in steps) / len(steps),
        "mean_alone_probability": sum(s["alone_probability"] for s in steps) / len(steps),
    }
