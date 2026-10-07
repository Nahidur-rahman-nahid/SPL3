# Fraud Detection Patterns & Q&A Reference

A reference for explaining, to a teacher or a buyer, (1) how the demo data is
built, (2) what patterns the model actually learned, (3) what input shapes
predict which risk tier, and (4) how to defend the GNN choice against "why
not just logistic regression / XGBoost."

---

## 1. How the demo/synthetic transactions are constructed

**Live stream (`stream_simulator.py`).** Every 2 seconds one transaction is
generated. Balances carry over from one transaction to the next, like a real
ledger, and transactions close in time share the same `step` (hour).

| Share | What happens | Accounts |
|---|---|---|
| 85% everyday | shop payment (≤ 10% of balance), money for the customer's own relative (≤ 15%), or a cash-out at an agent (5–30%) | 60 recurring customers, 12 established shops, 20 relatives, 6 agents |
| 11% drain | a victim's whole balance moves into a mule; 4–8 victims share each mule | fresh victim and mule accounts |
| 4% structuring | one sender sends the same receiver 3–5 transfers of BDT 45,000–49,999 on consecutive ticks | fresh accounts |

Fraud always arrives as a group, the way the GNN learned it, so the first
transaction of a group has no neighbours yet and usually passes; the rest
are caught once the group is visible.

**Score page presets (`ScoreForm.js`).**

| Preset | Transaction | Supplied related transactions | Retrained model |
|---|---|---|---|
| Normal payment | 5,000 of a 25,000 balance to a merchant | none | 2.8% ALLOW |
| Suspicious (structuring) | 47,500 from a 302,700 balance | 2 earlier near-limit transfers (48,200, 49,100) from the same sender to the same receiver | 0.1% alone → 63% REVIEW |
| High-risk (mule hub) | a 47,000 account drained into a mule | 3 other victims drained into the same mule | 12.6% alone → 95% BLOCK |

The fraud presets stay under the BDT 50,000 daily limit so the GNN's own
decision is shown, not the regulatory block.

**Fraud Patterns page (`pattern_demo.py`).** Replays the 6 injected fraud
patterns and a look-alike for each, one transaction at a time; see
`INJECTED_PATTERNS.md` for how the patterns were built and verified.

---

## 2. What patterns the model actually learned

> This section describes the **first model, trained on PaySim only**. The
> current model was retrained on PaySim plus injected graph patterns; what
> it learned, and how that was verified, is in `INJECTED_PATTERNS.md` and §3.

Two honest caveats before this section: (a) we have real measured feature
importance for the **XGBoost baseline**, but no equivalent formal
explainability run (e.g. GNNExplainer, per-feature ablation) on the fused
GNN itself — what follows for the GNN is inferred from training data
structure and aggregate performance numbers, not a verified attribution
study. (b) The fused GNN was trained only on `SAME_SENDER`/`SAME_RECEIVER`
edges — the money-flow (mule-chain) edge is not yet built (architecture.md
§13.2), so ring-shaped patterns are only *indirectly* visible (via a shared
receiver's neighbour list), not structurally guaranteed to be caught.

**Confirmed (from XGBoost's measured feature importance on the same PaySim
data):**
| Feature | Importance |
|---|---|
| `newbalanceOrig` | 0.394 |
| `oldbalanceOrg` | 0.272 |
| `newbalanceDest` | 0.139 |
| `amount` | 0.101 |
| `type` | 0.072 |
| `step` | 0.015 |
| `oldbalanceDest` | 0.008 |

The top two features alone (both balance-after-transaction fields) carry
two-thirds of the decision weight. PaySim's fraud rule is essentially
"did this account get drained to (near) zero" — that single signal is
unusually strong in this dataset specifically (see §5 for why that matters).

**What the GNN additionally has access to, structurally, that a tabular
model does not:**
- Same-account burst signal — whether the sender or receiver has been
  unusually active (multiple transactions) in the last 24h, via the
  `SAME_SENDER`/`SAME_RECEIVER` edges.
- Neighbourhood aggregation — `RealTimeNet` (SAGEConv) mean-pools the
  features of every related transaction found via Redis, so the score
  reflects "what has this account been doing lately," not just the single
  row in isolation.

---

## 3. Expected risk tier by data pattern (retrained model)

| Pattern | Example | Result (real-time mode) |
|---|---|---|
| Everyday payment, alone or repeated | customer pays the same shop or relative several times, small share of balance | ALLOW (0.5–3%); 0 of 308 everyday stream transactions flagged |
| Drain into a mule shared with other victims | full balance → mule that already received other drains | BLOCK (87–95%); 82% of stream drains blocked |
| Structuring | 3–5 transfers just under 50,000, same sender and receiver | 89–91% from the second transfer on |
| Smurf fan-out, agent collusion, account takeover, mule network | see `INJECTED_PATTERNS.md` | 89–95% once the group is visible; every look-alike ≤ 10% |
| Any fraud transaction with no related transactions yet | first transaction of a group | usually ALLOW: there is nothing for the graph to see yet |

**Overall test set (420,648 transactions, 0.67% fraud):** ROC-AUC 0.995,
recall 89%, precision 44%, false positive rate 0.77%. The first PaySim-only
model had recall 34% in real-time mode.

Why precision is 44% even though the false positive rate is under 1%:
legitimate transactions outnumber fraud about 150 to 1, so 0.77% of them
(≈ 3,200) is more than the ≈ 2,500 frauds caught. Random flagging would
give 0.67% precision; the CRITICAL tier alone is 61% fraud.

---

## 4. Q&A — defending the GNN choice

**Q: Logistic regression can already give a probability for a completely
new, never-seen sender/receiver/amount. XGBoost can too, and in your own
benchmark XGBoost had *higher* accuracy (ROC-AUC 0.999 vs. the GNN's 0.970
full / 0.956 real-time-only). Why use a GNN at all?**

**A:** All true, and worth being upfront about rather than hiding it.

*Why XGBoost wins on raw accuracy here:* PaySim's fraud-generation rule
makes balance-drain-to-zero an almost perfect single-row giveaway — see the
feature importance table above. Any model, even plain logistic regression,
can exploit that shortcut on *this specific dataset*. That's a property of
PaySim's simulator, not evidence that relational structure is unnecessary
for fraud detection in general. Real-world fraud (laundering chains,
coordinated mule networks) rarely hands you such a clean single-row tell.

*What logistic regression/XGBoost structurally cannot do, no matter how
well tuned:*
- They score every row independently. To catch "three different senders
  paid into this same account in the last hour," you would have to
  *anticipate that exact pattern* and hand-engineer a matching aggregate
  feature, recomputed live for every prediction. Every new pattern you
  want to catch needs its own hand-built column.
- They require a fixed-width feature vector. A transaction with zero
  related transactions and one with thirty look identical in *shape* to a
  tabular model unless you engineer count-based columns for every case you
  thought of in advance. The GNN handles a variable number of neighbours
  as normal operation, with no upper bound decided ahead of time.
- "Works on an unseen sender/receiver" is true for LR/XGBoost only because
  they never used account identity or relational context as a feature in
  the first place — they were never solving the inductive problem, they
  simply never attempted it. The GNN's inductive capability specifically
  means it *does* use relational context (who this account has recently
  interacted with) while *still* handling a brand-new account gracefully —
  a strictly harder and more useful property, not the same thing.
- Explainability is different in kind, not just degree: XGBoost's feature
  importance tells you which *columns* matter on average, globally. It
  can't show you *which other transactions* drove a specific score. The
  GNN's subgraph visualization shows the actual relational evidence behind
  one decision — a genuinely different, complementary kind of explanation.

*Measured proof that the graph adds something XGBoost cannot:* for five
graph-only patterns, every fraud row has an exact copy (same amount, same
balances) sent between unrelated accounts and labelled legit. Row by row the
two are identical, so XGBoost scores them the same and separates them by
0.00. The retrained GNN separates them by 0.80–1.00, because only the fraud
copy has related transactions (`09_evaluate_patterns.py`, check D).

**Q: So does the graph actually help, or is this all theoretical?**

**A:** It helps, and it is measured. In the first model, linking a
transaction to known fraud *lowered* its score, because in plain PaySim the
busiest accounts are legitimate. After retraining on injected graph patterns:
- linking to the fraud group raises the score (structuring: 0.1% alone →
  92% with its group; the old model went 49% → 23%);
- the six fraud patterns are caught 98–100% on held-out data the model never
  saw, versus 0–25% for the old model;
- identical rows are told apart only by the graph (0.80–1.00 vs XGBoost 0.00);
- plain PaySim fraud is still detected (real-time AUC 0.96 → 0.99).

XGBoost still has the higher raw AUC on plain PaySim rows (0.999), because
the balance-drain shortcut gives most PaySim fraud away row by row. Both
statements are true at once: XGBoost is best at single-row fraud, the GNN is
needed for fraud that only shows in the relationships.
