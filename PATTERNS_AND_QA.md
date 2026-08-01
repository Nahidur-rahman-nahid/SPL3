# Fraud Detection Patterns & Q&A Reference

A reference for explaining, to a teacher or a buyer, (1) how the demo data is
built, (2) what patterns the model actually learned, (3) what input shapes
predict which risk tier, and (4) how to defend the GNN choice against "why
not just logistic regression / XGBoost."

---

## 1. How the demo/synthetic transactions are constructed

Both `ScoreForm.js`'s presets and `stream_simulator.py`'s background
generator build transactions along the same three shapes, deliberately
mirroring PaySim's own structure:

**Normal payment**
```
amount            = random(500, 15,000)
oldbalanceOrg     = amount + random(5,000, 30,000)   -- healthy buffer beyond the payment
newbalanceOrig    = oldbalanceOrg - amount             -- partial draw, balance survives
type              = random(TRANSFER, CASH_OUT)
sender/receiver   = a small pool of recurring customer/merchant accounts
```
Balance stays healthy after the transaction — the account is not being emptied.

**Suspicious drain-to-mule**
```
amount            = random(80,000, 250,000)
oldbalanceOrg     = amount        -- the ENTIRE balance
newbalanceOrig    = 0             -- account emptied to zero
oldbalanceDest    = 0
newbalanceDest    = amount
type              = TRANSFER
sender/receiver   = a victim account -> a mule (personal C_ account, not a merchant M_ account)
```
This is PaySim's actual fraud signature: draining an account fully in one
TRANSFER into a personal account. It's not an arbitrary "make it look bad"
pattern — it's literally how PaySim's simulator generates its `isFraud=1`
rows.

**Ring / agent collusion**
Same drain shape as above, but the *sender* is chosen from a pool of several
different victim accounts, all paying into the *same* receiver (a shared
mule). This is the account-level shape of a fraud ring — several unrelated
parties converging on one account in a short window.

---

## 2. What patterns the model actually learned

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

## 3. Expected risk tier by data pattern

| Pattern | Example values | Design intent | Observed live behaviour |
|---|---|---|---|
| Normal payment | amount ≪ balance, balance survives | ALLOW / LOW | Matches consistently |
| Drain-to-mule | full balance → 0, personal receiver | REVIEW or BLOCK | Often lands ALLOW/REVIEW in real-time mode — see limitation below |
| Ring (multi-sender → one receiver) | several drain-pattern senders, same receiver | BLOCK / CRITICAL once ring detection exists | Not specifically detected yet — no dedicated ring logic built |

### Known limitation: why many "suspicious" transactions still show ALLOW live

This matches something we already measured during training, not a new bug.
The live stream always runs in **real-time-only mode** (`use_batch=False` —
no full offline graph, exactly like production). That mode's measured
performance on the held-out test set was:

```
ROC-AUC: 0.956   Precision: 0.623   Recall: 0.336   FPR: 0.0006
```

Recall of 0.336 means roughly **two-thirds of genuinely fraud-shaped
transactions do not cross the 0.75 BLOCK threshold** in real-time mode —
even when the transaction has the classic drain-to-zero shape. This is the
real, previously-quantified trade-off of the real-time-only pathway: very
few false alarms (0.06% FPR), at the cost of missing a majority of subtle
fraud cases. It is an honest limitation to state plainly rather than a
demo malfunction — and it's a legitimate discussion point about threshold
tuning being a business decision (lower the BLOCK threshold to catch more,
at the cost of more false positives) rather than a fixed "correct" answer.

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

*Honest next step, not yet done:* the rigorous way to prove "the graph adds
value beyond what the balance-drain shortcut already gives away" is the
feature-ablation experiment already wired into `03_graph_builder.py` /
`07_xgboost_baseline.py` (`REDUCED_FEATURE_COLS`, which drops the balance
columns XGBoost leans on). That comparison hasn't been run yet — it's the
real test of this claim, and it should be presented as pending, not as
already-proven.

**Q: So does the graph actually help, or is this all theoretical?**

**A:** Within what we've measured so far: the fused GNN (0.970 AUC full,
0.956 real-time-only) beats every plain single-pathway baseline we tried —
standalone GCN (0.80–0.90 AUC) and standalone SAGE (0.86–0.90 AUC) — by a
wide margin. So fusing the two pathways demonstrably helps *relative to
graph-only alternatives*. It does not yet beat XGBoost's raw accuracy on
*this* dataset, for the reason above. Both statements are true at once, and
saying so directly is more credible than picking one and ignoring the
other.
