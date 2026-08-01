# SPL3 — Session Summary

A working real-time, graph-based fraud detection system: PaySim data
pipeline → trained fused GNN → FastAPI backend with a live transaction
stream → Next.js dashboard. This summarizes what was built, the decisions
behind it, and what's still open.

## 1. Data pipeline (Colab)

- `01_load_paysim.py` — loads PaySim, keeps only TRANSFER/CASH_OUT (the
  only fraud-eligible types), label-encodes `type`, `StandardScaler`s
  amount/balance columns, and persists the fitted encoder/scaler so the
  backend can transform raw live transactions the same way later.
- `03_graph_builder.py` — builds the transaction graph: edges between
  transactions sharing a sender or receiver within a 24h window, via a
  two-pointer sliding window (O(n log n), not nested loops). A third edge
  type (money-flow / mule-chain) was attempted, crashed Colab's RAM
  (materialized a per-account dict of DataFrames across millions of
  accounts), and was reverted — **still open**, see §7.
- `04_validate_graph.py` — sanity-checks node/edge counts and plots a
  sample neighbourhood before trusting anything built on top of the graph.

## 2. Model

`main_gnn_model.py` / `train_main_gnn.py` — the flagship model, inspired by
BRIGHT + GraphSAGE:
- **BatchNet** (2× GCNConv) — offline, full-graph, transductive pathway.
- **RealTimeNet** (2× SAGEConv) — inductive pathway; the only one that runs
  in production (`use_batch=False`), since it doesn't need the full graph.
- **LambdaNeuralNetwork** — fuses both via concatenation + linear
  projection; zeroes the BatchNet half when `use_batch=False`.
- Trained with weighted BCE (class imbalance ~336:1), 70/15/15 split, early
  stopping on validation AUC-ROC.
- Fixed along the way: a GPU OOM (dual-encoder full-batch training exceeded
  T4 memory — dropped `hidden_channels` 64→32, reduced eval frequency) and
  a numpy 1.x/2.x pickle incompatibility in the saved checkpoint.

**Benchmarked against:** standalone GCN, standalone SAGE, and an XGBoost
tabular baseline — see `06_train_compare.py` / `07_xgboost_baseline.py`.

**Measured results:**
| Model | Setting | ROC-AUC |
|---|---|---|
| GCN | transductive / inductive | 0.797 / 0.904 |
| SAGE | transductive / inductive | 0.862 / 0.901 |
| XGBoost | tabular (no graph) | 0.999 |
| **Fused GNN** | **full / real-time-only** | **0.970 / 0.956** |

XGBoost's higher raw number is explained, not ignored: PaySim's fraud rule
makes balance-drain-to-zero an almost perfect single-row giveaway (its top
2 features carry two-thirds of its decision weight). The fused GNN clearly
beats every graph-only alternative, which is the fair comparison for the
architecture's actual contribution. The feature-ablation experiment that
would rigorously test "does the graph help once that shortcut is removed"
(`REDUCED_FEATURE_COLS` toggle) is built but **not yet run** — see §7.

## 3. Architecture decisions (documented in `architecture.md`)

- **FastAPI-only backend**, not the original FastAPI + Spring Boot split —
  no prior Java experience, tight solo timeline, avoids an extra network
  hop. Spring Boot left as a documented "if time allows" option.
- **PostgreSQL over NoSQL** — the schema is inherently relational
  (FK-heavy audit trail); Redis already covers the one place a fast
  ephemeral KV store is actually justified.
- **Hosted, no Docker** — Neon (Postgres) and Upstash (Redis) instead of
  local containers, matching the "avoid Docker" preference and doubling as
  the eventual production DB with zero migration.
- **Stream simulator instead of Kafka** — `stream_simulator.py` generates
  realistic transactions continuously through the exact same scoring
  pipeline a real Kafka consumer would use. Documented as a stand-in, with
  a clean swap path if real Kafka is added later.

## 4. Backend (FastAPI)

- JWT auth, two roles (ADMIN/ANALYST), no user table (see §7).
- `POST /score` — Redis neighbour lookup → feature scaling (persisted
  encoder/scaler) → real-time GNN inference → 4-tier decision → Postgres
  persistence → Redis self-registration → WebSocket broadcast. Accepts raw
  transaction values; scaling happens server-side.
- `scoring_pipeline.py` — the single shared implementation of the above,
  used by both `/score` and the stream simulator, so there's one code path
  for "what happens when a transaction is scored."
- `GET/POST /api/stream/status`, `/toggle` — pause/resume the live feed.
- `GET /api/fraud/results`, `/api/fraud/stats` — dashboard data.
- `POST /api/fraud/alerts/{id}/acknowledge` — human-in-the-loop review,
  broadcast live to all connected clients.
- `WS /ws/alerts` — native WebSocket push (token via query param, since
  browsers can't set custom headers on a WebSocket handshake).
- Redis lookups degrade gracefully (log + continue) rather than crash if
  unreachable.

Bugs caught and fixed during integration testing (not just "should work" —
every feature was tested against the real running server, real Neon DB,
and real WebSocket clients before being handed off):
- A subtle `torch_geometric.utils.subgraph()` bug (inferred node count from
  `edge_index` instead of the true graph size — broke on isolated nodes).
- A numpy/pandas ABI break from upgrading numpy alone without its
  dependents.
- `frontend/` got accidentally committed as a nested git repo (gitlink)
  instead of real files — caught before pushing.
- `backend/.gitignore` only excluded `model/*.pt`, not `.pkl` — the
  scaler/encoder would have leaked into git.
- Orphaned background processes left listening on port 8000 after
  "stopping" the server — traced to a multiprocessing child and cleaned up.

## 5. Frontend (Next.js + Tailwind)

- `/login` — JWT auth.
- `/dashboard` — stats panel, a Score-a-Transaction form (with presets and
  now-editable sender/receiver fields + a "New ID" button to demo scoring
  a truly never-seen account), a WebSocket-driven live feed with
  acknowledge/false-positive actions, and `SubgraphViz` — a hand-rolled SVG
  star diagram showing the *actual* subgraph fed to the model for that
  score, with native hover tooltips for exact figures.
- `/stream` — a dedicated Live Stream Monitor: connection status, a
  play/pause control wired to the backend toggle, live session throughput
  stats, and a real-time transaction ticker with column headers and an
  explicit "where this data comes from" banner (so the data source is
  never a mystery mid-demo).

## 6. Git / GitHub

Connected to an existing remote that already had prior work
(`PrimaryGnnBaseModel.py`, an independently-written version of the same
BatchNet/RealTimeNet architecture) — merged histories rather than
overwriting, per your choice. Verified no secrets or oversized files (the
471MB dataset/graph files, Neon password) made it into the push.

## 7. Known open items

- **Money-flow / ring edge** — not built in the offline graph (§13.2 of
  architecture.md); the "GNN structurally detects rings" claim isn't true
  yet. A real-time, Redis-based rule-engine alternative was planned but not
  built.
- **Feature-ablation run** — the toggle exists, the actual run (proving
  graph value once the balance-drain shortcut is removed) hasn't happened.
- **Graph congestion cap** — `SubgraphViz` renders every neighbour found
  (seen up to 30 in practice); capping to the most recent ~8 with a "+N
  more" indicator was proposed and explicitly deferred.
- **`users` table** — `alerts.acknowledged_by` is a free-text string, not a
  verified identity; noted as a real gap for audit-trail credibility.
- **Kafka, IEEE-CIS generalisation** — not started.
- **Deployment** — nothing is live on Render/Vercel; everything has been
  run and tested locally only.
- **Real-time recall** — measured at 0.336 at the BLOCK threshold; most
  fraud-shaped synthetic transactions in the live demo won't cross into
  BLOCK. Documented in `PATTERNS_AND_QA.md`, not hidden.

## 8. Reference docs produced

- `architecture.md` — system design, schema, ER diagram, decision log, Q&A.
- `PATTERNS_AND_QA.md` — how demo data is built, what patterns the model
  learned, expected risk tier per pattern, and the GNN-vs-LR/XGBoost
  defense.
- `backend/README.md`, `frontend/README.md` — setup instructions.
