# Project Architecture — Real-Time Graph-Based Fraud Detection System

## 1. Core Research Contribution

The primary paper implemented is **BRIGHT** (ACM CIKM 2022, arXiv 2205.13084) — a bi-level information propagation GNN for fraud detection. BRIGHT is **transductive**: it cannot score unseen nodes.

The novel contribution of this project is integrating **GraphSAGE inductive layers** (NeurIPS 2017, arXiv 1706.02216) so the model generalises to brand-new transaction nodes arriving in real time.

The system benchmarks three models:
- XGBoost tabular baseline
- FraudGNN transductive (BatchNet, GCNConv-based)
- FraudGNN with SAGEConv inductive layer (RealTimeNet)

Metrics reported for all three: **AUC-ROC, Precision, Recall, F1, False Positive Rate**.

---

## 2. Dataset Decision

### Primary dataset: PaySim
- Source: [kaggle.com/datasets/ealaxi/paysim1](https://kaggle.com/datasets/ealaxi/paysim1)
- 6.3 million synthetic mobile money transactions
- 11 columns: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`
- Fraud only occurs on `TRANSFER` and `CASH_OUT` transaction types
- Fraud rate ≈ 0.13%
- Directly models bKash and Nagad transaction patterns (CASH-IN, CASH-OUT, TRANSFER, PAYMENT)
- Graph edges drawn on `nameOrig` (sender) and `nameDest` (receiver) within 24-hour time windows

### Secondary dataset: IEEE-CIS
- Source: [kaggle.com/competitions/ieee-fraud-detection](https://kaggle.com/competitions/ieee-fraud-detection)
- 590,540 real e-commerce transactions
- Two files joined on `TransactionID`: `train_transaction.csv` and `train_identity.csv`
- 394 columns total
- Graph edges on `card1`, `addr1`, `DeviceInfo`, `P_emaildomain` with time windows of 24h, 6h, 1h, 12h respectively
- Used for **cross-dataset generalisation benchmark only** — not for primary training

### Training strategy
Train and validate on PaySim. Fine-tune and evaluate on IEEE-CIS to prove cross-dataset generalisation.

---

## 3. GNN Model Architecture (`05_gnn_model.py`)

Four components:

**BatchNet** — two `GCNConv` layers, hidden dimension 64, BatchNorm, ReLU, dropout 0.3. Processes the full training graph during offline batch training. Captures global structural fraud signals.

**RealTimeNet** — two `SAGEConv` layers, mean aggregation, hidden dimension 64, BatchNorm, ReLU, dropout 0.3. Inductive — learns aggregation functions, not fixed embeddings. Handles unseen transaction nodes at inference time.

**LambdaNeuralNetwork** — fuses BatchNet and RealTimeNet embeddings via concatenation and linear projection (`hidden_dim * 2 → hidden_dim`).
- Training: `use_batch=True`
- Real-time inference: `use_batch=False`, BatchNet embedding replaced with a zero vector

**FraudGNN** — LambdaNeuralNetwork plus classifier head (`Linear → ReLU → Dropout → Linear → Sigmoid`). Outputs fraud probability in range [0, 1].

**Input dimension:** 7 features per transaction node — amount (scaled), timestamp (scaled), transaction type (encoded), sender balance before (scaled), sender balance after (scaled), receiver balance before (scaled), receiver balance after (scaled).

**Loss function:** Weighted Binary Cross-Entropy, `pos_weight = n_legitimate / n_fraud`, to handle class imbalance of ≈ 770:1.

**Training config:** 70/15/15 split, Adam optimizer (`lr=0.001`, `weight_decay=1e-4`), early stopping (`patience=10` on validation AUC-ROC), save best `model.pt`.

---

## 4. Decision Threshold Logic

| Fraud probability | Decision | Risk level |
|---|---|---|
| < 0.40 | ALLOW | LOW |
| 0.40 – 0.75 | REVIEW | MEDIUM |
| 0.75 – 0.90 | BLOCK | HIGH |
| ≥ 0.90 | BLOCK | CRITICAL |

Threshold `0.75` is configurable in `application.yml`.

---

## 5. Dynamic Graph at Inference Time (critical concept)

**Training phase (offline, Colab, runs once):** full PaySim graph loaded, GNN trained, `model.pt` saved. Google Colab is never involved in production.

**Production phase (always on, web server):**
1. A new transaction arrives via Kafka.
2. Spring Boot queries Redis for recent neighbour transactions, indexed by `nameOrig` and `nameDest` keys, TTL 86400 seconds.
3. A local subgraph is assembled with the new transaction as node index 0 and neighbours as nodes 1..N.
4. This subgraph is sent to FastAPI, which runs `SAGEConv` inference and returns a fraud probability.
5. After scoring, the new transaction is stored in Redis under its own attribute keys, for future transactions to find as a neighbour.

Redis acts as the live graph memory. **No full graph is ever loaded in production.**

**Redis key structure:**
```
account:{nameOrig} → list of recent transaction feature vectors, TTL 86400s
account:{nameDest} → list of recent transaction feature vectors, TTL 86400s
```

---

## 6. Fraud Ring Detection — How It Actually Works

A fraud ring is detected when multiple transactions share the same attribute (sender, receiver, device) within a time window. Example: three transactions from different senders all going to the same receiver within 1 hour form a **star pattern** in the graph.

The GNN's 2-hop message passing sees this pattern — each node aggregates features from its neighbours (hop 1) and its neighbours' neighbours (hop 2) — and learns during training that this star topology correlates with the fraud label. This structural pattern is invisible to XGBoost, which sees each row independently.

**PaySim fraud pattern:** TRANSFER from victim account to mule account, immediately followed by CASH_OUT from mule account to agent. The two transactions are connected through the shared `nameDest`/`nameOrig`, and the GNN detects the chain.

> **⚠ Open item:** see [§13.2](#132-fraud-ring--mule-chain-edge-not-yet-implemented) — this specific edge (receive → later send, same account) is not currently built into the graph.

---

## 7. Complete System Architecture — 5 Components

> **Revision (post-training-phase):** the original plan split ML inference (FastAPI) and business logic (Spring Boot/Java) into two services. That was revised to a **single FastAPI backend** doing both, given no prior Java/Spring experience and a tight solo-dev timeline — two languages, two deployments, and an extra network hop per request weren't worth the cost for this project. Spring Boot remains a documented option if time allows later (see §8), but the default build target is FastAPI-only.

### Component 1: Offline GraphBuilder (Python, Colab)
- Files: `01_load_paysim.py`, `02_load_ieee_cis.py`, `03_graph_builder.py`, `04_validate_graph.py`
- Responsibility: Load PaySim CSV, apply sliding-window edge rules, output PyG `Data` object with `x` (node features), `edge_index` (COO format), `y` (fraud labels).
- Edge rules for PaySim: same `nameOrig` within 24 hours, same `nameDest` within 24 hours.
- Algorithm: Sort transactions by time within each account group, then two-pointer sliding window — O(n log n), not O(n²).

### Component 2: FraudGNN Model (Python, Colab)
- Files: `05_gnn_model.py`, `main_gnn_model.py` + `train_main_gnn.py` (done — fused BatchNet/RealTimeNet/Lambda model), `07_xgboost_baseline.py` (done), `09_ieee_generalization.py` (not started)
- Responsibility: Train GNN and XGBoost, produce `model.pt`, generate 3-model comparison results table, produce ROC curve plot.

### Component 3: FastAPI Backend (Python) — merged ML inference + business logic
- Folder: `backend/`
- Files: `main.py`, `graph_builder_realtime.py`, `redis_client.py`, `schemas.py`, `db.py` (SQLAlchemy models/session), `kafka_consumer.py`, `websocket_manager.py`
- Responsibility — everything Components 3 and 4 covered in the original split, in one Python service:
  - Load `model.pt` at startup; expose `POST /score` accepting subgraph JSON, returning fraud probability + 4-tier decision in under 200ms; build the real-time local subgraph from Redis neighbours.
  - Consume the `transactions` topic (via `aiokafka` or `confluent-kafka-python`), score each transaction, persist the result to PostgreSQL (`fraud_decisions`, `alerts`).
  - Push BLOCK/REVIEW alerts to connected dashboard clients over a native FastAPI `WebSocket` endpoint (no STOMP/SockJS needed — that was a Spring-specific dependency).
  - Expose REST endpoints `GET /api/fraud/results` and `GET /api/fraud/stats` for the dashboard.
- Port: **8000**
- Build order (see §12): `/score` first (curl-testable) → Postgres models/persistence → REST read endpoints → Redis integration → a simple synchronous trigger path (no Kafka yet) → Kafka consumer added once that path works → WebSocket alerts last.

### Component 4: Redis Graph Cache
- Configuration: docker-compose service, port 6379
- Key schema: `account:{account_id}` → JSON list of recent transaction feature vectors, TTL 86400s
- Responsibility: Sub-10ms neighbour lookup during real-time graph construction.

### Component 5: Next.js React Dashboard
- Folder: `frontend/`
- Files: `components/LiveFeed.jsx`, `components/GraphViz.jsx`, `components/StatsPanel.jsx`, `components/ExportButton.jsx`, `hooks/useWebSocket.js`
- Responsibility: WebSocket subscription to the FastAPI alerts endpoint, live colour-coded transaction table (green ALLOW, yellow REVIEW, red BLOCK), D3 force-directed cluster graph of flagged transactions, Chart.js analytics panels, PDF/CSV export using jsPDF and Papa Parse.
- Port: **3000**

### Docker Compose
- File: `docker-compose.yml`
- Services: `zookeeper`, `kafka`, `redis`, `postgres`, `backend` (FastAPI, merged)
- Command: `docker compose up` starts everything **except** the Next.js frontend.

---

## 8. Complete Technology Stack

| Layer | Technology |
|---|---|
| Backend / ML language | Python (single language — see §7 revision note) |
| Frontend | Next.js, React, D3.js (force simulation), Chart.js, native WebSocket |
| Backend framework | FastAPI + Uvicorn — REST, native WebSocket, and ML inference all in one service |
| ML inference | PyTorch Geometric, torch, redis-py |
| GNN layers | `GCNConv`, `SAGEConv` from `torch_geometric.nn` |
| Streaming | Apache Kafka + Zookeeper, consumed via `aiokafka`/`confluent-kafka-python` |
| Graph cache | Redis, TTL-based key expiry |
| Database | PostgreSQL (SQLAlchemy + asyncpg) — tables: `transactions`, `fraud_decisions`, `graph_edges`, `alerts`, `model_versions` |
| Training environment | Google Colab, free T4 GPU |
| Containerisation | Docker, Docker Compose |
| Version control | Git, GitHub |
| Deployment | Backend on Render (free tier), frontend on Vercel (free tier) |
| IDE | Visual Studio Code with GitHub Copilot |
| Experiment tracking | Weights & Biases or TensorBoard |
| *(Deferred option)* | Java / Spring Boot — original two-service split; revisit only if time allows and polyglot architecture is worth demonstrating separately |

---

## 9. PostgreSQL Database Schema

### `transactions`
| Field | Type | Notes |
|---|---|---|
| transaction_id | VARCHAR | PRIMARY KEY |
| sender_account | VARCHAR | NOT NULL |
| receiver_account | VARCHAR | NOT NULL |
| amount | DECIMAL | NOT NULL |
| transaction_type | VARCHAR | NOT NULL (TRANSFER, CASH_OUT) |
| step | INTEGER | NOT NULL |
| old_balance_sender | DECIMAL | |
| new_balance_sender | DECIMAL | |
| old_balance_receiver | DECIMAL | |
| new_balance_receiver | DECIMAL | |
| raw_payload | JSONB | |
| received_at | TIMESTAMP | NOT NULL |

### `fraud_decisions`
| Field | Type | Notes |
|---|---|---|
| decision_id | BIGSERIAL | PRIMARY KEY |
| transaction_id | VARCHAR | FK → transactions |
| fraud_probability | FLOAT | NOT NULL |
| decision | VARCHAR | NOT NULL (ALLOW, REVIEW, BLOCK) |
| risk_level | VARCHAR | NOT NULL (LOW, MEDIUM, HIGH, CRITICAL) |
| explanation | JSONB | |
| neighbour_count | INTEGER | |
| inference_latency_ms | INTEGER | |
| model_version_id | INTEGER | FK → model_versions |
| decided_at | TIMESTAMP | NOT NULL |

### `graph_edges`
| Field | Type | Notes |
|---|---|---|
| edge_id | BIGSERIAL | PRIMARY KEY |
| source_transaction_id | VARCHAR | FK → transactions |
| target_transaction_id | VARCHAR | FK → transactions |
| edge_type | VARCHAR | SAME_SENDER, SAME_RECEIVER |
| created_at | TIMESTAMP | NOT NULL |

### `alerts`
| Field | Type | Notes |
|---|---|---|
| alert_id | BIGSERIAL | PRIMARY KEY |
| decision_id | BIGINT | FK → fraud_decisions |
| pushed_at | TIMESTAMP | NOT NULL |
| acknowledged_by | VARCHAR | nullable |
| acknowledged_at | TIMESTAMP | nullable |
| is_false_positive | BOOLEAN | DEFAULT false |

### `model_versions`
| Field | Type | Notes |
|---|---|---|
| version_id | SERIAL | PRIMARY KEY |
| model_name | VARCHAR | NOT NULL |
| trained_at | TIMESTAMP | NOT NULL |
| auc_roc | FLOAT | |
| precision_score | FLOAT | |
| recall_score | FLOAT | |
| f1_score | FLOAT | |
| training_dataset | VARCHAR | PaySim, IEEE-CIS |
| is_active | BOOLEAN | DEFAULT false |

---

## 10. Functional Requirements Summary (43 total)

**FR-01 to FR-09 — Transaction graph construction:** node creation with 7-feature vector, four edge types with configurable time windows, sliding-window O(n log n) algorithm, PyG `Data` output, graph validation.

**FR-10 to FR-18 — GNN model:** BatchNet GCNConv layers, RealTimeNet SAGEConv inductive layers, Lambda Neural Network fusion, Sigmoid output, weighted BCE loss, 70/15/15 split, early stopping, model persistence, XGBoost baseline.

**FR-19 to FR-24 — Streaming pipeline:** Kafka producer replaying PaySim, FastAPI Kafka consumer (`aiokafka`), Redis neighbour lookup, online subgraph assembly, in-process inference call (no network hop — merged service), Redis self-registration after scoring.

**FR-25 to FR-29 — Alert engine:** four-tier decision thresholds, graph explanation, PostgreSQL audit trail, WebSocket push within 5 seconds, per-account cooldown of 5 minutes.

**FR-30 to FR-37 — Dashboard:** live feed of last 100 transactions colour-coded, WebSocket real-time updates, D3 force graph of last 30 minutes of flagged transactions, fraud rate time series, decision breakdown donut, performance metrics panel, PDF/CSV export, transaction search.

**FR-38 to FR-43 — Training and evaluation:** PaySim primary training, IEEE-CIS generalisation, 7-feature harmonisation between datasets, three-model comparison table, inductive evaluation split, ROC curve visualisation.

---

## 11. Non-Functional Requirements Summary

- **Performance:** inference under 200ms, Redis lookup under 10ms, dashboard update within 5 seconds, Colab training under 4 hours.
- **Scalability:** 10 transactions/second sustained throughput minimum.
- **Reliability:** FastAPI retry with exponential backoff, up to 3 times; Redis TTL auto-expiry.
- **Security:** no raw CSV data transmitted to the inference service; secrets in environment variables, never committed to GitHub.
- **Maintainability:** all five components independently deployable; all thresholds configurable in `config.py` (e.g. `BLOCK_THRESHOLD = 0.75`).
- **Portability:** single `docker compose up` deploys the full production stack; deployable on Render and Vercel free tiers.
- **Usability:** dashboard understandable without ML knowledge; colour-coded decisions with plain-English labels.

---

## 12. Implementation Order — Do Not Deviate

### Phase 1 — Colab (weeks 1–6)
`01_load_paysim.py` → `03_graph_builder.py` → `04_validate_graph.py` → `05_gnn_model.py` (already done) → `06_train_gnn.py` → `07_train_xgboost.py` → `08_evaluate.py`
Output: `model.pt` and results comparison table.

### Phase 2 — Backend (weeks 7–9)
Single FastAPI project (see §7 revision note — merged ML inference + business logic, no Spring Boot). `POST /score` first, tested via curl. Then PostgreSQL models/persistence (SQLAlchemy). Then `GET /api/fraud/results` / `GET /api/fraud/stats` REST endpoints. Then Redis integration. Then a simple synchronous trigger path — **skip Kafka initially**. Then the Kafka consumer added once that path is verified working. Then WebSocket alerts last.

### Phase 3 — Frontend (weeks 10–12)
Next.js project with hardcoded mock data first. `LiveFeed.jsx` → `GraphViz.jsx` → `StatsPanel.jsx` → connect WebSocket. Then `ExportButton.jsx` last.

**Critical rule:** Colab is only for training. Google Colab is never called during production. `model.pt` is downloaded once and deployed to the FastAPI server.

---

## 13. Common Questions for Supervisor Q&A

**Why not implement BRIGHT from scratch?**
BRIGHT is the architectural inspiration. The bi-level concept is implemented using PyG's production-ready `GCNConv` and `SAGEConv` layers. Writing BRIGHT from scratch would take months and produce an inferior result compared to using battle-tested PyG implementations.

**Why PaySim not IEEE-CIS first?**
PaySim has 11 simple columns, direct MFS relevance matching bKash/Nagad transaction types, and requires 30 minutes of preprocessing. IEEE-CIS has 394 columns requiring weeks of feature engineering. PaySim is the standard benchmark for mobile money fraud research.

**Why GraphSAGE specifically?**
GraphSAGE is the only published GNN architecture that explicitly solves the unseen-node problem through inductive aggregation functions. BRIGHT is transductive and cannot score new streaming transactions without full retraining. GraphSAGE integration directly solves BRIGHT's production deployment limitation.

**Why FastAPI-only instead of the original FastAPI + Spring Boot split?**
The original design separated ML inference (Python/FastAPI) from business logic (Java/Spring Boot) — a legitimate polyglot microservice pattern. It was consolidated into a single FastAPI service because the added complexity (a second language to learn from scratch, two independent deployments, an extra network hop on every request eating into the 200ms latency budget) wasn't justified for a solo-developer, time-boxed project. FastAPI's ecosystem (`aiokafka`, SQLAlchemy, native WebSocket support) covers every responsibility Spring Boot would have handled, in one service. The two-service split remains a documented option if there's time to demonstrate it separately.

---

## 13.2 Fraud Ring / Mule-Chain Edge — Not Yet Implemented

The claim in §6 ("the two transactions are connected through the shared nameDest/nameOrig and the GNN detects the chain") requires a **money-flow edge**: connecting a transaction where an account *received* money to a later transaction where that same account *sent* money. This is distinct from the same-sender and same-receiver edges currently built.

This edge type was attempted once (implemented, then reverted) — the first implementation built a Python dict with one DataFrame per unique account (`{acc: df for acc, df in df.groupby(...)}`), which blew up Colab's RAM given PaySim's millions of distinct account IDs. It needs to be rebuilt using a memory-safe approach (e.g. a single sort + vectorized two-pointer scan without materializing a per-account dict of DataFrames) before this section's claim is accurate.

Until this is implemented, the graph only contains `SAME_SENDER` / `SAME_RECEIVER` edges (matches the `graph_edges.edge_type` enum in §9, which currently only lists those two values — will need `MONEY_FLOW` added once built).

---

## 14. Known Gaps vs. Current Implementation (as of this document's writing)

These are differences between this specification and what actually exists in the codebase right now. Listed so nothing downstream (SRS, diagrams, task breakdowns) is generated against an assumption that isn't true yet.

1. **`05_gnn_model.py`** currently implements a plain 2-layer `SAGEConv` classifier with `CrossEntropyLoss` (2-class logits) — not the `BatchNet` + `RealTimeNet` + `LambdaNeuralNetwork` fusion architecture with sigmoid/BCE output described in §3. A separate `05_gnn_model_gcn.py` (`FraudGCN`, GCNConv-only) was built as a standalone comparison baseline rather than as a fused BatchNet component. Needs a rewrite to match §3.
2. **Money-flow / mule-chain edge** does not exist in the current `03_graph_builder.py` — see §13.2.
3. **File plan mismatch:** files currently on disk are `01_load_paysim.py`, `03_graph_builder.py`, `04_validate_graph.py`, `05_gnn_model.py`, `05_gnn_model_gcn.py`, `06_train_compare.py`, `07_xgboost_baseline.py` — numbering and responsibilities differ from the `02_load_ieee_cis.py` / `06_train_gnn.py` / `07_train_xgboost.py` / `08_evaluate.py` / `09_ieee_generalization.py` plan in §7/§12. IEEE-CIS loading, fine-tuning, and generalisation evaluation have not been started.
4. **Decision threshold logic** (§4, four-tier ALLOW/REVIEW/BLOCK-HIGH/BLOCK-CRITICAL off a single sigmoid probability) does not exist yet — current `06_train_compare.py` uses a single F1-tuned binary threshold per model/setting combination, not the four-tier scheme.
5. **XGBoost baseline** (`07_xgboost_baseline.py`) currently trains on the full 7-feature set including balance columns; §3's fraud-pattern signal (near-total origin balance drain) makes this baseline outperform the GNNs by a wide margin on raw metrics. A feature-ablation toggle (excluding balance columns) exists in the current scripts to test whether graph structure adds value once that shortcut is removed — results pending.
6. **Components 3–6** (FastAPI service, Spring Boot backend, Redis, Next.js dashboard) and the **Kafka/Docker Compose** infrastructure have not been started — only Phase 1 (Colab) work exists so far.
