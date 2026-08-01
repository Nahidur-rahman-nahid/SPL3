# SPL3 — Real-Time Graph-Based Fraud Detection

A fraud detection system built around a fused bi-level Graph Neural Network
(BatchNet + RealTimeNet + LambdaNeuralNetwork, inspired by BRIGHT and
GraphSAGE), trained on PaySim and served through a real-time FastAPI
backend + Next.js dashboard.

Full architecture, database schema, ER diagram, and design decisions (why
PaySim first, why FastAPI-only, why PostgreSQL over NoSQL, etc.) live in
**[architecture.md](architecture.md)** — that's the source of truth for
this project.

## Layout

- `01_load_paysim.py`, `03_graph_builder.py`, `04_validate_graph.py` — Colab data pipeline: load PaySim, build the transaction graph
- `main_gnn_model.py`, `train_main_gnn.py` — the flagship fused GNN and its training script
- `05_gnn_model.py`, `05_gnn_model_gcn.py`, `06_train_compare.py`, `07_xgboost_baseline.py` — benchmark models (GCN/SAGE/XGBoost) used to justify the fused architecture
- `PrimaryGnnBaseModel.py` — an earlier standalone implementation of the same BatchNet/RealTimeNet architecture, kept for reference
- `backend/` — FastAPI service: JWT auth, real-time scoring, PostgreSQL persistence, Redis-backed neighbour lookup, WebSocket live alerts (see `backend/README.md`)
- `frontend/` — Next.js dashboard: login, live transaction feed, stats, score-and-explain UI with subgraph visualization
- `docker-compose.yml` — local Postgres

## Running it

See `backend/README.md` and `frontend/` for setup instructions for each half of the app. Training/data pipeline instructions are in `architecture.md`.
