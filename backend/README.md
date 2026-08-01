# Fraud Detection Backend (FastAPI)

Merged ML inference + business logic service — see `architecture.md` section 7
for why this replaced the original FastAPI + Spring Boot split.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

### Database (Postgres via Neon)

1. Sign up at [neon.tech](https://neon.tech), create a project.
2. Copy the connection string from the project dashboard.
3. Paste it into `.env` as `DATABASE_URL`.

Tables are created automatically on first run — no manual SQL needed.

### Redis (live graph memory)

1. Sign up at [upstash.com](https://upstash.com), create a Redis database (free tier).
2. Copy the connection string, paste it into `.env` as `REDIS_URL`.

Not strictly required to run the app — `/score` degrades gracefully (fewer
neighbours found, never crashes) if Redis is unreachable, so you can skip
this and come back to it later.

### Model + preprocessing artifacts

Download three files from Google Drive (`SPL3/models/` for the model,
project root for the other two — see below) into `backend/model/`:

- `FraudGNN_main.pt` (from `train_main_gnn.py`)
- `type_encoder.pkl`, `scaler.pkl` (from `01_load_paysim.py` — **you need to
  re-run that script once** with the updated version that saves these; it's
  just cleaning, not training, so it's quick. Re-fitting on the same CSV
  produces numerically identical parameters to what's already baked into
  `paysim_graph.pt`, so nothing else needs retraining.)

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/docs** — interactive Swagger UI. Log in via
`/auth/login`, click "Authorize" with the returned token, then you can test
every endpoint from the browser without curl.

## Feature scaling — now handled automatically

`/score` accepts **raw** transaction values (real amounts, real balances,
`type` as `"TRANSFER"`/`"CASH_OUT"`). `ml_service.py` applies the exact same
`LabelEncoder`/`StandardScaler` that `01_load_paysim.py` fit during training
before the values reach the model — so real-time inference sees inputs on
the identical scale the model was trained on. No manual pre-scaling needed
by callers (Kafka producer, curl, whatever).

## Smoke test

```bash
curl -X POST http://127.0.0.1:8000/auth/login -d "username=admin&password=admin123"
# copy access_token from the response

curl -X POST http://127.0.0.1:8000/score \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "t1", "sender_account": "A1", "receiver_account": "A2",
    "features": {"step": 5, "type": "TRANSFER", "amount": 181000.0,
                 "oldbalanceOrg": 181000.0, "newbalanceOrig": 0.0,
                 "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
    "neighbours": []
  }'

curl http://127.0.0.1:8000/api/fraud/results -H "Authorization: Bearer <token>"
curl http://127.0.0.1:8000/api/fraud/stats -H "Authorization: Bearer <token>"
```

## What's built vs. what's next

Built: JWT auth (ADMIN/ANALYST), `/score` (real-time inference via
`use_batch=False`, with automatic raw-feature scaling and Redis-backed
neighbour lookup — merged with any client-supplied `neighbours`),
PostgreSQL persistence, results/stats endpoints, alert acknowledgement
(now broadcast live), and a native WebSocket endpoint (`/ws/alerts`) that
pushes every new decision and every acknowledgement to connected clients
in real time.

Not yet wired: Kafka consumer (transactions currently arrive via direct
`/score` calls, not a stream), IEEE-CIS generalisation, the money-flow /
fraud-ring graph edge (architecture.md section 13.2).
