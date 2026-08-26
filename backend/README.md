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

On first run, a bootstrap admin account is created automatically from
`BOOTSTRAP_ADMIN_USERNAME`/`BOOTSTRAP_ADMIN_PASSWORD` in `.env` (only if the
`users` table is empty). Every other account is created afterwards through
that admin's account — see "Auth" below.

Open **http://127.0.0.1:8000/docs** for interactive testing. Auth is
cookie-based now (not a bearer token), so Swagger's "Authorize" button
doesn't apply the same way — either drive it from the frontend (which sends
cookies automatically), or use curl with a cookie jar (see Smoke test).

## Auth

No public signup — accounts are created by an existing ADMIN via
`POST /auth/users`, which returns a one-time temp password
(`must_change_password` is set on the new account, forcing a change at first
login via `POST /auth/change-password`). `POST /auth/login` sets three
cookies: `access_token` (short-lived, httpOnly), `refresh_token`
(long-lived, httpOnly, rotated on every use via `POST /auth/refresh`), and
`csrf_token` (**not** httpOnly — read it client-side and echo it back as an
`X-CSRF-Token` header on every mutating request; see `auth.py`'s docstring
for why). `GET /auth/me` returns the current session's user profile.

Local dev (`http://localhost:3000` talking to `http://localhost:8000`) works
with the default `COOKIE_SECURE=False`/`COOKIE_SAMESITE=lax` because both are
"localhost" — cookie scoping ignores port. A real deployment with
frontend/backend on different domains needs `COOKIE_SECURE=True`,
`COOKIE_SAMESITE=none`, and `CORS_ORIGINS` set to the real frontend origin —
see `.env.example` for details and caveats (Safari/Chrome third-party-cookie
restrictions).

## Feature scaling — now handled automatically

`/score` accepts **raw** transaction values (real amounts, real balances,
`type` as `"TRANSFER"`/`"CASH_OUT"`). `ml_service.py` applies the exact same
`LabelEncoder`/`StandardScaler` that `01_load_paysim.py` fit during training
before the values reach the model — so real-time inference sees inputs on
the identical scale the model was trained on. No manual pre-scaling needed
by callers (Kafka producer, curl, whatever).

## Smoke test

Cookie-based auth needs a cookie jar, and mutating endpoints need the CSRF
header echoed back:

```bash
curl -c cookies.txt -X POST http://127.0.0.1:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "<your BOOTSTRAP_ADMIN_PASSWORD>"}'

CSRF=$(grep csrf_token cookies.txt | awk '{print $7}')

curl -b cookies.txt http://127.0.0.1:8000/auth/me

curl -b cookies.txt -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" -H "X-CSRF-Token: $CSRF" \
  -d '{
    "transaction_id": "t1", "sender_account": "A1", "receiver_account": "A2",
    "features": {"step": 5, "type": "TRANSFER", "amount": 181000.0,
                 "oldbalanceOrg": 181000.0, "newbalanceOrig": 0.0,
                 "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
    "neighbours": []
  }'

curl -b cookies.txt http://127.0.0.1:8000/api/fraud/results
curl -b cookies.txt http://127.0.0.1:8000/api/fraud/stats
```

## What's built vs. what's next

Built: DB-backed accounts (ADMIN/ANALYST) with cookie sessions, refresh-token
rotation with reuse/theft detection, admin-only user management (create,
deactivate, change role, reset password — no public signup, no email —
see "Auth" above), RBAC on admin-only endpoints, `/score` (real-time
inference via `use_batch=False`, with automatic raw-feature scaling and
Redis-backed neighbour lookup — merged with any client-supplied
`neighbours`), PostgreSQL persistence, results/stats endpoints, alert
acknowledgement (now broadcast live, attributed to the authenticated user),
and a native WebSocket endpoint (`/ws/alerts`) that pushes every new
decision and every acknowledgement to connected clients in real time.

Not yet wired: Kafka consumer (transactions currently arrive via direct
`/score` calls, not a stream), IEEE-CIS generalisation, the money-flow /
fraud-ring graph edge (architecture.md section 13.2).
