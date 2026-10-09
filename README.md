# FraudScope — Real-Time Graph-Based Fraud Detection for Mobile Financial Services

FraudScope scores every mobile-money transaction in tens of milliseconds and decides
**ALLOW**, **REVIEW** or **BLOCK**. Instead of judging each transaction alone, it also looks
at the sender's and receiver's other recent transactions and scores them together with a
**Graph Neural Network**, so it can catch organised fraud (mule accounts, structuring, smurfing,
agent collusion, account takeover) that looks normal one transaction at a time.

Built for the Bangladeshi MFS setting (bKash, Nagad, Upay): Bangladesh Bank daily limits and a
night-time high-value rule run on top of the model.

> **Live demo:** _add the link here after deployment_

## What it does

- **Real-time scoring** — a transaction plus its related transactions (same sender or same receiver in the last 24 hours, kept in Redis) is scored by `FraudGNN` (GCN batch pathway + GraphSAGE real-time pathway).
- **Explainable decisions** — each related transaction's contribution to the score is measured (leave-one-out), and the exact subgraph the model scored is drawn next to the explanation.
- **Rules on top of the model** — daily limits (TRANSFER BDT 50,000, CASH_OUT BDT 30,000) and a night-time high-value rule.
- **Analyst workflow** — live dashboard feed, alert queue (acknowledge / false positive), SLA-based escalation after 10 minutes.
- **Fraud Patterns demo** — replays six attack patterns and their honest look-alikes through the live pipeline, with an account graph that builds transaction by transaction.
- **Live stream simulator** — realistic MFS traffic (mostly normal, some fraud, some borderline cases) standing in for a Kafka feed.
- **Accounts and security** — Administrator / Analyst roles, cookie-based JWT with CSRF header and refresh rotation, login rate limiting, email-based password reset, light/dark theme.

## Results (retrained model)

| Check | Result |
|---|---|
| Fraud group vs its isolated, transaction-level twin (fraud probability gap) | 0.72 – 1.00 (XGBoost: 0.00) |
| Real-time ROC-AUC on PaySim | 0.992 |
| Automated tests | 188 backend tests + 18 end-to-end browser tests |

The model is trained on PaySim plus injected graph fraud patterns and legitimate look-alikes
(see `INJECTED_PATTERNS.md`). The data is synthetic; see *Limitations* below.

## Tech stack

| Part | Technology |
|---|---|
| Model | PyTorch, PyTorch Geometric (GCN + GraphSAGE), scikit-learn |
| Backend | FastAPI, SQLAlchemy, WebSockets |
| Database | PostgreSQL (Neon), or SQLite for local use |
| Graph memory | Redis (Upstash), or a fake Redis for local use |
| Frontend | Next.js 16, React 19, Tailwind CSS 4 |
| Testing | pytest, Playwright |

## Project layout

```
backend/            FastAPI service: auth, scoring pipeline, decision engine, alerts, simulator, tests
  model/            Trained FraudGNN_main.pt + scaler.pkl + type_encoder.pkl
frontend/           Next.js dashboard
e2e/                Playwright end-to-end tests (also produce the UAT screenshots)
00_inject_fraud_patterns.py, 01_load_paysim.py, 03_graph_builder.py, ...
                    Data pipeline and training scripts (run on Colab / a GPU machine)
train_main_gnn.py, main_gnn_model.py
                    The FraudGNN model and its training script
architecture.md     Architecture, database schema and design decisions
```

## Run it locally

### Prerequisites

- **Python 3.12** and **Node.js 20 or newer**
- About 2 GB of free disk space (PyTorch)
- No cloud account is needed for the quick start below

### 1. Get the code

```bash
git clone https://github.com/Nahidur-rahman-nahid/SPL3.git
cd SPL3
```

Check that `backend/model/` contains `FraudGNN_main.pt`, `scaler.pkl` and `type_encoder.pkl`.

### 2. Install the backend

```bash
cd backend
pip install -r requirements.txt
```

### 3. Choose where the data lives

**Option A: quick start, nothing to sign up for (SQLite + fake Redis).**
Open a **separate terminal** and keep it running; it plays the role of Redis:

```bash
pip install fakeredis
python -c "from fakeredis import TcpFakeServer; s=TcpFakeServer(('127.0.0.1',6379), server_type='redis'); print('fake redis on 6379'); s.serve_forever()"
```

Do **not** create a `.env`: the defaults are SQLite (`fraud.db`) and Redis on `localhost:6379`.
The fake Redis forgets everything when you stop it, which is fine for a demo.

**Option B: cloud services (Neon + Upstash), as in production.**

```bash
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
```

Fill in `.env`:

1. `DATABASE_URL`: a free Postgres from [neon.tech](https://neon.tech) (tables are created automatically).
2. `REDIS_URL`: a free Redis from [upstash.com](https://upstash.com), TCP tab, `rediss://` URL.
3. `SECRET_KEY`: a long random string, from `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
4. `BOOTSTRAP_ADMIN_PASSWORD`: the first admin's temporary password.

If Redis is unreachable the app still runs, but transactions are scored without neighbours
(graph fraud is then missed), and daily limits and rate limits are not enforced.

### 4. Start the backend

```bash
cd backend
python -m uvicorn main:app --reload
```

The API runs on http://127.0.0.1:8000 (interactive docs at `/docs`).

### 5. Start the frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**.

### 6. First login

- Username `admin`; password from `BOOTSTRAP_ADMIN_PASSWORD` (default `change-me-on-first-login` if you made no `.env`).
- You are asked to set a new password straight away.
- Open **Live Stream** and click **Resume Stream** to start the simulated traffic, try **Score** (three ready-made examples), and run a pattern on **Fraud Patterns**.
- Create more users under **Admin → Manage Users**. There is no public sign-up.

## Configuration

Backend settings are environment variables or entries in `backend/.env` (see `backend/config.py`).

| Variable | Purpose | Default |
|---|---|---|
| `SECRET_KEY` | Signs login tokens. **Change it in any real deployment.** | dev-only value |
| `DATABASE_URL` | SQLAlchemy URL (Postgres or SQLite) | `sqlite:///./fraud.db` |
| `REDIS_URL` | Redis for neighbours, daily limits, rate limits | `redis://localhost:6379/0` |
| `BOOTSTRAP_ADMIN_USERNAME`, `BOOTSTRAP_ADMIN_PASSWORD`, `BOOTSTRAP_ADMIN_EMAIL` | First admin, created only when the users table is empty | `admin`, `change-me-on-first-login`, `admin@example.com` |
| `COOKIE_SECURE`, `COOKIE_SAMESITE` | Use `true` and `lax` behind HTTPS | `false`, `lax` |
| `CORS_ORIGINS`, `FRONTEND_BASE_URL` | Address of the frontend | `http://localhost:3000` |
| `STREAM_ENABLED` | Start the simulator automatically | `false` |
| `ALERT_SLA_MINUTES` | Minutes before an unhandled alert escalates | `10` |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM` | Email for password-reset links. When `SMTP_HOST` is empty the link is only written to the backend log. | empty |

The frontend reads `NEXT_PUBLIC_API_BASE_URL` (default `http://127.0.0.1:8000`) from
`frontend/.env.local`. It proxies all API and WebSocket calls through Next.js, so the
browser sees a single origin.

## Tests

**Backend** (uses fake Redis and SQLite, no cloud services needed):

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests
```

**End-to-end browser tests** (Playwright). They build the frontend and start isolated servers
on ports 3001, 8001 and 6390, so they never touch your real data or send email:

```bash
cd e2e
pip install -r requirements.txt
python -m playwright install chromium
python -m pytest
```

Screenshots of every step are saved to `e2e/screenshots/`.

## Retraining the model

The shipped model was trained on Google Colab. In short:

1. `01_load_paysim.py`: load and clean PaySim, fit the scaler and type encoder.
2. `00_inject_fraud_patterns.py`: inject the graph fraud patterns and legitimate look-alikes (`INJECTED_PATTERNS.md` lists them).
3. `03_graph_builder.py`, `04_validate_graph.py`: build and validate the transaction graph.
4. `train_main_gnn.py`: train `FraudGNN` (600 epochs).
5. Copy `FraudGNN_main.pt`, `scaler.pkl` and `type_encoder.pkl` into `backend/model/` and restart the backend.

The PaySim dataset is not in this repository (it is large); download it from Kaggle
(`ealaxi/paysim1`).

## Limitations

- Trained on synthetic data (PaySim plus injected patterns), not real MFS transactions.
- Only `TRANSFER` and `CASH_OUT` are scored; `PAYMENT`, `CASH_IN` and `DEBIT` are not supported yet.
- Graph links are "same sender" and "same receiver" within 24 hours, not full money-flow paths.
- Precision depends on the fraud base rate: even at a false-positive rate under 1%, a large share of flagged transactions in a mostly-honest stream are still honest ones.
- The built-in simulator stands in for a real payment-processor feed (for example Kafka).

## Documentation

- `architecture.md`: architecture, database schema, design decisions
- `INJECTED_PATTERNS.md`: the fraud patterns and look-alikes used for training
- `TEST_CASES.md`: test cases
- `backend/README.md`: backend details and API smoke tests
