# Input for Section 5 — Class-Based Modeling

Paste this whole file into Claude (claude.ai) together with the prompt in §0. Everything below is
taken from the actual running code of the project, so the generated classes match the system.

---

## 0. Prompt to paste

> I am writing the SRS / technical report for my final-year software project (SPL3). Using ONLY
> the project information below, write **Section 5: Class-Based Modeling** with this structure:
>
> - **5.1 Analysis Classes** — first a short paragraph on how the classes were identified (from
>   the noun listing of the usage scenarios), then a table of the selected analysis classes with
>   their stereotype (Entity / Boundary / Control) and a one-line description. Also list the
>   nouns that were rejected and why.
> - **5.2 Class Cards** — one CRC class card per analysis class, as a table with: Class name,
>   Type (stereotype), Description, Attributes, Responsibilities (operations), Collaborators.
> - **5.3 CRC Diagram** — the relationships between classes (association, aggregation,
>   composition, dependency, generalization) with multiplicity, given as (a) a relationship
>   table and (b) Mermaid `classDiagram` code I can render into an image.
>
> Use formal report language. Keep attribute and method names as given. Do not invent features
> that are not listed.

---

## 1. Project in short

**Title:** Graph Neural Network–based Real-Time Fraud Detection for Mobile Financial Services
(bKash / Nagad / Upay style MFS in Bangladesh).

**Idea:** Every incoming transaction (TRANSFER or CASH_OUT) is scored in real time. The system
finds the transaction's *related* recent transactions (same sender or same receiver within
24 hours) from a Redis cache, builds a small graph around it, and a GNN (FraudGNN) outputs a
fraud probability. A decision engine converts it into ALLOW / REVIEW / BLOCK, explains which
related transactions raised the score, applies Bangladesh Bank rules, stores the result and
pushes an alert to analysts over WebSocket. Unacknowledged alerts are escalated after an SLA.

**Tech:** FastAPI (Python) backend, PostgreSQL (Neon) via SQLAlchemy, Redis (Upstash) as the
live graph cache, PyTorch Geometric model, Next.js frontend dashboard.

**Actors:**
- **Administrator (ADMIN)** — manages user accounts, deletes/cleans up transaction data.
- **Fraud Analyst (ANALYST)** — scores transactions, reviews/acknowledges alerts, watches the live
  stream, runs the Fraud Patterns demo.
- **Stream Simulator** (background process) — generates live transactions automatically.
- **Escalation Watcher** (background process) — escalates overdue alerts every 60 s.

---

## 2. Business rules (used inside class responsibilities)

| Rule | Value |
|---|---|
| Decision tiers | p < 0.40 → ALLOW / LOW; 0.40–0.75 → REVIEW / MEDIUM; 0.75–0.90 → BLOCK / HIGH; ≥ 0.90 → BLOCK / CRITICAL |
| Alert threshold | an alert is created when p ≥ 0.40 |
| Bangladesh Bank daily limit | CASH_OUT 30,000 BDT/day, TRANSFER 50,000 BDT/day per sender → exceeding = BLOCK, risk "REGULATORY" |
| Night rule | amount ≥ 10,000 BDT between 23:00–03:59 (Bangladesh time) → at least REVIEW |
| Neighbour window | same sender / same receiver, last 24 h, max 50 per account (Redis TTL 24 h) |
| Alert SLA | unacknowledged for 10 min → escalated |
| Login lockout | 5 failed attempts → locked 15 min |
| Tokens | access token 15 min, refresh token 7 days (rotated; reuse revokes the whole family), CSRF token on every state-changing request |

---

## 3. Noun listing → candidate classes

| Noun from scenarios | Decision | Reason |
|---|---|---|
| User (Admin, Analyst) | **Class** (Entity) | stored, has behaviour |
| Refresh Token | **Class** (Entity) | stored, used for session rotation |
| Transaction | **Class** (Entity) | core stored object |
| Fraud Decision | **Class** (Entity) | stored result of each scoring |
| Alert | **Class** (Entity) | stored, has review/escalation state |
| Neighbour (related transaction) | **Class** (Entity, transient) | needed for graph + explanation |
| Explanation | attribute of FraudDecision | stored as JSON, no own behaviour |
| Decision / Risk level | attribute | enumerated values only |
| Fraud Detection Model (GNN) | **Class** (Control) | does the inference |
| Graph Cache (Redis) | **Class** (Boundary) | external store of recent transactions |
| Scoring Pipeline | **Class** (Control) | coordinates the whole scoring flow |
| Decision Engine | **Class** (Control) | tiers, rules, explanation |
| Authentication Service | **Class** (Control) | login, tokens, CSRF |
| User Service | **Class** (Control) | user CRUD |
| Login Rate Limiter | **Class** (Control) | brute-force protection |
| Alert Notifier (WebSocket manager) | **Class** (Boundary) | pushes live events to dashboards |
| Escalation Watcher | **Class** (Control, active) | background SLA check |
| Stream Simulator | **Class** (Control, active) | background transaction generator |
| Transaction Admin (data cleanup) | **Class** (Control) | delete / purge transactions |
| Pattern Demo | **Class** (Control) | replays fraud patterns for demonstration |
| Dashboard / UI pages | **Class** (Boundary) | user interface |
| API Gateway (REST endpoints) | **Class** (Boundary) | entry point for all requests |
| JWT / access token | rejected | stateless, never stored |
| Model version, Graph edge | rejected | tables exist but never used by any code path |
| Organization / tenant | rejected | one dedicated deployment per MFS client |
| Kafka | rejected | future work (simulator is used instead) |

---

## 4. Class details (attributes + operations from the code)

### Entity classes (database tables, `backend/models.py`)

**User**
- Attributes: id, username (unique), email, hashed_password, role (ADMIN | ANALYST), is_active,
  must_change_password, created_by_id (FK → User), created_at, last_login_at
- Operations (via UserService/Auth): authenticate(), changePassword(), resetPassword(),
  activate()/deactivate(), changeRole()
- Relations: an ADMIN creates many Users (self-association); a User owns many RefreshTokens;
  a User acknowledges many Alerts.

**RefreshToken**
- Attributes: id, user_id (FK → User), token_hash, family_id, expires_at, revoked_at, created_at
- Operations: issue(), rotate(), revoke(), revokeFamily()

**Transaction**
- Attributes: transaction_id (PK), sender_account, receiver_account, amount, transaction_type
  (TRANSFER | CASH_OUT), step, old_balance_sender, new_balance_sender, old_balance_receiver,
  new_balance_receiver, raw_payload, received_at
- Operations: save(), delete()
- Relations: one Transaction has 1..* FraudDecisions.

**FraudDecision**
- Attributes: decision_id (PK), transaction_id (FK), fraud_probability, decision
  (ALLOW | REVIEW | BLOCK), risk_level (LOW | MEDIUM | HIGH | CRITICAL | REGULATORY),
  explanation (JSON: summary, neighbour list with edge_type + contribution + prior decision),
  neighbour_count, inference_latency_ms, decided_at
- Relations: belongs to one Transaction; may raise 0..1 Alert.

**Alert**
- Attributes: alert_id (PK), decision_id (FK), pushed_at, acknowledged_by, acknowledged_at,
  is_false_positive, escalated, escalated_at
- Operations: acknowledge(isFalsePositive), escalate(), isOverdue(now, sla)

**NeighbourInfo** (transient, not stored)
- Attributes: transaction_id, edge_type (SAME_SENDER | SAME_RECEIVER), features
- Used to build the graph and the explanation.

### Control classes (`backend/*.py`)

**ScoringPipeline** (`scoring_pipeline.py`)
- processTransaction(request, source), gatherNeighbours(request),
  checkRegulatoryLimits(request), releaseReservation(), scoreAndRecord()
- Collaborators: GraphCache, FraudDetectionModel, DecisionEngine, Transaction, FraudDecision,
  Alert, AlertNotifier

**FraudDetectionModel** (`ml_service.py` + `gnn_model.py`)
- Attributes: model (FraudGNN = BatchNet + RealTimeNet + LambdaNeuralNetwork), scaler,
  type_encoder, money_transform, model_info
- Operations: loadModel(), score(target, neighbours) → (probability, latency),
  featuresToVector(), infer(), modelInfo()
- Composition: FraudGNN is composed of BatchNet (GCN), RealTimeNet (GraphSAGE) and
  LambdaNeuralNetwork (combiner).

**DecisionEngine** (`decision_engine.py`)
- tierDecision(probability), shouldAlert(probability), applyNightRule(request, decision),
  buildExplanation(request, neighbours, baseProbability) — leave-one-out contribution of each
  neighbour, contributionScores(), summarySentence()
- Collaborators: FraudDetectionModel (re-scores without each neighbour), FraudDecision (prior
  decisions of neighbours)

**AuthService** (`auth.py`)
- hashPassword(), verifyPassword(), createAccessToken(), decodeAccessToken(),
  getCurrentUser(), requireRole(roles), issueRefreshToken(), rotateRefreshToken(),
  revokeRefreshToken(), generateCsrfToken(), verifyCsrf(), setSessionCookies(),
  clearSessionCookies()
- Collaborators: User, RefreshToken, LoginRateLimiter

**UserService** (`user_service.py`)
- getUserByUsername(), listUsers(), createUser() (returns temporary password), updateUser(),
  resetPassword(), changePassword(), authenticate(), bootstrapAdmin()
- Collaborators: User, AuthService

**LoginRateLimiter** (`rate_limit.py`)
- isLockedOut(username, ip), recordFailedAttempt(), clearAttempts()
- Collaborator: GraphCache (Redis counters)

**EscalationWatcher** (`escalation.py`, background loop every 60 s)
- runEscalationWatch(), isOverdue(alert, now, sla), notifyEscalation(alert)
- Collaborators: Alert, AlertNotifier

**StreamSimulator** (`stream_simulator.py`, background loop every 2 s)
- generateTransaction() (normal / drain / suspicious / ring), runStream(), setEnabled(),
  isEnabled()
- Collaborator: ScoringPipeline

**TransactionAdmin** (`transaction_admin.py`, ADMIN only)
- deleteTransactions(ids), purge(source = stream | manual | all, before), countMatching()
- Deletes Alerts → FraudDecisions → Transactions together and clears related Redis entries.
- Collaborators: Transaction, FraudDecision, Alert, GraphCache, AlertNotifier

**PatternDemo** (`pattern_demo.py`)
- listPatterns(), runPattern(patternId, variant = fraud | lookalike)
- 6 patterns: Mule hub, Structuring, Smurf fan-out, Agent collusion, Account takeover,
  Mule network — each with a legitimate look-alike.
- Collaborators: ScoringPipeline, FraudDetectionModel

### Boundary classes

**GraphCache** (`redis_client.py`, Redis)
- getRecentTransactions(account), storeTransaction(), getDailyTotal(account, type),
  reserveDailyAmount(), releaseDailyAmount(), forgetTransactions()

**AlertNotifier** (`websocket_manager.py` → ConnectionManager)
- Attributes: active_connections
- connect(websocket), disconnect(websocket), broadcast(message)

**APIGateway** (`main.py`, FastAPI REST + WebSocket)
- /auth/login, /auth/refresh, /auth/logout, /auth/me, /auth/change-password, /auth/users (CRUD,
  reset-password), /score, /api/fraud/results, /api/fraud/stats,
  /api/fraud/alerts/{id}/acknowledge, /api/stream/status, /api/stream/toggle,
  /api/transactions (delete), /api/transactions/purge, /api/patterns, /api/patterns/{id}/run,
  /ws/alerts

**Dashboard UI** (Next.js pages)
- Login, Dashboard (stats + live feed), Score (manual scoring form + graph view), Fraud Patterns,
  Alerts (review queue), Live Stream, Admin → Manage Users, Admin → Data Cleanup, Account
  (change password)

---

## 5. Relationships (for the CRC diagram)

| From | Relationship | To | Multiplicity |
|---|---|---|---|
| User (ADMIN) | creates (self-association) | User | 1 → 0..* |
| User | owns | RefreshToken | 1 → 0..* |
| User | acknowledges | Alert | 1 → 0..* |
| Transaction | is scored by (composition) | FraudDecision | 1 → 1..* |
| FraudDecision | raises | Alert | 1 → 0..1 |
| FraudDecision | references | NeighbourInfo (in explanation) | 1 → 0..* |
| FraudGNN | composed of | BatchNet, RealTimeNet, LambdaNeuralNetwork | 1 → 1 each |
| Admin, Analyst | generalization of | User (role) | — |
| ScoringPipeline | uses | GraphCache, FraudDetectionModel, DecisionEngine, AlertNotifier | dependency |
| ScoringPipeline | creates | Transaction, FraudDecision, Alert | dependency |
| StreamSimulator, PatternDemo | call | ScoringPipeline | dependency |
| EscalationWatcher | updates | Alert; notifies AlertNotifier | dependency |
| TransactionAdmin | deletes | Transaction, FraudDecision, Alert; clears GraphCache | dependency |
| AuthService | uses | User, RefreshToken, LoginRateLimiter | dependency |
| APIGateway | delegates to | AuthService, UserService, ScoringPipeline, TransactionAdmin, PatternDemo | dependency |
| Dashboard UI | calls | APIGateway; listens to AlertNotifier | dependency |

---

## 6. Main scoring flow (helps write responsibilities correctly)

1. Transaction arrives (Analyst score form, Stream Simulator, or Pattern Demo) → APIGateway → ScoringPipeline.
2. ScoringPipeline checks the Bangladesh Bank daily limit with GraphCache (skipped for stream/demo).
3. GraphCache returns related transactions (same sender / same receiver, last 24 h) → NeighbourInfo list.
4. FraudDetectionModel builds a star graph and returns fraud probability + latency.
5. DecisionEngine → tier (ALLOW/REVIEW/BLOCK), night rule, explanation with per-neighbour contribution.
6. Transaction + FraudDecision saved; Alert created if p ≥ 0.40; transaction stored in GraphCache.
7. AlertNotifier broadcasts the result/alert to all open dashboards.
8. Analyst acknowledges the alert (or marks false positive); if not within 10 min, EscalationWatcher escalates it.
