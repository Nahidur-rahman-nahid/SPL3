# SRS Source Content — GNN Fraud Detection Platform

This document contains the **full content** needed to write the SRS/technical report (Project
Overview, Requirements Analysis, Usage Scenario, System Modeling, Data & Information Modeling,
Test Plan), in the same section structure as the reference midterm template.

**Important framing:** the system was implemented first, in a basic form, before this SRS was
written. This document describes the **target design** at a deliberately simple, four-table scope
— a real `User` table replacing today's two hardcoded accounts, alongside the three tables that
already exist (`Transaction`, `FraudDecision`, `Alert`). Where the target design differs from
what's literally running in code today, that is called out explicitly so the report stays honest
about implementation status.

**Deployment model decision — one dedicated deployment per client, not shared multi-tenant SaaS.**
An earlier draft of this SRS designed a shared multi-tenant platform (an `Organization` table,
`org_id` scoping on every row, Row-Level Security policies) so bKash, Nagad, and Upay could all run
on one shared instance. That has been deliberately dropped in favor of a simpler model: **each
client organization gets its own dedicated deployment** (its own database, its own backend
instance). The reasoning:
- bKash, Nagad, and Upay are direct competitors. None of them would accept sharing *any*
  infrastructure with a competitor for fraud/transaction data, regardless of how strong the
  row-level isolation guarantees are — the trust model for this domain expects physical separation.
- It removes an entire category of implementation risk (a forgotten `org_id` filter leaking one
  client's data to another) by making cross-client leakage structurally impossible rather than
  merely policy-enforced.
- It matches the four core tables actually chosen for this design: `User`, `Transaction`,
  `FraudDecision`, `Alert` — no `Organization` or `ApiKey` table, because a single deployment only
  ever serves one client, so there's nothing to scope by.

**Explicit exclusions from the ER diagram / schema:**
- `Token` / `JWT` — a JWT is stateless, signed and verified, never persisted server-side.
- `ModelVersion` — defined in the current codebase's ORM models but never actually written to by
  any code path (`FraudDecision.model_version_id` stays null everywhere); model-version lineage
  tracking isn't a stated objective of this system, so it is deliberately left out rather than
  modeled speculatively.
- `GraphEdge` — also defined in the codebase and also never written to by any code path today (not
  by live scoring, not by the offline pipeline). Unlike `ModelVersion`, it isn't deleted from the
  discussion entirely — it's the one piece of infrastructure a *future* multi-hop cluster-graph
  view would need — but it is **not part of the core four-table design**; see §7 Roadmap.
- `Organization` / `ApiKey` / `Account` — considered in an earlier draft, dropped per the
  deployment-model decision above.

---

## 1. Project Overview

### 1.1 Project Title
Graph Neural Network Fraud Detection Platform — a real-time transaction fraud detection and review
system for a mobile financial service (MFS) provider, deployed one dedicated instance per client.

### 1.2 Problem Statement
Mobile financial services (bKash, Nagad, Upay-style mobile money operators) process high-volume
peer-to-peer transfers and cash-outs where fraud rings launder money through short-lived "mule"
accounts — a victim's account is drained to a receiver who is itself part of a chain of transfers,
often completing within minutes. Row-by-row tabular fraud models miss this because they score each
transaction in isolation; the fraud signal is in the *relationships* between transactions (shared
accounts, tight time windows), not any single transaction's own features. Conventional monitoring
also cannot distinguish "this looks unusual" from "this is structurally connected to other flagged
activity" — it can flag an outlier but not explain which relationships made it suspicious.

### 1.3 Objectives
- **O1 — Graph-based fraud representation:** model transactions as nodes connected by shared
  account relationships (not isolated rows), so structural fraud patterns are visible to the model.
- **O2 — Real-time scoring:** score an incoming transaction and return a decision within
  milliseconds, using only the inductive (no full-graph-reload) pathway of the trained model.
- **O3 — Explainable decisions:** every flagged transaction must come with a concrete explanation
  of *which* relationships drove the score and how much each contributed — not just a probability.
- **O4 — Human-in-the-loop review with escalation:** flagged transactions enter a review queue; if
  no analyst acts within an SLA window, the system escalates automatically.
- **O5 — Client isolation via dedicated deployment:** each client organization runs its own
  instance of the platform — isolation is a property of the deployment topology, not of row-level
  application logic.
- **O6 — Controlled ingestion:** accept transactions from a dashboard-triggered manual score, the
  client's own backend calling the scoring API, or — at target maturity — a Kafka topic, all through
  the identical scoring/decision code path, and never start ingesting on its own without an
  explicit action.
- **O7 — Live operational visibility:** a real-time dashboard, alert review queue, and live stream
  monitor so analysts can watch the system work without manually triggering every check.

### 1.4 Scope

**In scope:**
- Real user accounts and login (replacing today's two hardcoded credentials), with a bootstrap flow
  for the very first administrator on a fresh deployment.
- A trained fused Graph Neural Network (GCN + GraphSAGE) served for real-time inference.
- An offline data pipeline (PaySim-based) that builds the transaction graph and trains the model.
- Redis-backed short-term "live graph memory" for real-time neighbour lookup.
- PostgreSQL persistence of every transaction, every decision (full audit trail), and every alert.
- A pluggable transaction ingestion path: an in-process simulator today (defaults to paused —
  started explicitly from the dashboard), a real Kafka consumer at target maturity, both feeding
  one shared scoring pipeline.
- A dashboard: auth, stats + manual scoring + live feed, a dedicated alerts/review-queue page, and
  a live stream monitor page.

**Out of scope:**
- Non-financial anomaly domains (this is transaction fraud only).
- Automated fund freezing/reversal at the banking core — this system recommends
  ALLOW/REVIEW/BLOCK and raises alerts; it does not itself move money.
- A shared multi-tenant SaaS platform, tenant self-service registration, and per-tenant API Keys
  (deliberately dropped — see the deployment-model decision above).
- Billing/subscription-plan management.
- Chat-based / narrative root-cause summarization.
- Per-client custom-trained models (one model per deployment, trained centrally and shipped as a
  checkpoint — retraining per client is not planned).
- Model-version lineage tracking (see exclusion note above).

### 1.5 Deliverables
| # | Deliverable | Description |
|---|---|---|
| 1 | Backend | FastAPI: user auth (bootstrap + login), real-time scoring, decision/alert engine, escalation watcher, REST + WebSocket API |
| 2 | Trained Fused GNN | `FraudGNN_main.pt` + fitted `LabelEncoder`/`StandardScaler`, loaded once at startup |
| 3 | Offline Data & Graph Pipeline | PaySim ingestion, graph construction, model training/benchmarking scripts |
| 4 | Transaction Ingestion Layer | In-process simulator (current, off by default) / Kafka consumer (target), both calling one shared scoring pipeline |
| 5 | Dashboard | Next.js SPA: auth, stats/scoring/live-feed page, alerts & review-queue page, live stream monitor page |
| 6 | Documentation | Architecture reference, this SRS, patterns/QA reference |

---

## 2. Requirements Analysis

### 2.1 Functional Requirements

Grouped by capability area so related requirements read as a coherent set rather than near-identical
restatements of one pipeline.

**A. Authentication & User Management**
| ID | Functional Requirement |
|---|---|
| FR-1 | On a fresh deployment with no users yet, the system shall allow exactly one bootstrap request to create the first `ADMIN` account; the bootstrap path shall refuse to run again once any user exists. |
| FR-2 | The system shall allow an `ADMIN` to create, deactivate, and list `ANALYST` (or additional `ADMIN`) accounts. |
| FR-3 | The system shall authenticate a user by username and password and issue a signed JWT carrying their role, with no self-service public registration. |

**B. Real-Time Transaction Scoring**
| ID | Functional Requirement |
|---|---|
| FR-4 | The system shall accept a transaction for scoring — from the dashboard or from the client's own backend — containing its raw feature values and sender/receiver account identifiers. |
| FR-5 | The system shall assemble the transaction's live subgraph from each account's recent transaction history in the live graph cache, tagging every neighbour found with the relationship (same-sender / same-receiver) that produced it, and run real-time GNN inference over it to produce a fraud probability between 0 and 1. |
| FR-6 | The system shall convert the fraud probability into a decision (`ALLOW`/`REVIEW`/`BLOCK`) and risk level using configurable thresholds, and generate a relationship-based explanation — per neighbour, its relationship type, whether it was itself previously flagged, and a leave-one-out contribution score. |
| FR-7 | The system shall persist every scored transaction and its full decision permanently, regardless of tier, so the audit trail is never selective. |

**C. Alerting & Review**
| ID | Functional Requirement |
|---|---|
| FR-8 | The system shall raise an alert whenever a transaction's fraud probability clears a configurable alert threshold, independent of the ALLOW/REVIEW/BLOCK tiering thresholds. |
| FR-9 | The system shall let an Analyst view and filter the review queue by Needs Review / Escalated / All. |
| FR-10 | The system shall let an Analyst acknowledge an alert, optionally marking it a false positive, recorded against their own authenticated identity rather than free text. |
| FR-11 | The system shall automatically mark an alert "escalated" and broadcast a distinct notification if it remains unacknowledged past a configurable SLA window. |

**D. Live Ingestion & Dashboard**
| ID | Functional Requirement |
|---|---|
| FR-12 | The system shall allow the transaction feed (simulator today, Kafka consumer at target maturity) to be explicitly started and paused; it shall default to paused on every startup and never begin ingesting on its own. |
| FR-13 | The system shall push real-time scoring, acknowledgment, and escalation events to every connected dashboard over an authenticated WebSocket channel. |
| FR-14 | The dashboard shall display aggregate statistics — total scored, per-tier counts, average latency. |
| FR-15 | The dashboard shall let a user manually submit a transaction and view a subgraph visualization of exactly the neighbours the model used for it. |
| FR-16 | The dashboard shall provide a live stream monitor view with connection status, throughput, a real-time transaction ticker, and an explicit start/stop control. |

### 2.2 Non-Functional Requirements
| Category | Requirement |
|---|---|
| Performance | Real-time scoring — neighbour lookup, GNN inference, explanation/contribution computation — must complete well within a sub-second budget; the dashboard must reflect a new event within a few seconds of it occurring. |
| Security | Passwords must never be stored in plaintext (bcrypt hashing only); dashboard and API traffic is authenticated via JWT Bearer tokens; the WebSocket handshake is authenticated via a token query parameter (browsers cannot set custom headers on a WebSocket handshake). |
| Deployment Isolation | Each client organization runs on its own dedicated deployment (separate database, separate backend instance) — isolation is achieved at the infrastructure level, not via shared-database row filtering, which is the appropriate model for clients who are direct competitors. |
| Availability | A live-graph-cache outage degrades scoring gracefully (fewer neighbours found, not a failed request); the ingestion loop and the escalation watcher run as independent background processes that cannot block request handling. |
| Predictability | Background processes never take an action a user didn't explicitly request — the transaction feed never starts itself; it always requires an explicit start action, defaulting to off. |
| Usability | The review queue defaults to showing only actionable (REVIEW/BLOCK, unacknowledged) items; an escalated alert is visually distinguished without requiring a manual refresh. |
| Maintainability | The Detection Engine (pure GNN inference) and the Alert & Decision Engine (tiering, explanation, alert policy) are separate modules connected by a single call boundary, so either can be modified — a different model, a different alerting policy — without touching the other. |
| Auditability | Every scored transaction, not only flagged ones, produces a permanent decision record; every alert acknowledgment is tied to a real, authenticated user identity, not free text. |

### 2.3 Business rules / thresholds (exact values to use in NFR / decision-logic descriptions)
| Rule | Value |
|---|---|
| ALLOW tier | fraud probability < 0.40 |
| REVIEW tier | 0.40 ≤ probability < 0.75 |
| BLOCK tier (HIGH) | 0.75 ≤ probability < 0.90 |
| BLOCK tier (CRITICAL) | probability ≥ 0.90 |
| Alert raised | probability ≥ 0.40 (independently configurable, not derived from the tiering thresholds) |
| Escalation SLA | 10 minutes unacknowledged |
| Escalation check interval | every 60 seconds |
| Neighbour lookup window | 24 hours |
| Live-cache neighbour cap | 50 most recent per account |
| Contribution-scoring cap | 15 neighbours (bounds re-inference latency) |
| JWT expiry | 60 minutes |
| Transaction feed default state | **disabled at startup** — requires an explicit start action from the dashboard, never auto-starts |

### 2.4 Stakeholders
| Stakeholder | Role / Interest |
|---|---|
| Fraud Analyst (`ANALYST` role) | Primary daily user — reviews the Alerts & Review Queue, acknowledges/dismisses alerts, watches the live feed |
| Administrator (`ADMIN` role) | The first bootstrapped account on a deployment; manages that deployment's other users; also has Analyst capabilities |
| MFS Client Organization (bKash / Nagad / Upay) | The paying customer; each runs its own dedicated deployment, staffed by the roles above |
| Platform Engineering Team | Builds the backend, the model pipeline, and the dashboard; deploys a new instance per client |
| End customers of the MFS provider | Indirect beneficiaries — protected from fraud without needing to know this system exists |
| Course Supervisor / Evaluator | Assesses the project against course deliverable criteria |

---

## 3. Usage Scenario

### 3.1 Deployment Setup & User Bootstrap
When a new client (e.g. bKash) is onboarded, the engineering team stands up a dedicated deployment
for them — a fresh database, a fresh backend instance. On first startup there are no users at all.
A one-time bootstrap request creates the very first `ADMIN` account; the same request path refuses
to run again once any user row exists, so there's no standing "create the first admin" door left
open. This is deliberately not a public self-service sign-up — the client is a regulated financial
institution, and standing up their instance is part of a delivery/security-review process, not a
form anyone can submit. Once logged in, that `ADMIN` creates `ANALYST` accounts for their own staff
directly from the dashboard.

### 3.2 Authentication
A user logs in with a username and password. The system verifies the bcrypt hash and issues a
signed JWT carrying their identity and role (60-minute expiry). Every subsequent request presents
this token as a Bearer header (or, for the WebSocket connection, as a query parameter). The
client's own backend system, when it wants to submit transactions programmatically rather than
through the dashboard, uses this same login flow via a dedicated service account — there is no
separate long-lived API-key mechanism in this design (see §7 Roadmap for that as a future
refinement).

### 3.3 Real-Time Transaction Scoring (Detection Engine)
A transaction arrives — via the dashboard's manual scoring form, the client's own backend calling
the scoring API, or the ingestion feed (simulator today, Kafka at target maturity) — with its raw
feature values and account identifiers. The system looks up the sender's and receiver's recent
transaction history from the live graph cache, tags each result with the relationship that
surfaced it, merges in any explicitly supplied neighbours, and assembles a small star-shaped
subgraph. This subgraph is fed through the trained model's real-time (inductive) pathway to produce
a single fraud probability. This step — and only this step — is the Detection Engine; it has no
concept of thresholds, alerts, or explanations.

### 3.4 Decision, Explanation & Alerting (Alert & Decision Engine)
Given the raw probability, a separate module converts it into a decision tier and risk level, and
independently decides whether the probability clears the alert threshold. If neighbours exist, the
system builds a relationship-based explanation: for each neighbour, which edge type connected it,
whether it was itself previously flagged, and a leave-one-out contribution score. Every scored
transaction — regardless of tier — is permanently persisted with its full explanation. If the alert
threshold is cleared, a review-queue entry is created and broadcast live to every connected
dashboard.

### 3.5 Review & Escalation
An Analyst working the Alerts & Review Queue page sees every alert-worthy transaction, with its
relationship explanation expandable per row, and can acknowledge it (optionally marking it a false
positive) — recorded against their own user identity. A background watcher runs every 60 seconds;
any alert that has sat unacknowledged past the SLA window is marked escalated, logged
server-side, and broadcast as a distinct real-time event.

### 3.6 Live Transaction Ingestion
A background feed can generate realistic transactions (mostly normal payments, some suspicious
drain patterns, occasional ring-shaped fraud patterns) and route each one through the shared
scoring path described above — but it never runs unless explicitly started: the feed defaults to
paused on every server start, and a user must press "Resume Stream" on the dashboard before any
transaction flows. This is an explicitly documented stand-in for a real Kafka consumer — swapping
it in changes only "where a transaction comes from," not the scoring, decision, or alerting logic.

### 3.7 Dashboard & Real-Time Monitoring
The dashboard opens a JWT-authenticated WebSocket connection and shows: aggregate stats, a manual
scoring form with a live subgraph diagram of exactly what the model saw, and a live feed table of
every transaction scored. The separate Alerts page and Stream Monitor page open their own WebSocket
connections and render the same event stream filtered/formatted for their specific purpose.

---

## 4. System Modeling

### 4.1 Actors

**Primary actors** (each one initiates a use case on its own — including the two autonomous
background processes, which start their own use cases on a timer with no external trigger, the
same way the reference template treats its in-cluster Agent as primary rather than secondary):
- **Administrator (`ADMIN`)** — bootstraps and manages this deployment's users.
- **Analyst (`ANALYST`)** — scores transactions manually, reviews and acknowledges alerts, watches
  the live feed and stream monitor.
- **Client Backend System** — the client's own external service, authenticated as a service
  account, submitting transactions programmatically.
- **Ingestion Process** — the background simulator today / Kafka consumer at target maturity;
  autonomously drives the "Ingest & Detect" use case once started.
- **Escalation Watcher** — background process that autonomously drives the "Escalate" use case on
  its own 60-second timer.

**Secondary actors** (queried/used in service of a primary actor's use case; never initiate one
themselves):
- **Detection Engine (GNN inference)** — invoked mid-flow to produce a raw probability.
- **Live Graph Cache (Redis)** — external dependency queried/updated during scoring.

### 4.2 Use case levels

**Level 0 — System overview:** Bootstrap & Manage Users, Score a Transaction, Manage Alerts &
Review Queue, Control & Monitor Live Stream, (background) Ingest & Detect, (background) Escalate.

**Level 1.1 — User Bootstrap & Management**
Primary actors: Administrator, Analyst
- Action: the first request is made against a deployment with zero users.
  Reply: system creates that request's account as `ADMIN`; the bootstrap path is now permanently
  closed for this deployment.
- Action: `ADMIN` submits a username, email, and role to create a new user.
  Reply: system validates the username is unique, hashes the password, creates the user.
- Action: `ADMIN` deactivates a user.
  Reply: system sets `is_active = false`; that user can no longer authenticate.
- Action: a user submits username and password.
  Reply: system verifies the hash and issues a JWT (`sub`, `role`, 60-minute expiry).
- Action: a user's access token expires mid-session.
  Reply: system rejects the request with 401; the client re-authenticates (no refresh token).

**Level 1.2 — Real-Time Scoring & Decision**
Primary actors: Analyst (manual) / Client Backend System / Ingestion Process
- Action: a transaction is submitted with sender/receiver account identifiers and raw features.
  Reply: system looks up recent same-account transactions from the live graph cache and tags each
  by relationship type.
- Action: the assembled subgraph is passed to the Detection Engine.
  Reply: system returns a raw fraud probability.
- Action: the Alert & Decision Engine receives that probability.
  Reply: system computes the decision tier, risk level, relationship-based explanation (with
  per-neighbour contribution scores), and whether to raise an alert; persists the decision.

**Level 1.3 — Alerts & Review Queue**
Primary actor: Analyst
- Action: Analyst opens the Alerts & Review Queue page.
  Reply: system returns alert-worthy transactions, filterable by Needs Review / Escalated / All.
- Action: Analyst expands a row.
  Reply: system shows the full relationship explanation and per-neighbour contribution table.
- Action: Analyst acknowledges an alert (optionally as a false positive).
  Reply: system records the acknowledging identity and timestamp, broadcasts the update.

**Level 1.4 — Escalation (background)**
Primary actor: Escalation Watcher
- Action: watcher wakes on its interval.
  Reply: system scans all unacknowledged, non-escalated alerts.
- Action: an alert has exceeded the SLA window.
  Reply: system marks it escalated, logs it, and broadcasts an `alert_escalated` event.

**Level 1.5 — Live Ingestion Control**
Primary actor: Analyst / Administrator
- Action: user opens the Stream Monitor page for the first time after a server restart.
  Reply: system reports the feed as **paused** — it never auto-starts.
- Action: user presses "Resume Stream."
  Reply: system begins routing generated/consumed transactions through Flow D (§4.3) until
  explicitly paused again.

**Level 1.6 — Dashboard & Real-Time Monitoring**
Primary actor: Analyst / Administrator
- Action: user opens the dashboard; browser connects to the authenticated WebSocket.
  Reply: system pushes `new_decision`, `alert_acknowledged`, and `alert_escalated` frames as they
  occur, with no page refresh required.
- Action: user views aggregate stats.
  Reply: system returns counts and average latency.
- Action: user submits a manual score.
  Reply: system returns the decision and renders the subgraph visualization of the neighbours used.

### 4.3 Activity Diagram Flows

**Flow A — Bootstrap First Admin & User Management**
Start → request arrives → any user already exists? → **yes:** reject (bootstrap already used;
route to Level 1.1's regular "create user" instead, which requires an existing `ADMIN`) → **no:**
create the request's account as `ADMIN` → bootstrap path now closed → `ADMIN` logs in → `ADMIN`
submits new user details → username unique? → **no:** return validation error → **yes:** hash
password, create user → End.

**Flow B — Login**
Start → user submits username + password → look up user row by username → found and active? →
**no:** return 401 → **yes:** verify password hash → matches? → **no:** return 401 → **yes:** issue
JWT (sub, role, 60-min expiry) → End.

**Flow C — Score a Transaction (Detection + Decision Engine)**
Start → transaction request arrives → look up sender's recent transactions in the live cache (tag
SAME_SENDER) → look up receiver's recent transactions (tag SAME_RECEIVER) → merge any manually
supplied neighbours → build star-graph tensor → run GNN forward pass → raw probability → tier
decision (thresholds in §2.3) → neighbours exist? → **no:** explanation = "scored in isolation" →
**yes:** look up each neighbour's prior decision, compute leave-one-out contribution for up to 15
neighbours, assemble relationship summary → persist Transaction + FraudDecision rows → probability
≥ alert threshold? → **yes:** create Alert row → broadcast `new_decision` over WebSocket → register
transaction back into the live cache for both accounts → End.

**Flow D — Escalation Watch (background, loops forever)**
Start (loop) → sleep 60 seconds → query all alerts where acknowledged_at is null and escalated is
false → for each, is (now − pushed_at) ≥ SLA minutes? → **no:** skip → **yes:** mark escalated=true,
log a warning, broadcast `alert_escalated` → loop back.

**Flow E — Live Transaction Ingestion (background, loops forever; starts paused every run)**
Start (loop) → sleep interval → ingestion enabled? → **no (default at startup):** skip this tick →
**yes (only after an explicit start action):** obtain next transaction (simulator-generated today /
consumed from Kafka at target maturity) → run it through Flow C → loop back.

**Flow F — Dashboard Real-Time Update**
Start → dashboard opens authenticated WebSocket → await frames → frame type? → `new_decision`:
append to live feed, refresh stats → `alert_acknowledged`: update matching row's acknowledged
state → `alert_escalated`: flag matching row escalated, show SLA-breach badge → loop back to await
frames → connection closes → attempt reconnect after a short delay.

---

## 5. Data & Information Modeling

### 5.1 Noun listing (P = Problem space / stored data, S = Solution space / not stored)
| Noun | P/S | Notes |
|---|---|---|
| User | P | a human dashboard account for this deployment |
| Transaction | P | core entity — every scored transaction, fraud or not |
| Fraud Decision | P | one per scoring attempt — probability, tier, explanation, latency |
| Alert | P | created when a decision clears the alert threshold; tracks review + escalation state |
| Decision tier / Risk level | P | attribute values, not separate objects |
| Explanation / contribution score | P | stored as JSON on FraudDecision, not a separate table |
| Live graph cache (Redis) | S | ephemeral, 24h TTL — operational cache, not the system of record |
| JWT / Access Token | S | **stateless — never persisted; excluded from the ER diagram** |
| Model checkpoint / model version | S | **deliberately excluded — see the note at the top of this file** |
| Graph Edge | S (for this design) | **excluded from the core ER diagram — see the note at the top of this file and §7** |
| WebSocket connection | S | transient runtime concept |
| Ingestion process (simulator / Kafka consumer) | S | a process, not data |
| Escalation Watcher | S | a background process, not data |

### 5.2 Data objects & full schema (target design — four tables)

> **Design note (Alert identity):** `Alert.acknowledged_by_user_id` is a foreign key to `User`, not
> a free-text string. This directly closes a known gap from the earliest version of the system,
> where acknowledgment identity was an unverified free-text field — a real audit-trail weakness now
> that real User accounts exist to reference.
>
> **Design note (no per-client scoping):** none of these tables carry an `org_id` or equivalent —
> deliberately, because a single deployment only ever serves one client (see the deployment-model
> decision at the top of this file). Isolation between clients is a property of running separate
> deployments, not of a column in this schema.

**User**
| Attribute | Type | Notes |
|---|---|---|
| user_id | UUID | PK |
| username | VARCHAR(100) | UNIQUE, NOT NULL |
| email | VARCHAR(255) | UNIQUE, NOT NULL |
| hashed_password | VARCHAR(255) | bcrypt hash |
| role | VARCHAR(20) | `ADMIN` \| `ANALYST` |
| is_active | BOOLEAN | default TRUE |
| created_at | TIMESTAMPTZ | default NOW() |
| last_login_at | TIMESTAMPTZ | nullable |

**Transaction**
| Attribute | Type | Notes |
|---|---|---|
| transaction_id | VARCHAR(100) | PK (caller-supplied ID — safe as the primary key in a single-client deployment, no cross-client collision risk) |
| sender_account | VARCHAR(100) | NOT NULL, indexed |
| receiver_account | VARCHAR(100) | NOT NULL, indexed |
| amount | FLOAT | |
| transaction_type | VARCHAR(20) | `TRANSFER` \| `CASH_OUT` |
| step | INTEGER | PaySim timestep (training data) / minutes-since-epoch equivalent (live data) |
| old_balance_sender | FLOAT | |
| new_balance_sender | FLOAT | |
| old_balance_receiver | FLOAT | |
| new_balance_receiver | FLOAT | |
| raw_payload | JSONB | nullable |
| received_at | TIMESTAMPTZ | default NOW() |

**FraudDecision**
| Attribute | Type | Notes |
|---|---|---|
| decision_id | SERIAL | PK |
| transaction_id | VARCHAR(100) | FK → Transaction |
| fraud_probability | FLOAT | raw model output, 0–1 |
| decision | VARCHAR(10) | `ALLOW` \| `REVIEW` \| `BLOCK` |
| risk_level | VARCHAR(10) | `LOW` \| `MEDIUM` \| `HIGH` \| `CRITICAL` |
| explanation | JSONB | `{summary, neighbour_count, neighbours: [{transaction_id, edge_type, type, amount, step, contribution, prior_decision}]}` |
| neighbour_count | INTEGER | |
| inference_latency_ms | INTEGER | |
| decided_at | TIMESTAMPTZ | default NOW() |
| — | — | INDEX (decided_at DESC) — primary dashboard/history access pattern |

**Alert**
| Attribute | Type | Notes |
|---|---|---|
| alert_id | SERIAL | PK |
| decision_id | INTEGER | FK → FraudDecision |
| pushed_at | TIMESTAMPTZ | default NOW() |
| acknowledged_by_user_id | UUID | FK → User, nullable |
| acknowledged_at | TIMESTAMPTZ | nullable |
| is_false_positive | BOOLEAN | default FALSE |
| escalated | BOOLEAN | default FALSE |
| escalated_at | TIMESTAMPTZ | nullable |
| — | — | PARTIAL INDEX WHERE acknowledged_at IS NULL — the review-queue query |

### 5.3 Relationships
| Parent | Relationship | Child | Cardinality |
|---|---|---|---|
| User | acknowledges | Alert | 1 : N (optional) |
| Transaction | is scored by | FraudDecision | 1 : N (re-scoring is supported without deleting history) |
| FraudDecision | may raise | Alert | 1 : 0..1 |

### 5.4 Current-implementation delta (for an honest "Implementation Status" note in the report)
The running system today implements `Transaction`, `FraudDecision`, and `Alert` (including
`escalated`/`escalated_at`) essentially as designed. `User` does not exist yet — authentication
today is two fixed accounts (`ADMIN`/`ANALYST`) from environment variables, and
`Alert.acknowledged_by` is currently a free-text string rather than a `User` foreign key.

Two additional tables exist in the live `models.py` but are **dead** — declared (so
`Base.metadata.create_all()` creates the empty table) but never written to by any code path,
confirmed by searching the whole codebase:
- `ModelVersion` (+ `FraudDecision.model_version_id`) — no motivating requirement anywhere;
  recommended for outright removal.
- `GraphEdge` — zero rows in the live database today; not part of this SRS's four-table design;
  see §7 Roadmap for its only motivated future use.

This SRS describes the target schema the live tables are designed to grow into.

### 5.5 Model / dataset description
| Attribute | Description |
|---|---|
| Node feature vector (7 dims) | step, type (label-encoded), amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest |
| Edge types (2) | SAME_SENDER, SAME_RECEIVER — built via an O(n log n) two-pointer sliding window per account group over a 24-hour window |
| Architecture | Fused model: BatchNet (2× GCNConv, transductive, offline-only) + RealTimeNet (2× SAGEConv, inductive, the only pathway used in production) + a fusion layer; the production pathway zeroes the batch half so no full graph load is ever needed to serve a request |
| Training | Weighted BCE loss (class imbalance ~336:1), 70/15/15 split, early stopping on validation AUC-ROC |
| Data source | PaySim synthetic mobile-money simulation — TRANSFER/CASH_OUT types only (the only fraud-eligible types in PaySim) |
| Preprocessing | `LabelEncoder` on `type`, `StandardScaler` on amount + 4 balance columns; both fitted offline and reused unchanged at inference time |
| Benchmarked results (ROC-AUC) | GCN 0.797 (transductive) / 0.904 (inductive); SAGE 0.862 / 0.901; XGBoost (tabular, no graph) 0.999; **Fused GNN 0.970 (full) / 0.956 (real-time-only)** |
| Note on XGBoost's higher number | PaySim's fraud rule makes balance-drain-to-zero an almost perfect single-row giveaway; the fused GNN is the strongest model when compared fairly against other *graph-based* approaches, which is the relevant comparison for what the graph architecture itself contributes |
| Model versioning | Deliberately not tracked at the schema level (see exclusion note) — the checkpoint file itself is the only version record, loaded once at process startup |

---

## 6. Preliminary Test Plan
| Test Case | Feature | Test Type | Priority |
|---|---|---|---|
| TC-01 | Bootstrap creates exactly one ADMIN on a fresh deployment; a second bootstrap attempt is rejected | Unit + API | High |
| TC-02 | ADMIN can create/deactivate users; deactivated users can no longer log in | Unit + API | High |
| TC-03 | Login issues a valid JWT; wrong password rejected | Unit + API | High |
| TC-04 | `/score` returns a probability/decision/explanation for a known transaction | API | High |
| TC-05 | Live-cache neighbour lookup correctly tags SAME_SENDER vs SAME_RECEIVER | Unit | High |
| TC-06 | Decision tiering matches the threshold table in §2.3 at each boundary | Unit | High |
| TC-07 | Alert is created only when probability ≥ ALERT_THRESHOLD | Unit + API | High |
| TC-08 | Escalation watcher flags an alert only after the SLA window elapses | Integration | High |
| TC-09 | Acknowledging an alert records the real User identity and broadcasts the update | Integration | Medium |
| TC-10 | Live-cache outage → scoring still succeeds with zero neighbours (no crash) | Integration (fault injection) | High |
| TC-11 | Transaction feed never starts automatically — a fresh server start reports the feed as paused until an explicit start action | API + Manual | High |
| TC-12 | Full ROC-AUC benchmark reproduces the values in §5.5 on the held-out split | Offline / notebook | Medium |
| TC-13 | Dashboard/Alerts/Stream pages reconnect automatically after a dropped WebSocket | Manual / UI | Medium |

---

## 7. Roadmap (beyond this SRS's target design)
- **Kafka ingestion:** replace the in-process simulator with a real Kafka consumer — already an
  explicit, documented swap point (§3.6); requires no change to the scoring pipeline.
- **Dedicated machine credential type:** today a client's backend authenticates the same way a
  dashboard user does (JWT login via a service account). A longer-lived, revocable API-key
  mechanism — separate from human login — is a reasonable refinement once that friction matters.
- **Multi-hop / money-flow graph edges:** the offline graph currently only connects transactions
  sharing a direct sender or receiver; a longer "mule chain" edge type (A→B→C laundering paths) was
  attempted, hit a memory ceiling during offline construction, and was reverted.
- **Feature-ablation run:** a toggle exists to retrain with reduced features (removing the
  balance-drain shortcut) to more rigorously prove the graph's contribution independent of that one
  dominant tabular signal; the run hasn't been executed yet.
- **Suspicious-cluster graph visualization page:** a force-directed, multi-hop view of a whole fraud
  ring, distinct from the existing single-transaction star diagram — this is what would finally
  give the currently-dead `GraphEdge` table a reason to exist and be written to.
- **CSV report export page:** filterable export of historical decisions.
- **Deployment:** nothing is deployed to a public environment yet; run and tested locally / against
  a hosted dev database only.
