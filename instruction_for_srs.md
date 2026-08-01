# SRS Source Content — GNN Fraud Detection Platform

This document contains the **full content** needed to write the SRS/technical report (Project
Overview, Requirements Analysis, Usage Scenario, System Modeling, Data & Information Modeling,
Test Plan), in the same section structure as the reference midterm template.

**Important framing:** the system was implemented first, in a single-tenant form, before this SRS
was written. This document describes the **target design** — including multi-tenant Organization/
User management as a core part of the data model, not a footnote — because that is the correct
scope for a system meant to be sold to multiple independent clients (bKash, Nagad, Upay). Where the
target design differs from what's literally running in code today, that is called out explicitly
so the report stays honest about implementation status.

**Explicit exclusions from the ER diagram / schema:**
- `Token` / `JWT` — a JWT is stateless, signed and verified, never persisted server-side.
- `ModelVersion` — defined in the current codebase's ORM models but never actually written to by
  any code path (`FraudDecision.model_version_id` stays null everywhere); model-version lineage
  tracking isn't a stated objective of this system, so it is deliberately left out rather than
  modeled speculatively. If reproducible model versioning becomes a real requirement later, it can
  be added as a small lookup table then.

---

## 1. Project Overview

### 1.1 Project Title
Graph Neural Network Fraud Detection Platform — a multi-tenant, real-time transaction fraud
detection and review system for mobile financial service (MFS) providers.

### 1.2 Problem Statement
Mobile financial services (bKash, Nagad, Upay-style mobile money operators) process high-volume
peer-to-peer transfers and cash-outs where fraud rings launder money through short-lived "mule"
accounts — a victim's account is drained to a receiver who is itself part of a chain of transfers,
often completing within minutes. Row-by-row tabular fraud models miss this because they score each
transaction in isolation; the fraud signal is in the *relationships* between transactions (shared
accounts, tight time windows), not any single transaction's own features. Conventional monitoring
also cannot distinguish "this looks unusual" from "this is structurally connected to other flagged
activity" — it can flag an outlier but not explain which relationships made it suspicious.
Additionally, because competing MFS providers would be simultaneous clients of the same platform,
each client's transaction data must be provably isolated from every other client's.

### 1.3 Objectives
- **O1 — Graph-based fraud representation:** model transactions as nodes connected by shared
  account relationships (not isolated rows), so structural fraud patterns are visible to the model.
- **O2 — Real-time scoring:** score an incoming transaction and return a decision within
  milliseconds, using only the inductive (no full-graph-reload) pathway of the trained model.
- **O3 — Explainable decisions:** every flagged transaction must come with a concrete explanation
  of *which* relationships drove the score and how much each contributed — not just a probability.
- **O4 — Human-in-the-loop review with escalation:** flagged transactions enter a review queue; if
  no analyst acts within an SLA window, the system escalates automatically.
- **O5 — Multi-tenant isolation:** support multiple independent client organizations on one shared
  platform, sharing inference infrastructure but never data, with database-enforced isolation.
- **O6 — Flexible ingestion:** accept transactions from a dashboard-triggered manual score, a
  client's own backend calling the scoring API directly (via a per-org API key), or — at target
  maturity — a Kafka topic, all through the identical scoring/decision code path.
- **O7 — Live operational visibility:** a real-time dashboard, alert review queue, and live stream
  monitor so analysts can watch the system work without manually triggering every check.

### 1.4 Scope

**In scope:**
- Multi-tenant SaaS platform: organization provisioning, per-organization users and roles, API Key
  management for machine-to-machine ingestion, database-enforced multi-tenant data isolation.
- A trained fused Graph Neural Network (GCN + GraphSAGE) served for real-time inference, shared
  across all tenants (one model, isolated data — not one model per tenant).
- An offline data pipeline (PaySim-based) that builds the transaction graph and trains the model.
- Redis-backed short-term "live graph memory" for real-time neighbour lookup, namespaced per org.
- PostgreSQL persistence of every account, transaction, decision (full audit trail), and alert.
- A pluggable transaction ingestion path: an in-process simulator today (defaults to paused —
  started explicitly from the dashboard), a real Kafka consumer at target maturity, both feeding
  one shared scoring pipeline.
- A dashboard: auth, stats + manual scoring + live feed, a dedicated alerts/review-queue page, and
  a live stream monitor page.

**Out of scope:**
- Non-financial anomaly domains (this is transaction fraud only).
- Automated fund freezing/reversal at the banking core — this system recommends
  ALLOW/REVIEW/BLOCK and raises alerts; it does not itself move money.
- Billing/subscription-plan management for tenant organizations.
- Chat-based / narrative root-cause summarization.
- Per-tenant custom-trained models (deliberate design choice — see O5/O6 and §5.5).
- Model-version lineage tracking (see exclusion note above).

### 1.5 Deliverables
| # | Deliverable | Description |
|---|---|---|
| 1 | Multi-Tenant Backend | FastAPI: org/user auth, API Keys, real-time scoring, decision/alert engine, escalation watcher, REST + WebSocket API, org-scoped data access throughout |
| 2 | Trained Fused GNN | `FraudGNN_main.pt` + fitted `LabelEncoder`/`StandardScaler`, a platform-shared artifact loaded once at startup |
| 3 | Offline Data & Graph Pipeline | PaySim ingestion, graph construction, model training/benchmarking scripts |
| 4 | Transaction Ingestion Layer | In-process simulator (current, off by default) / Kafka consumer (target), both calling one shared scoring pipeline |
| 5 | Dashboard | Next.js SPA: auth, stats/scoring/live-feed page, alerts & review-queue page, live stream monitor page |
| 6 | Documentation | Architecture reference, this SRS, patterns/QA reference |

---

## 2. Requirements Analysis

### 2.1 Functional Requirements
| ID | Functional Requirement |
|---|---|
| FR-1 | The system shall allow a platform administrator to provision a new client Organization together with its initial `ORG_ADMIN` user. |
| FR-2 | The system shall allow an `ORG_ADMIN` to create, deactivate, and list `ANALYST` (or additional `ORG_ADMIN`) accounts within their own Organization only. |
| FR-3 | The system shall authenticate a User by username and password, scoped to their Organization, and issue a signed JWT carrying the user's role and organization identifier. |
| FR-4 | The system shall allow an `ORG_ADMIN` to generate, list (by prefix only), and revoke API Keys used to authenticate machine-to-machine transaction submission on behalf of their Organization. |
| FR-5 | The system shall accept a transaction for scoring via an authenticated request (dashboard JWT or Organization API Key), containing its raw feature values and sender/receiver account identifiers. |
| FR-6 | The system shall look up each account's recent transaction history from a live graph cache, tagging each result with the relationship (same-sender / same-receiver) that produced it, scoped to the submitting Organization only. |
| FR-7 | The system shall run real-time GNN inference over the transaction and its retrieved neighbours to produce a fraud probability between 0 and 1. |
| FR-8 | The system shall convert the fraud probability into a decision (`ALLOW`/`REVIEW`/`BLOCK`) and a risk level (`LOW`/`MEDIUM`/`HIGH`/`CRITICAL`) using configurable thresholds. |
| FR-9 | The system shall generate a relationship-based explanation for every scored transaction, including, per neighbour: which relationship connected it, whether it was previously flagged, and a leave-one-out contribution score. |
| FR-10 | The system shall persist every scored transaction and its decision permanently, scoped to the originating Organization, regardless of decision tier. |
| FR-11 | The system shall raise an alert (add the transaction to the review queue) whenever its fraud probability clears a configurable alert threshold, independent of the decision-tier thresholds. |
| FR-12 | The system shall let an Analyst view the review queue for their own Organization, filterable by Needs Review / Escalated / All. |
| FR-13 | The system shall let an Analyst acknowledge an alert, optionally marking it a false positive; the acknowledging identity shall be the authenticated User, not free text. |
| FR-14 | The system shall automatically mark an alert "escalated" and broadcast a distinct notification if it remains unacknowledged past a configurable SLA window. |
| FR-15 | The system shall allow an Organization's transaction feed (simulator today, Kafka consumer at target maturity) to be started, paused, and resumed independently per Organization — defaulting to paused on startup, never auto-starting — with no change to scoring/decision logic. |
| FR-16 | The system shall push real-time scoring, acknowledgment, and escalation events to connected dashboards over an authenticated, Organization-scoped WebSocket channel. |
| FR-17 | The dashboard shall display aggregate statistics (total scored, per-tier counts, average latency) scoped to the logged-in user's Organization. |
| FR-18 | The dashboard shall let a user manually submit a transaction for scoring and view a subgraph visualization of exactly the neighbours the model used. |
| FR-19 | The dashboard shall provide a live stream monitor view showing connection status, throughput, and a real-time transaction ticker, with an explicit start/stop control. |

### 2.2 Non-Functional Requirements
| Category | Requirement |
|---|---|
| Performance | Real-time scoring — including neighbour lookup, GNN inference, and explanation/contribution computation — must complete well within a sub-second budget; the dashboard must reflect a new event within a few seconds of it occurring. |
| Security | Passwords and API Key secrets must never be stored in plaintext (bcrypt hashing only); dashboard traffic is authenticated via JWT Bearer tokens; machine ingestion traffic is authenticated via a per-Organization API Key; the WebSocket handshake is authenticated via a token query parameter (browsers cannot set custom headers on a WebSocket handshake). |
| Privacy / Data Isolation | Every tenant-scoped table carries an `org_id` column, denormalized even where technically derivable through a foreign-key chain, specifically so PostgreSQL Row-Level Security policies can enforce — at the database layer, not just in application code — that no Organization can read another Organization's accounts, transactions, decisions, or alerts. |
| Scalability | The GNN inference path is shared platform infrastructure serving all tenants (no per-tenant retraining); every tenant-scoped table's natural query and index pattern leads with `org_id`, so the schema partitions cleanly if sharded later. |
| Availability | A live-graph-cache outage degrades scoring gracefully (fewer neighbours found, not a failed request); the ingestion loop and the escalation watcher run as independent background processes that cannot block request handling. |
| Predictability | Background processes (ingestion, escalation) never take an action a user didn't explicitly request without a clear, visible control — e.g. the transaction feed never starts itself; it always requires an explicit start action, defaulting to off. |
| Usability | The review queue defaults to showing only actionable (REVIEW/BLOCK, unacknowledged) items; an escalated alert is visually distinguished without requiring a manual refresh. |
| Maintainability | The Detection Engine (pure GNN inference) and the Alert & Decision Engine (tiering, explanation, alert policy) are separate modules connected by a single call boundary, so either can be modified — a different model, a different alerting policy — without touching the other. |
| Auditability | Every scored transaction, not only flagged ones, produces a permanent decision record; every alert acknowledgment is tied to a real, authenticated User identity. |

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
| Organization Administrator (`ORG_ADMIN` role) | Manages their own Organization's users and API Keys; also has Analyst capabilities |
| Platform Administrator | Provisions new client Organizations; operates the shared platform and model |
| MFS Client Organization (bKash / Nagad / Upay) | The paying customer; their staff are the Analysts/Org Admins above, strictly scoped to their own data |
| Platform Engineering Team | Builds and operates the backend, the model pipeline, and the dashboard |
| End customers of the MFS provider | Indirect beneficiaries — protected from fraud without needing to know this system exists |
| Course Supervisor / Evaluator | Assesses the project against course deliverable criteria |

---

## 3. Usage Scenario

### 3.1 Organization & User Onboarding
A Platform Administrator provisions a new client Organization (e.g. "bKash") together with its
first `ORG_ADMIN` user, handing over a temporary password. This is deliberately not open
self-service registration — the clients are regulated financial institutions, and onboarding is
part of a sales/security-review process, not a public sign-up form. Once logged in, that
`ORG_ADMIN` creates `ANALYST` accounts for their own staff directly from the dashboard; they can
never see or affect another Organization's users.

### 3.2 Authentication
A User logs in with a username and password. The system verifies the bcrypt hash against their
Organization-scoped account and issues a signed JWT carrying their user identity, role, and
`org_id` (60-minute expiry). Every subsequent request presents this token as a Bearer header (or,
for the WebSocket connection, as a query parameter); every downstream query is automatically
scoped to the `org_id` embedded in that token.

### 3.3 API Key Management
Before a client's own backend system can submit transactions programmatically (rather than through
the dashboard), an `ORG_ADMIN` generates an API Key from Settings. The system creates a
cryptographically random secret, stores only its bcrypt hash, and displays the plaintext value
exactly once. Existing keys are listed by prefix and label only, and can be revoked at any time,
immediately invalidating any client system still using them.

### 3.4 Real-Time Transaction Scoring (Detection Engine)
A transaction arrives — via the dashboard's manual scoring form, an Organization's own backend
calling the scoring API with its API Key, or the ingestion feed (simulator today, Kafka at target
maturity) — with its raw feature values and account identifiers. The system looks up the sender's
and receiver's recent transaction history from the live graph cache (scoped to the submitting
Organization), tags each result with the relationship that surfaced it, merges in any explicitly
supplied neighbours, and assembles a small star-shaped subgraph. This subgraph is fed through the
trained model's real-time (inductive) pathway to produce a single fraud probability. This step —
and only this step — is the Detection Engine; it has no concept of thresholds, alerts, or
explanations, and no concept of which Organization it's running for (the org-scoping happens
around it, not inside it).

### 3.5 Decision, Explanation & Alerting (Alert & Decision Engine)
Given the raw probability, a separate module converts it into a decision tier and risk level, and
independently decides whether the probability clears the alert threshold. If neighbours exist, the
system builds a relationship-based explanation: for each neighbour, which edge type connected it,
whether it was itself previously flagged, and a leave-one-out contribution score. Every scored
transaction — regardless of tier — is permanently persisted with its full explanation, scoped to
the Organization that submitted it. If the alert threshold is cleared, a review-queue entry is
created and broadcast live to every connected dashboard within that Organization only.

### 3.6 Review & Escalation
An Analyst working the Alerts & Review Queue page sees every alert-worthy transaction for their own
Organization, with its relationship explanation expandable per row, and can acknowledge it
(optionally marking it a false positive) — recorded against their own User identity. A background
watcher runs every 60 seconds across all Organizations; any alert that has sat unacknowledged past
the SLA window is marked escalated, logged server-side, and broadcast as a distinct real-time event
to that Organization's connected dashboards.

### 3.7 Live Transaction Ingestion
A background feed can generate realistic transactions (mostly normal payments, some suspicious
drain patterns, occasional ring-shaped fraud patterns) and route each one through the shared
scoring path described above — but it never runs unless explicitly started: the feed defaults to
paused on every server start, and a user must press "Resume Stream" on the dashboard before any
transaction flows. This is an explicitly documented stand-in for a real Kafka consumer per
Organization — swapping it in changes only "where a transaction comes from," not the scoring,
decision, or alerting logic, since that logic already only depends on a well-formed transaction
request, never its source.

### 3.8 Dashboard & Real-Time Monitoring
The dashboard opens an Organization-scoped, JWT-authenticated WebSocket connection and shows:
aggregate stats, a manual scoring form with a live subgraph diagram of exactly what the model saw,
and a live feed table of every transaction scored for that Organization. The separate Alerts page
and Stream Monitor page open their own WebSocket connections and render the same event stream
filtered/formatted for their specific purpose.

---

## 4. System Modeling

### 4.1 Actors
- **Platform Administrator** — provisions Organizations; operates the shared platform.
- **Organization Administrator (`ORG_ADMIN`)** — manages their own Organization's users and API
  Keys; has all Analyst capabilities.
- **Analyst (`ANALYST`)** — scores transactions manually, reviews and acknowledges alerts, watches
  the live feed and stream monitor.
- **Client Backend System** — an Organization's own service authenticating with an API Key to
  submit transactions programmatically (secondary/system actor).
- **Ingestion Process** — the background simulator today / Kafka consumer at target maturity;
  generates or relays transactions into the scoring pipeline (secondary/system actor).
- **Escalation Watcher** — background process that periodically escalates overdue alerts.
- **Detection Engine (GNN inference)** — secondary/system actor: produces a raw probability.
- **Live Graph Cache (Redis)** — secondary actor: supplies/receives recent-transaction lookups.

### 4.2 Use case levels

**Level 0 — System overview:** Manage Organizations & Users, Manage API Keys, Score a Transaction,
Manage Alerts & Review Queue, Monitor Live Stream, (background) Ingest & Detect, (background)
Escalate.

**Level 1.1 — Organization, User & Authentication**
Primary actors: Platform Administrator, Organization Administrator, Analyst
- Action: Platform Administrator provisions an Organization with a name and initial `ORG_ADMIN`.
  Reply: system creates the Organization and User rows; the admin's temporary password is issued.
- Action: `ORG_ADMIN` submits a username, email, and role to create a new user.
  Reply: system validates uniqueness within the Organization, hashes the password, creates the
  User row scoped to that `org_id`.
- Action: `ORG_ADMIN` deactivates a user.
  Reply: system sets `is_active = false`; every org-scoped query additionally checks `is_active`.
- Action: a User submits username and password.
  Reply: system verifies the hash against that Organization's User record and issues a JWT
  (`sub`, `org_id`, `role`, 60-minute expiry).
- Action: a User's access token expires mid-session.
  Reply: system rejects the request with 401; the client re-authenticates (no refresh token in the
  current design).

**Level 1.2 — API Key Management**
Primary actor: Organization Administrator
- Action: `ORG_ADMIN` requests a new API Key.
  Reply: system generates a random secret, stores only its bcrypt hash plus a display prefix, and
  returns the plaintext value exactly once.
- Action: `ORG_ADMIN` lists their Organization's keys.
  Reply: system returns each key's prefix, label, active status, and last-used timestamp — never
  the plaintext value.
- Action: `ORG_ADMIN` revokes a key.
  Reply: system marks it inactive; any client system still using it is rejected on its next call.

**Level 1.3 — Real-Time Scoring & Decision**
Primary actors: Analyst (manual) / Client Backend System (API Key) / Ingestion Process
- Action: a transaction is submitted with sender/receiver account identifiers and raw features.
  Reply: system resolves both accounts (creating new Account rows on first sight), looks up recent
  same-account transactions from the live graph cache within the submitting Organization, and tags
  each by relationship type.
- Action: the assembled subgraph is passed to the Detection Engine.
  Reply: system returns a raw fraud probability.
- Action: the Alert & Decision Engine receives that probability.
  Reply: system computes the decision tier, risk level, relationship-based explanation (with
  per-neighbour contribution scores), and whether to raise an alert; persists the decision.

**Level 1.4 — Alerts & Review Queue**
Primary actor: Analyst
- Action: Analyst opens the Alerts & Review Queue page.
  Reply: system returns alert-worthy transactions for the Analyst's own Organization, filterable
  by Needs Review / Escalated / All.
- Action: Analyst expands a row.
  Reply: system shows the full relationship explanation and per-neighbour contribution table.
- Action: Analyst acknowledges an alert (optionally as a false positive).
  Reply: system records the acknowledging User identity and timestamp, broadcasts the update.

**Level 1.5 — Escalation (background)**
Primary actor: Escalation Watcher
- Action: watcher wakes on its interval.
  Reply: system scans all Organizations' unacknowledged, non-escalated alerts.
- Action: an alert has exceeded the SLA window.
  Reply: system marks it escalated, logs it, and broadcasts an `alert_escalated` event scoped to
  that alert's Organization.

**Level 1.6 — Live Ingestion Control**
Primary actor: Analyst / Organization Administrator
- Action: user opens the Stream Monitor page for the first time after a server restart.
  Reply: system reports the feed as **paused** — it never auto-starts.
- Action: user presses "Resume Stream."
  Reply: system begins routing generated/consumed transactions through Flow D (§4.3) for that
  Organization until explicitly paused again.

**Level 1.7 — Dashboard & Real-Time Monitoring**
Primary actor: Analyst / Organization Administrator
- Action: user opens the dashboard; browser connects to the Organization-scoped WebSocket.
  Reply: system pushes `new_decision`, `alert_acknowledged`, and `alert_escalated` frames as they
  occur for that Organization, with no page refresh required.
- Action: user views aggregate stats.
  Reply: system returns counts and average latency scoped to the user's Organization.
- Action: user submits a manual score.
  Reply: system returns the decision and renders the subgraph visualization of the neighbours used.

### 4.3 Activity Diagram Flows

**Flow A — Organization & User Provisioning**
Start → Platform Administrator submits org name → system creates Organization row → system creates
initial `ORG_ADMIN` User row scoped to that org → temporary credentials issued → `ORG_ADMIN` logs
in → `ORG_ADMIN` submits new user details → username unique within org? → **no:** return
validation error → **yes:** hash password, create User row scoped to the same `org_id` → End.

**Flow B — API Key Generation**
Start → `ORG_ADMIN` requests a new key → generate random secret → bcrypt-hash it → store hash +
prefix + `org_id` → display plaintext once → End. *(Revocation: Start → select key → set
is_active=false → End.)*

**Flow C — Login**
Start → user submits username + password → look up User row by (org_id-scoped) username → found
and active? → **no:** return 401 → **yes:** verify password hash → matches? → **no:** return 401 →
**yes:** issue JWT (sub, org_id, role, 60-min expiry) → End.

**Flow D — Score a Transaction (Detection + Decision Engine)**
Start → transaction request arrives (with caller's `org_id` resolved from JWT or API Key) → resolve
sender Account within org (create if new) → resolve receiver Account within org (create if new) →
look up sender's recent transactions in the live cache (tag SAME_SENDER) → look up receiver's
recent transactions (tag SAME_RECEIVER) → merge any manually supplied neighbours → build star-graph
tensor → run GNN forward pass → raw probability → tier decision (thresholds in §2.3) → neighbours
exist? → **no:** explanation = "scored in isolation" → **yes:** look up each neighbour's prior
decision, compute leave-one-out contribution for up to 15 neighbours, assemble relationship
summary → persist Transaction + FraudDecision rows (org-scoped) → probability ≥ alert threshold? →
**yes:** create Alert row (org-scoped) → broadcast `new_decision` over the org's WebSocket channel
→ register transaction back into the live cache for both accounts → End.

**Flow E — Escalation Watch (background, loops forever)**
Start (loop) → sleep 60 seconds → query all Organizations' alerts where acknowledged_at is null and
escalated is false → for each, is (now − pushed_at) ≥ SLA minutes? → **no:** skip → **yes:** mark
escalated=true, log a warning, broadcast `alert_escalated` scoped to that alert's org → loop back.

**Flow F — Live Transaction Ingestion (background, loops forever; starts paused every run)**
Start (loop) → sleep interval → ingestion enabled for this Organization? → **no (default at
startup):** skip this tick → **yes (only after an explicit start action):** obtain next transaction
(simulator-generated today / consumed from Kafka at target maturity) → run it through Flow D →
loop back.

**Flow G — Dashboard Real-Time Update**
Start → dashboard opens authenticated, org-scoped WebSocket → await frames → frame type? →
`new_decision`: append to live feed, refresh stats → `alert_acknowledged`: update matching row's
acknowledged state → `alert_escalated`: flag matching row escalated, show SLA-breach badge → loop
back to await frames → connection closes → attempt reconnect after a short delay.

---

## 5. Data & Information Modeling

### 5.1 Noun listing (P = Problem space / stored data, S = Solution space / not stored)
| Noun | P/S | Notes |
|---|---|---|
| Organization | P | one row per client (bKash / Nagad / Upay); the tenancy boundary |
| User | P | a human dashboard account, scoped to one Organization |
| API Key | P | a machine credential, scoped to one Organization |
| Account | P | a sender/receiver identity as known to one Organization — promoted from a bare string to a first-class entity so it can be org-scoped and FK-referenced |
| Transaction | P | core entity — every scored transaction, fraud or not |
| Fraud Decision | P | one per scoring attempt — probability, tier, explanation, latency |
| Alert | P | created when a decision clears the alert threshold; tracks review + escalation state |
| Graph Edge | P | SAME_SENDER / SAME_RECEIVER relationship between two transactions — **defined in the schema but currently written by no code path at all** (not live scoring, not the offline pipeline); retained in the target design specifically to back the future multi-hop cluster-graph feature (§7), since Redis's 24h TTL means no permanent edge history exists anywhere today |
| Decision tier / Risk level | P | attribute values, not separate objects |
| Explanation / contribution score | P | stored as JSON on FraudDecision, not a separate table |
| Live graph cache (Redis) | S | ephemeral, namespaced per org, 24h TTL — operational cache, not the system of record |
| JWT / Access Token | S | **stateless — never persisted; excluded from the ER diagram** |
| Model checkpoint / model version | S | **deliberately excluded — see the note at the top of this file** |
| WebSocket connection | S | transient runtime concept |
| Ingestion process (simulator / Kafka consumer) | S | a process, not data |
| Escalation Watcher | S | a background process, not data |

### 5.2 Data objects & full schema (target design)

> **Design note (senior-level rationale):** `org_id` is denormalized directly onto every
> tenant-scoped table below — `Account`, `Transaction`, `FraudDecision`, `Alert`, `GraphEdge` — even
> though it is technically derivable through foreign-key chains from `Account`/`Transaction`. This
> is the standard multi-tenant SaaS pattern: it lets a PostgreSQL Row-Level Security policy be
> written once per table as `USING (org_id = current_org())`, with no joins required, so isolation
> is enforced at the database layer even if an application-layer bug forgets a filter.
>
> **Design note (Account promotion):** the currently-running implementation stores
> `sender_account`/`receiver_account` as bare strings directly on `Transaction`. This SRS promotes
> `Account` to a first-class, org-scoped entity, referenced by foreign key. This is a normalization
> the multi-tenant design requires regardless (two Organizations' account-ID spaces must never be
> confused), and it also gives account-level aggregates (e.g. a cached risk score) a proper home
> instead of requiring a full transaction scan on every read.
>
> **Design note (Alert identity):** `Alert.acknowledged_by_user_id` is a foreign key to `User`, not
> a free-text string. This directly closes a known gap from the single-tenant version of the system,
> where acknowledgment identity was an unverified free-text field — a real audit-trail weakness now
> that real User accounts exist to reference.
>
> **Design note (no ModelVersion table):** intentionally omitted — see the exclusion note at the
> top of this file. Nothing in the current or target functional requirements needs per-decision
> model-version lineage; adding the table anyway would be unmotivated schema bloat.

**Organization**
| Attribute | Type | Notes |
|---|---|---|
| org_id | UUID | PK, default gen_random_uuid() |
| name | VARCHAR(255) | NOT NULL |
| slug | VARCHAR(100) | UNIQUE, NOT NULL |
| is_active | BOOLEAN | default TRUE |
| created_at | TIMESTAMPTZ | default NOW() |

**User**
| Attribute | Type | Notes |
|---|---|---|
| user_id | UUID | PK |
| org_id | UUID | FK → Organization, ON DELETE CASCADE |
| username | VARCHAR(100) | NOT NULL |
| email | VARCHAR(255) | UNIQUE, NOT NULL |
| hashed_password | VARCHAR(255) | bcrypt hash |
| role | VARCHAR(20) | `ORG_ADMIN` \| `ANALYST` |
| is_active | BOOLEAN | default TRUE |
| created_at | TIMESTAMPTZ | default NOW() |
| last_login_at | TIMESTAMPTZ | nullable |
| — | — | UNIQUE (org_id, username) |

**ApiKey**
| Attribute | Type | Notes |
|---|---|---|
| api_key_id | UUID | PK |
| org_id | UUID | FK → Organization, ON DELETE CASCADE |
| key_hash | VARCHAR(255) | UNIQUE, bcrypt hash of the secret |
| key_prefix | VARCHAR(20) | for display only |
| name | VARCHAR(100) | nullable, user-defined label |
| is_active | BOOLEAN | default TRUE |
| created_by_user_id | UUID | FK → User, nullable (audit) |
| created_at | TIMESTAMPTZ | default NOW() |
| last_used_at | TIMESTAMPTZ | nullable |

**Account**
| Attribute | Type | Notes |
|---|---|---|
| account_id | UUID | PK (internal surrogate key) |
| org_id | UUID | FK → Organization, ON DELETE CASCADE |
| external_account_id | VARCHAR(100) | the MFS's own customer account identifier |
| first_seen_at | TIMESTAMPTZ | default NOW() |
| last_seen_at | TIMESTAMPTZ | default NOW() |
| cached_risk_score | FLOAT | nullable — materialized/refreshed async, not a source of truth |
| — | — | UNIQUE (org_id, external_account_id) |

**Transaction**
| Attribute | Type | Notes |
|---|---|---|
| transaction_id | UUID | PK (internal surrogate — decouples storage from caller-supplied IDs, which could collide across Organizations) |
| org_id | UUID | FK → Organization |
| external_transaction_id | VARCHAR(100) | caller-supplied ID |
| sender_account_id | UUID | FK → Account |
| receiver_account_id | UUID | FK → Account |
| amount | FLOAT | |
| transaction_type | VARCHAR(20) | `TRANSFER` \| `CASH_OUT` |
| step | INTEGER | PaySim timestep (training data) / minutes-since-epoch equivalent (live data) |
| old_balance_sender | FLOAT | |
| new_balance_sender | FLOAT | |
| old_balance_receiver | FLOAT | |
| new_balance_receiver | FLOAT | |
| raw_payload | JSONB | nullable |
| received_at | TIMESTAMPTZ | default NOW() |
| — | — | UNIQUE (org_id, external_transaction_id) |

**FraudDecision**
| Attribute | Type | Notes |
|---|---|---|
| decision_id | UUID | PK |
| org_id | UUID | FK → Organization (denormalized) |
| transaction_id | UUID | FK → Transaction |
| fraud_probability | FLOAT | raw model output, 0–1 |
| decision | VARCHAR(10) | `ALLOW` \| `REVIEW` \| `BLOCK` |
| risk_level | VARCHAR(10) | `LOW` \| `MEDIUM` \| `HIGH` \| `CRITICAL` |
| explanation | JSONB | `{summary, neighbour_count, neighbours: [{transaction_id, edge_type, type, amount, step, contribution, prior_decision}]}` |
| neighbour_count | INTEGER | |
| inference_latency_ms | INTEGER | |
| decided_at | TIMESTAMPTZ | default NOW() |
| — | — | INDEX (org_id, decided_at DESC) — primary dashboard/history access pattern |

**Alert**
| Attribute | Type | Notes |
|---|---|---|
| alert_id | UUID | PK |
| org_id | UUID | FK → Organization (denormalized — the review-queue hot path) |
| decision_id | UUID | FK → FraudDecision |
| pushed_at | TIMESTAMPTZ | default NOW() |
| acknowledged_by_user_id | UUID | FK → User, nullable |
| acknowledged_at | TIMESTAMPTZ | nullable |
| is_false_positive | BOOLEAN | default FALSE |
| escalated | BOOLEAN | default FALSE |
| escalated_at | TIMESTAMPTZ | nullable |
| — | — | PARTIAL INDEX (org_id) WHERE acknowledged_at IS NULL — the review-queue query |

**GraphEdge** — *currently zero rows in the live database; no code path writes to it today (verified by search — not `scoring_pipeline.py`, not the offline pipeline). Kept in the target schema only because the planned cluster-graph feature (§7) needs a persisted edge history Redis's 24h TTL can't provide — not because anything live populates it now.*
| Attribute | Type | Notes |
|---|---|---|
| edge_id | UUID | PK |
| org_id | UUID | FK → Organization (denormalized) |
| source_transaction_id | UUID | FK → Transaction |
| target_transaction_id | UUID | FK → Transaction |
| edge_type | VARCHAR(20) | `SAME_SENDER` \| `SAME_RECEIVER` |
| created_at | TIMESTAMPTZ | default NOW() |

### 5.3 Relationships
| Parent | Relationship | Child | Cardinality |
|---|---|---|---|
| Organization | owns | User | 1 : N |
| Organization | owns | ApiKey | 1 : N |
| Organization | owns | Account | 1 : N |
| Organization | owns | Transaction | 1 : N |
| Organization | owns | FraudDecision | 1 : N (denormalized scope) |
| Organization | owns | Alert | 1 : N (denormalized scope) |
| Organization | owns | GraphEdge | 1 : N (denormalized scope) |
| User | creates | ApiKey | 1 : N (optional, audit) |
| User | acknowledges | Alert | 1 : N (optional) |
| Account | sends | Transaction | 1 : N |
| Account | receives | Transaction | 1 : N |
| Transaction | is scored by | FraudDecision | 1 : N (re-scoring is supported without deleting history) |
| FraudDecision | may raise | Alert | 1 : 0..1 |
| Transaction | connects via | GraphEdge (source) | 1 : N |
| Transaction | connects via | GraphEdge (target) | 1 : N |

### 5.4 Current-implementation delta (for an honest "Implementation Status" note in the report)
The running system today implements a **single-tenant subset** of the schema above:
`Transaction`, `FraudDecision`, and `Alert` (including `escalated`/`escalated_at`) exist
essentially as designed, minus `org_id` and minus the `Account` promotion (sender/receiver are
still bare strings on `Transaction`). `Organization`, `User`, and `ApiKey` do not exist yet —
authentication today is two fixed accounts (`ADMIN`/`ANALYST`) from environment variables, and
`Alert.acknowledged_by` is currently a free-text string rather than a `User` foreign key.

Two tables in the live `models.py` are **dead** — declared (so `Base.metadata.create_all()`
creates the empty table) but never written to by any code path, confirmed by searching the whole
codebase:
- `ModelVersion` (+ `FraudDecision.model_version_id`) — no motivating requirement anywhere;
  recommended for outright removal, not retention. See the exclusion note at the top of this file.
- `GraphEdge` — zero rows in the live Neon database today; kept in the target schema only because
  the future cluster-graph feature (§7) needs it, not because anything currently populates it.

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
| Multi-tenancy & the model | One model checkpoint is shared across every Organization — training a separate model per client is not planned; isolation applies to data, not inference infrastructure |
| Model versioning | Deliberately not tracked at the schema level (see exclusion note) — the checkpoint file itself is the only version record, loaded once at process startup |

---

## 6. Preliminary Test Plan
| Test Case | Feature | Test Type | Priority |
|---|---|---|---|
| TC-01 | Organization provisioning creates the org and its initial ORG_ADMIN correctly | Unit + API | High |
| TC-02 | ORG_ADMIN can create/deactivate users only within their own Organization | Unit + API | High |
| TC-03 | Login issues a JWT scoped to the correct org_id; wrong password rejected | Unit + API | High |
| TC-04 | API Key generation, listing (prefix only), and revocation | Unit + API | High |
| TC-05 | `/score` returns a probability/decision/explanation for a known transaction | API | High |
| TC-06 | Live-cache neighbour lookup correctly tags SAME_SENDER vs SAME_RECEIVER | Unit | High |
| TC-07 | Decision tiering matches the threshold table in §2.3 at each boundary | Unit | High |
| TC-08 | Alert is created only when probability ≥ ALERT_THRESHOLD | Unit + API | High |
| TC-09 | Escalation watcher flags an alert only after the SLA window elapses | Integration | High |
| TC-10 | Acknowledging an alert records the real User identity and broadcasts the update | Integration | Medium |
| TC-11 | Live-cache outage → scoring still succeeds with zero neighbours (no crash) | Integration (fault injection) | High |
| TC-12 | **Transaction feed never starts automatically** — a fresh server start reports the feed as paused for every Organization until an explicit start action | API + Manual | High |
| TC-13 | **Cross-organization data isolation** — Organization A cannot read Organization B's accounts, transactions, decisions, or alerts through any endpoint | Security | Critical |
| TC-14 | Row-Level Security policies reject a query missing/mismatching the session's org context, even if application-layer filtering is bypassed | Security | Critical |
| TC-15 | Full ROC-AUC benchmark reproduces the values in §5.5 on the held-out split | Offline / notebook | Medium |
| TC-16 | Dashboard/Alerts/Stream pages reconnect automatically after a dropped WebSocket | Manual / UI | Medium |

---

## 7. Roadmap (beyond this SRS's target design)
- **Kafka ingestion:** replace the in-process simulator with a real per-Organization Kafka consumer
  — already an explicit, documented swap point (§3.7); requires no change to the scoring pipeline.
- **Multi-hop / money-flow graph edges:** the offline graph currently only connects transactions
  sharing a direct sender or receiver; a longer "mule chain" edge type (A→B→C laundering paths) was
  attempted, hit a memory ceiling during offline construction, and was reverted.
- **Feature-ablation run:** a toggle exists to retrain with reduced features (removing the
  balance-drain shortcut) to more rigorously prove the graph's contribution independent of that one
  dominant tabular signal; the run hasn't been executed yet.
- **Suspicious-cluster graph visualization page:** a force-directed, multi-hop view of a whole fraud
  ring, distinct from the existing single-transaction star diagram.
- **CSV report export page:** filterable export of historical decisions.
- **Deployment:** nothing is deployed to a public environment yet; run and tested locally / against
  hosted dev databases only.
