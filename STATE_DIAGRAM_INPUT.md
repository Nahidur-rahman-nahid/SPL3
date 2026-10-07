# Input for Behavioral Modeling — State Transition Diagrams (Event-Centric)

Paste this whole file into Claude (claude.ai) together with the prompt in §0. Every state, event
and rule below is taken from the actual running code, so the diagrams match the system.

---

## 0. Prompt to paste

> I am writing the SRS / technical report for my final-year software project (SPL3). Using ONLY
> the project information below, write the **Behavioral Modeling** section, event-centric:
>
> 1. **Identifying events** — a short paragraph explaining that events were identified from the
>    usage scenarios, then the event table (Event, Initiator, Object affected, Resulting state
>    change), using §3.
> 2. **State transition diagrams** — one diagram per object in §4. For each: a one-paragraph
>    description, a state table (State, Meaning), a transition table (From, Event [guard] /
>    action, To), and **Mermaid `stateDiagram-v2` code** I can render into an image. Label every
>    transition as `event [guard] / action`. Mark the initial and final states.
>
> Use formal report language. Keep state and event names as given. Do not invent states or
> events that are not listed.

---

## 1. Project in short

**Graph Neural Network–based Real-Time Fraud Detection for Mobile Financial Services**
(bKash / Nagad / Upay style). Each transaction (TRANSFER or CASH_OUT) is scored in real time:
related recent transactions (same sender / same receiver, last 24 h) are fetched from a Redis
cache, a GNN outputs a fraud probability, a decision engine turns it into ALLOW / REVIEW /
BLOCK, Bangladesh Bank rules are applied, the result is stored, alerts are pushed to analysts
over WebSocket, and unacknowledged alerts are escalated.

**Actors / event sources:** Administrator, Fraud Analyst, Client system (submits transactions),
Stream Simulator (background, every 2 s), Escalation Watcher (background, every 60 s), System
clock (timeouts / expiry).

---

## 2. Rules that act as guards

| Guard | Value |
|---|---|
| Tier | p < 0.40 → ALLOW (LOW); 0.40 ≤ p < 0.75 → REVIEW (MEDIUM); 0.75 ≤ p < 0.90 → BLOCK (HIGH); p ≥ 0.90 → BLOCK (CRITICAL) |
| Alert created | p ≥ 0.40, **or** the night rule raised the decision to REVIEW |
| BB daily limit (only for transactions submitted through the API) | sender's total today + amount > 30,000 (CASH_OUT) / 50,000 (TRANSFER) → BLOCK, risk REGULATORY |
| Night rule (only API transactions) | amount > 10,000 and local time 23:00–03:59 → ALLOW raised to REVIEW (never lowers a decision) |
| Escalation | alert not acknowledged and not escalated, and 10 minutes since it was pushed |
| Login lockout | 5 failed attempts for the same username + IP → locked for 15 minutes |
| Access token | valid 15 minutes |
| Refresh token | valid 7 days; single use (rotated on every refresh) |

---

## 3. Event list (event-centric view)

| # | Event | Initiator | Object affected | State change |
|---|---|---|---|---|
| E1 | Transaction submitted | Client system / Analyst / Stream Simulator | Transaction | none → Received |
| E2 | Daily limit exceeded | System (limit check) | Transaction | Received → Blocked (REGULATORY) |
| E3 | Neighbours fetched | System (Redis) | Transaction | → Neighbours gathered |
| E4 | Model scored | System (GNN) | Transaction | → Scored (probability known) |
| E5 | Tier decided | System (decision engine) | Transaction | → Allowed / Under review / Blocked |
| E6 | Night rule triggered | System | Transaction | Allowed → Under review |
| E7 | Decision recorded | System | Transaction, Alert | → Recorded; Alert created if guard holds |
| E8 | Alert pushed | System (WebSocket) | Alert | → Pending |
| E9 | Alert acknowledged (confirmed fraud) | Analyst | Alert | Pending / Escalated → Acknowledged |
| E10 | Alert marked false positive | Analyst | Alert | Pending / Escalated → False positive |
| E11 | SLA expired (10 min) | Escalation Watcher | Alert | Pending → Escalated |
| E12 | Transaction deleted | Administrator | Transaction, Alert | any → Deleted (final) |
| E13 | User created | Administrator | User | none → Awaiting password change |
| E14 | Password changed | User | User | Awaiting password change → Active |
| E15 | Password reset | Administrator | User | Active → Awaiting password change |
| E16 | User deactivated / reactivated | Administrator | User | Active ↔ Deactivated |
| E17 | Login submitted | User | Login attempt / Session | Logged out → Logged in, or failure counted |
| E18 | 5th failed login | User | Login attempt | Counting → Locked |
| E19 | Lockout expired (15 min) | System clock | Login attempt | Locked → Open |
| E20 | Access token expired (15 min) | System clock | Session | Logged in → Token expired |
| E21 | Refresh requested | Browser (automatic) | Session, Refresh token | Token expired → Logged in (token rotated) |
| E22 | Old refresh token reused | Attacker / stale tab | Session, Refresh token family | → Revoked (forced re-login) |
| E23 | Logout | User | Session | → Logged out |
| E24 | Stream toggled on / off | Analyst / Admin | Stream Simulator | Stopped ↔ Running |
| E25 | Stream tick (every 2 s) | Stream Simulator | Transaction | triggers E1 |

---

## 4. Objects and their states

### 4.1 Transaction (scoring life cycle) — `scoring_pipeline.py`

States: **Received** → **Limit checked** → **Neighbours gathered** → **Scored** → one of
**Allowed / Under review / Blocked / Regulatory blocked** → **Recorded** → (final) or
**Deleted** (final).

Transitions:
- Received —`transaction submitted [source = API]` / reserve amount in daily total→ Limit checked
- Received —`transaction submitted [source = stream or pattern demo]`→ Neighbours gathered (BB rules skipped)
- Limit checked —`[limit exceeded]` / mark REGULATORY→ Neighbours gathered (model still runs, decision is forced to BLOCK later)
- Limit checked —`[within limit]`→ Neighbours gathered
- Neighbours gathered —`neighbours fetched (0..50 per account)`→ Scored
- Scored —`[limit exceeded]`→ Regulatory blocked
- Scored —`[p < 0.40]`→ Allowed
- Scored —`[0.40 ≤ p < 0.75]`→ Under review
- Scored —`[p ≥ 0.75]`→ Blocked
- Allowed —`night rule [API, amount > 10,000, 23:00–03:59]` / attach rule to explanation→ Under review
- Blocked / Regulatory blocked —`/ release daily-limit reservation`→ Recorded
- Allowed / Under review —`/ save Transaction + FraudDecision; create Alert if guard holds; store in Redis; broadcast`→ Recorded
- Recorded —`deleted by admin` / remove decisions, alerts, Redis history, give back daily limit→ Deleted
- Any state —`scoring error` / release reservation→ (aborted, nothing saved)

### 4.2 Alert — `models.Alert`, `escalation.py`, `/api/fraud/alerts/{id}/acknowledge`

States: **Pending** (pushed, not acknowledged), **Escalated** (overdue, still not
acknowledged), **Acknowledged – fraud confirmed**, **Acknowledged – false positive**,
**Deleted** (final).

Transitions:
- (initial) —`decision recorded [p ≥ 0.40 or night rule forced REVIEW]` / push over WebSocket→ Pending
- Pending —`SLA expired [10 min, not acknowledged]` / set escalated, escalated_at, notify, broadcast→ Escalated
- Pending / Escalated —`acknowledged [is_false_positive = false]` / record analyst + time, broadcast→ Acknowledged – fraud confirmed
- Pending / Escalated —`acknowledged [is_false_positive = true]`→ Acknowledged – false positive
- Acknowledged (either) —`acknowledged again` / overwrite analyst, time and flag→ Acknowledged (either)
- Any —`transaction deleted by admin`→ Deleted
- Note: an escalated alert stays marked escalated after it is acknowledged.

### 4.3 User account — `models.User`, `user_service.py`

States: **Awaiting password change** (temporary password, `must_change_password = true`),
**Active**, **Deactivated** (`is_active = false`).

Transitions:
- (initial) —`user created by admin` / generate temporary password→ Awaiting password change
- (initial) —`first start with no users` / bootstrap admin from configuration→ Awaiting password change
- Awaiting password change —`password changed [current password correct, new ≥ 8 chars]`→ Active
- Active —`password reset by admin` / new temporary password→ Awaiting password change
- Active / Awaiting password change —`deactivated by admin`→ Deactivated (cannot log in; refresh fails)
- Deactivated —`reactivated by admin`→ previous state
- While Awaiting password change the user can log in but only reach the change-password page.

### 4.4 Login session — `auth.py`, `/auth/login`, `/auth/refresh`, `/auth/logout`

States: **Logged out**, **Logged in** (valid access token), **Access token expired**,
**Revoked** (refresh token family revoked).

Transitions:
- Logged out —`login [credentials correct, user active, not locked]` / issue access token (15 min), refresh token (7 days), CSRF token→ Logged in
- Logged out —`login [wrong credentials]` / count failed attempt→ Logged out
- Logged in —`access token expired [15 min]`→ Access token expired
- Access token expired —`refresh [refresh token valid and unused, user active]` / mark old token used, issue new pair (same family)→ Logged in
- Access token expired —`refresh [token expired or user deactivated]` / clear cookies→ Logged out
- Any —`refresh with an already-used token` / revoke whole token family, clear cookies→ Revoked → Logged out
- Logged in —`logout` / revoke token family, clear cookies→ Logged out

### 4.5 Login attempt counter (brute-force protection) — `rate_limit.py` (Redis)

States: **Open** (0 failures), **Counting** (1–4 failures), **Locked** (≥ 5 failures).

Transitions:
- Open —`failed login`→ Counting
- Counting —`failed login [failures < 5]`→ Counting
- Counting —`failed login [failures = 5]`→ Locked (login answers HTTP 429)
- Counting —`successful login` / clear counter→ Open
- Locked —`15 minutes pass` / counter key expires→ Open
- Any —`Redis unavailable`→ Open (fail open: login is not blocked)

### 4.6 Stream Simulator — `stream_simulator.py`, `/api/stream/toggle`

States: **Stopped**, **Running – idle** (waiting for next tick), **Running – generating**
(one transaction being created and scored).

Transitions:
- (initial) —`server start [STREAM_ENABLED]`→ Running – idle; otherwise → Stopped
- Stopped —`toggle on`→ Running – idle
- Running (any) —`toggle off`→ Stopped
- Running – idle —`tick (every 2 s)`→ Running – generating
- Running – generating —`transaction scored (E1–E7)`→ Running – idle
- Running – generating —`error` / skip this transaction, log warning→ Running – idle
- Generation inside a tick: 85% everyday traffic (shop payment, money for a relative, cash-out at
  an agent), 11% a victim's full balance drained into a shared mule, 4% start of a structuring
  burst (3–5 near-limit transfers to the same receiver on the following ticks).
