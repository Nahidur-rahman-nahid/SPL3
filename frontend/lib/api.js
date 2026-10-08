// All requests go through Next.js rewrites (next.config.mjs) so the browser
// sees them as same-origin — this fixes SameSite=lax cookies between
// localhost:3000 (frontend) and localhost:8000 (backend).
const API_BASE = "";

// Auth is entirely cookie-based (see backend/auth.py) — access_token and
// refresh_token are httpOnly (never touched here), csrf_token is the one
// cookie the frontend does read, so it can be echoed back as a header on
// mutating requests (double-submit CSRF defense).
function getCsrfToken() {
  if (typeof document === "undefined") return null;
  const match = document.cookie.match(/(?:^|; )csrf_token=([^;]*)/);
  return match ? decodeURIComponent(match[1]) : null;
}

const MUTATING_METHODS = new Set(["POST", "PATCH", "PUT", "DELETE"]);

// Pages a signed-out visitor may stay on (no redirect to /login from here).
const PUBLIC_PATHS = new Set(["/login", "/forgot-password", "/reset-password"]);

// Concurrent 401s must share one in-flight /auth/refresh call rather than
// each firing their own — the backend's refresh-token rotation treats a
// second, independent use of the same (still valid) refresh token as reuse
// and revokes the whole session, which would otherwise force a surprise
// logout whenever two requests raced.
let refreshPromise = null;

function refreshSession() {
  if (!refreshPromise) {
    refreshPromise = fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      credentials: "include",
      headers: { "X-CSRF-Token": getCsrfToken() || "" },
    })
      .then((res) => {
        if (!res.ok) throw new Error("refresh failed");
        return res.json();
      })
      .finally(() => {
        refreshPromise = null;
      });
  }
  return refreshPromise;
}

async function apiFetch(path, options = {}, _isRetry = false) {
  const method = (options.method || "GET").toUpperCase();
  const headers = { ...(options.headers || {}) };
  if (options.body && typeof options.body === "string") {
    headers["Content-Type"] = "application/json";
  }
  if (MUTATING_METHODS.has(method)) {
    headers["X-CSRF-Token"] = getCsrfToken() || "";
  }

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers, credentials: "include" });

  if (res.status === 401 && !_isRetry) {
    try {
      await refreshSession();
      return apiFetch(path, options, true);
    } catch {
      // Not from a public page: AuthProvider (root layout) calls /auth/me on
      // every page load, including /login itself, so redirecting from there
      // would reload /login → /auth/me → 401 → redirect … in an endless loop
      // (and would push a signed-out user off the forgot/reset pages).
      if (typeof window !== "undefined" && !PUBLIC_PATHS.has(window.location.pathname)) {
        window.location.href = "/login";
      }
      throw new Error("Session expired — please log in again.");
    }
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Request failed (${res.status})`);
  }
  if (res.status === 204) return null;
  return res.json();
}

// ---------------------------------------------------------------- session

export async function login(username, password) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    if (res.status === 429) throw new Error("Too many failed attempts. Try again in a few minutes.");
    throw new Error("Invalid username or password.");
  }
  return res.json(); // UserOut
}

export async function logoutUser() {
  try {
    await fetch(`${API_BASE}/auth/logout`, {
      method: "POST",
      credentials: "include",
      headers: { "X-CSRF-Token": getCsrfToken() || "" },
    });
  } catch {
    // best-effort — the client-side session state gets cleared regardless
  }
}

export function getMe() {
  return apiFetch("/auth/me");
}

export function changePassword(currentPassword, newPassword) {
  return apiFetch("/auth/change-password", {
    method: "POST",
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
}

// Signed-out "forgot password" flow. Errors carry the backend's message.
async function publicPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? "Please check the form." : data.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

export function requestPasswordReset(email) {
  return publicPost("/auth/forgot-password", { email });
}

export function resetPasswordWithToken(token, newPassword) {
  return publicPost("/auth/reset-password", { token, new_password: newPassword });
}

// ------------------------------------------------------- admin: user mgmt

export function listUsers() {
  return apiFetch("/auth/users");
}

export function createUser({ username, email, role }) {
  return apiFetch("/auth/users", {
    method: "POST",
    body: JSON.stringify({ username, email: email || null, role }),
  });
}

export function updateUser(userId, { role, is_active, email } = {}) {
  const payload = {};
  if (role !== undefined) payload.role = role;
  if (is_active !== undefined) payload.is_active = is_active;
  if (email !== undefined) payload.email = email;
  return apiFetch(`/auth/users/${userId}`, { method: "PATCH", body: JSON.stringify(payload) });
}

export function resetUserPassword(userId) {
  return apiFetch(`/auth/users/${userId}/reset-password`, { method: "POST" });
}

// ------------------------------------------------------------- websocket
// Browsers attach cookies to a WebSocket handshake the same way they do to
// a normal request, so the access_token cookie rides along automatically —
// no token in the URL (and therefore no token in server logs/browser
// history) unlike the old ?token= query param.
export function buildAlertsSocketUrl() {
  // Use the browser's own origin so WebSocket goes through Next.js proxy.
  if (typeof window === "undefined") return "ws://127.0.0.1:3000/ws/alerts";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/alerts`;
}

// -------------------------------------------------------------- fraud API

export function scoreTransaction(payload) {
  return apiFetch("/score", { method: "POST", body: JSON.stringify(payload) });
}

export function getResults({ limit = 50, riskLevel, onlyAlerts, unacknowledgedOnly } = {}) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (riskLevel) params.set("risk_level", riskLevel);
  if (onlyAlerts) params.set("only_alerts", "true");
  if (unacknowledgedOnly) params.set("unacknowledged_only", "true");
  return apiFetch(`/api/fraud/results?${params.toString()}`);
}

export function getStats() {
  return apiFetch("/api/fraud/stats");
}

export function acknowledgeAlert(alertId, isFalsePositive) {
  return apiFetch(`/api/fraud/alerts/${alertId}/acknowledge`, {
    method: "POST",
    body: JSON.stringify({ is_false_positive: isFalsePositive }),
  });
}

export function getStreamStatus() {
  return apiFetch("/api/stream/status");
}

export function toggleStream() {
  return apiFetch("/api/stream/toggle", { method: "POST" });
}

// ------------------------------------------ admin: delete transactions

export function deleteTransactions(transactionIds) {
  return apiFetch("/api/transactions", {
    method: "DELETE",
    body: JSON.stringify({ transaction_ids: transactionIds }),
  });
}

// source: "stream" | "manual" | "all"; before: optional Date — only rows
// received before it.
function purgeParams(source, before) {
  return { source, ...(before ? { before: before.toISOString() } : {}) };
}

export function getPurgePreview(source, before) {
  const params = new URLSearchParams(purgeParams(source, before));
  return apiFetch(`/api/transactions/purge-preview?${params.toString()}`);
}

export function purgeTransactions(source, before) {
  return apiFetch("/api/transactions/purge", {
    method: "POST",
    body: JSON.stringify(purgeParams(source, before)),
  });
}

// ------------------------------------------------ fraud patterns demo page

export function getPatterns() {
  return apiFetch("/api/patterns");
}

// variant: "fraud" (the pattern) | "lookalike" (its legitimate twin)
export function runPattern(patternId, variant) {
  return apiFetch(`/api/patterns/${patternId}/run?variant=${variant}`, { method: "POST" });
}
