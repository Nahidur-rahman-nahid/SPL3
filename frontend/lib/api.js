const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

function getToken() {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("token");
}

async function apiFetch(path, options = {}) {
  const token = getToken();
  const headers = { ...(options.headers || {}) };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (options.body && typeof options.body === "string") {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (res.status === 401) {
    localStorage.removeItem("token");
    localStorage.removeItem("role");
    if (typeof window !== "undefined") window.location.href = "/login";
    throw new Error("Session expired — please log in again.");
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Request failed (${res.status})`);
  }
  return res.json();
}

export async function login(username, password) {
  const body = new URLSearchParams();
  body.append("username", username);
  body.append("password", password);

  const res = await fetch(`${API_BASE}/auth/login`, { method: "POST", body });
  if (!res.ok) throw new Error("Invalid username or password.");

  const data = await res.json();
  localStorage.setItem("token", data.access_token);
  localStorage.setItem("role", data.role);
  localStorage.setItem("username", username);
  return data;
}

export function logout() {
  localStorage.removeItem("token");
  localStorage.removeItem("role");
  localStorage.removeItem("username");
}

export function isAuthenticated() {
  return !!getToken();
}

export function getRole() {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("role");
}

export function getUsername() {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("username");
}

// WebSocket can't send an Authorization header from a browser, so the token
// travels as a query param instead (validated server-side in /ws/alerts).
export function buildAlertsSocketUrl() {
  const token = getToken();
  const wsBase = API_BASE.replace(/^http/, "ws");
  return `${wsBase}/ws/alerts?token=${encodeURIComponent(token || "")}`;
}

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

export function acknowledgeAlert(alertId, ackedBy, isFalsePositive) {
  return apiFetch(`/api/fraud/alerts/${alertId}/acknowledge`, {
    method: "POST",
    body: JSON.stringify({ acknowledged_by: ackedBy, is_false_positive: isFalsePositive }),
  });
}

export function getStreamStatus() {
  return apiFetch("/api/stream/status");
}

export function toggleStream() {
  return apiFetch("/api/stream/toggle", { method: "POST" });
}
