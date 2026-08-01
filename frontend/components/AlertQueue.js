"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { acknowledgeAlert, buildAlertsSocketUrl, getResults, getUsername } from "@/lib/api";

// This page is the Alert & Decision Engine surfaced as its own feature —
// everything here is downstream of a probability ml_service.py already
// computed: which tier it landed in, WHY (real relationship-based
// explanation, not a raw neighbour dump), and what happens to it next
// (acknowledge, false-positive, or SLA escalation if nobody does).

const BADGE_STYLES = {
  ALLOW: "bg-emerald-500/15 text-emerald-400",
  REVIEW: "bg-amber-500/15 text-amber-400",
  BLOCK: "bg-rose-500/15 text-rose-400",
};

const EDGE_STYLES = {
  SAME_SENDER: "bg-sky-500/15 text-sky-400",
  SAME_RECEIVER: "bg-violet-500/15 text-violet-400",
  SUPPLIED: "bg-slate-700 text-slate-400",
};

const TABS = [
  { key: "needs_review", label: "Needs Review" },
  { key: "escalated", label: "Escalated" },
  { key: "all", label: "All Alerts" },
];

const MAX_ROWS = 100;
const RECONNECT_DELAY_MS = 3000;

function formatContribution(value) {
  if (value === null || value === undefined) return "—";
  const pct = (value * 100).toFixed(1);
  return value > 0 ? `+${pct}pp` : `${pct}pp`;
}

export default function AlertQueue() {
  const [tab, setTab] = useState("needs_review");
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [connected, setConnected] = useState(false);
  const [ackingId, setAckingId] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  const loadInitial = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getResults({ limit: MAX_ROWS, onlyAlerts: true });
      setRows(data);
    } catch {
      // WebSocket will still deliver new alert-worthy events even if this fails
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadInitial();
  }, [loadInitial]);

  useEffect(() => {
    let cancelled = false;

    function connect() {
      const ws = new WebSocket(buildAlertsSocketUrl());
      socketRef.current = ws;

      ws.onopen = () => setConnected(true);

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === "new_decision" && msg.alert_id) {
          setRows((prev) => {
            const withoutDup = prev.filter((r) => r.transaction_id !== msg.transaction_id);
            return [msg, ...withoutDup].slice(0, MAX_ROWS);
          });
        }

        if (msg.type === "alert_acknowledged") {
          setRows((prev) =>
            prev.map((r) =>
              r.alert_id === msg.alert_id
                ? { ...r, acknowledged_by: msg.acknowledged_by, is_false_positive: msg.is_false_positive }
                : r
            )
          );
        }

        if (msg.type === "alert_escalated") {
          setRows((prev) =>
            prev.map((r) => (r.alert_id === msg.alert_id ? { ...r, escalated: true, escalated_at: msg.escalated_at } : r))
          );
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (!cancelled) reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
      };
      ws.onerror = () => ws.close();
    }

    connect();
    return () => {
      cancelled = true;
      clearTimeout(reconnectTimerRef.current);
      socketRef.current?.close();
    };
  }, []);

  async function handleAcknowledge(row, isFalsePositive) {
    setAckingId(row.alert_id);
    try {
      const ackedBy = getUsername() || "unknown";
      await acknowledgeAlert(row.alert_id, ackedBy, isFalsePositive);
      setRows((prev) =>
        prev.map((r) => (r.alert_id === row.alert_id ? { ...r, acknowledged_by: ackedBy, is_false_positive: isFalsePositive } : r))
      );
    } catch {
      // leave the row as-is; the buttons stay visible so they can retry
    } finally {
      setAckingId(null);
    }
  }

  // This page is specifically the REVIEW/BLOCK review queue — never show
  // ALLOW here even if the backend's alert threshold is misconfigured or a
  // stale server hasn't picked up the current policy. Belt-and-suspenders
  // on top of the `only_alerts=true` server-side filter in loadInitial().
  const alertWorthyRows = rows.filter((r) => r.decision !== "ALLOW");

  const visibleRows = alertWorthyRows.filter((r) => {
    if (tab === "needs_review") return !!r.alert_id && !r.acknowledged_by;
    if (tab === "escalated") return r.escalated;
    return true;
  });

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-1.5">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
                tab === t.key
                  ? "bg-indigo-600 border-indigo-500 text-white"
                  : "border-slate-700 text-slate-400 hover:border-slate-500"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <span className={`text-xs flex items-center gap-1.5 ${connected ? "text-emerald-400" : "text-slate-500"}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-emerald-400" : "bg-slate-600"}`} />
          {connected ? "live" : "reconnecting…"}
        </span>
      </div>

      <div className="max-h-[600px] overflow-y-auto divide-y divide-slate-800">
        {!loading && visibleRows.length === 0 && (
          <p className="px-4 py-10 text-center text-sm text-slate-500">
            {tab === "needs_review" ? "Nothing waiting on review." : tab === "escalated" ? "No escalated alerts." : "No alerts yet."}
          </p>
        )}

        {visibleRows.map((r) => {
          const expanded = expandedId === r.transaction_id;
          return (
            <div key={r.transaction_id}>
              <button
                onClick={() => setExpandedId(expanded ? null : r.transaction_id)}
                className="w-full text-left px-4 py-3 flex items-center justify-between gap-4 flex-wrap hover:bg-slate-800/50 transition-colors"
              >
                <div className="flex items-center gap-3 min-w-0">
                  {r.escalated && (
                    <span className="shrink-0 text-xs font-semibold px-2 py-0.5 rounded bg-rose-600 text-white animate-pulse">
                      SLA BREACHED
                    </span>
                  )}
                  <span className={`shrink-0 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${BADGE_STYLES[r.decision] || ""}`}>
                    {r.decision} {r.risk_level}
                  </span>
                  <span className="text-sm text-slate-300 font-mono truncate">
                    {r.sender_account} → {r.receiver_account}
                  </span>
                </div>
                <div className="flex items-center gap-4 text-xs text-slate-400 shrink-0">
                  <span>{Number(r.amount).toLocaleString()}</span>
                  <span>{(r.fraud_probability * 100).toFixed(1)}%</span>
                  <span>{new Date(r.decided_at).toLocaleTimeString()}</span>
                  {r.acknowledged_by ? (
                    <span className="text-slate-500">{r.is_false_positive ? "false positive" : "acked"} · {r.acknowledged_by}</span>
                  ) : (
                    <span className="text-amber-400">needs review</span>
                  )}
                </div>
              </button>

              {expanded && (
                <div className="px-4 pb-4 bg-slate-950/50">
                  <p className="text-sm text-slate-300 mb-3">{r.explanation?.summary || "No explanation recorded."}</p>

                  {r.explanation?.neighbours?.length > 0 && (
                    <div className="overflow-x-auto mb-3">
                      <table className="w-full text-xs">
                        <thead className="text-slate-500 uppercase">
                          <tr>
                            <th className="text-left py-1 pr-3">Relationship</th>
                            <th className="text-left py-1 pr-3">Type</th>
                            <th className="text-right py-1 pr-3">Amount</th>
                            <th className="text-right py-1 pr-3">Contribution</th>
                            <th className="text-left py-1">Prior decision</th>
                          </tr>
                        </thead>
                        <tbody>
                          {r.explanation.neighbours.map((n) => (
                            <tr key={n.transaction_id} className="border-t border-slate-800">
                              <td className="py-1.5 pr-3">
                                <span className={`px-1.5 py-0.5 rounded ${EDGE_STYLES[n.edge_type] || ""}`}>{n.edge_type}</span>
                              </td>
                              <td className="py-1.5 pr-3 text-slate-300">{n.type}</td>
                              <td className="py-1.5 pr-3 text-right text-slate-300 tabular-nums">{Number(n.amount).toLocaleString()}</td>
                              <td className={`py-1.5 pr-3 text-right tabular-nums ${(n.contribution || 0) > 0 ? "text-rose-400" : "text-emerald-400"}`}>
                                {formatContribution(n.contribution)}
                              </td>
                              <td className="py-1.5">
                                {n.prior_decision ? (
                                  <span className={`px-1.5 py-0.5 rounded ${BADGE_STYLES[n.prior_decision] || ""}`}>{n.prior_decision}</span>
                                ) : (
                                  <span className="text-slate-600">first time seen</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {!r.acknowledged_by && (
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleAcknowledge(r, false)}
                        disabled={ackingId === r.alert_id}
                        className="text-xs px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-50"
                      >
                        Acknowledge
                      </button>
                      <button
                        onClick={() => handleAcknowledge(r, true)}
                        disabled={ackingId === r.alert_id}
                        className="text-xs px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-400 disabled:opacity-50"
                      >
                        False positive
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
