"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { acknowledgeAlert, buildAlertsSocketUrl, getResults, getUsername } from "@/lib/api";

const ROW_STYLES = {
  ALLOW: "border-l-4 border-emerald-500",
  REVIEW: "border-l-4 border-amber-500",
  BLOCK: "border-l-4 border-rose-500",
};

const BADGE_STYLES = {
  ALLOW: "bg-emerald-500/15 text-emerald-400",
  REVIEW: "bg-amber-500/15 text-amber-400",
  BLOCK: "bg-rose-500/15 text-rose-400",
};

const MAX_ROWS = 50;
const RECONNECT_DELAY_MS = 3000;

export default function LiveFeed({ onEvent }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [connected, setConnected] = useState(false);
  const [ackingId, setAckingId] = useState(null);
  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  const loadInitial = useCallback(async () => {
    try {
      const data = await getResults(MAX_ROWS);
      setRows(data);
    } catch {
      // WebSocket will still deliver new events even if this initial load fails
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

        if (msg.type === "new_decision") {
          setRows((prev) => {
            const withoutDup = prev.filter((r) => r.transaction_id !== msg.transaction_id);
            return [msg, ...withoutDup].slice(0, MAX_ROWS);
          });
          onEvent?.();
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
      };

      ws.onclose = () => {
        setConnected(false);
        if (!cancelled) {
          reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };

      ws.onerror = () => ws.close();
    }

    connect();
    return () => {
      cancelled = true;
      clearTimeout(reconnectTimerRef.current);
      socketRef.current?.close();
    };
  }, [onEvent]);

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

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
        <h2 className="text-sm font-medium text-slate-300">Live Transaction Feed</h2>
        <span className={`text-xs flex items-center gap-1.5 ${connected ? "text-emerald-400" : "text-slate-500"}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-emerald-400" : "bg-slate-600"}`} />
          {connected ? "live" : "reconnecting…"}
        </span>
      </div>
      <div className="overflow-x-auto max-h-[420px] overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="text-xs text-slate-500 uppercase sticky top-0 bg-slate-900">
            <tr>
              <th className="text-left px-4 py-2">Time</th>
              <th className="text-left px-4 py-2">Transaction</th>
              <th className="text-left px-4 py-2">Sender → Receiver</th>
              <th className="text-right px-4 py-2">Amount</th>
              <th className="text-right px-4 py-2">Probability</th>
              <th className="text-left px-4 py-2">Decision</th>
              <th className="text-left px-4 py-2">Alert</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.transaction_id} className={`${ROW_STYLES[r.decision] || ""} hover:bg-slate-800/50`}>
                <td className="px-4 py-2 text-slate-400 whitespace-nowrap">
                  {new Date(r.decided_at).toLocaleTimeString()}
                </td>
                <td className="px-4 py-2 font-mono text-slate-300">{r.transaction_id}</td>
                <td className="px-4 py-2 text-slate-300 whitespace-nowrap">
                  {r.sender_account} → {r.receiver_account}
                </td>
                <td className="px-4 py-2 text-right text-slate-300">{Number(r.amount).toLocaleString()}</td>
                <td className="px-4 py-2 text-right text-slate-300">{(r.fraud_probability * 100).toFixed(1)}%</td>
                <td className="px-4 py-2">
                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${BADGE_STYLES[r.decision] || ""}`}>
                    {r.decision} · {r.risk_level}
                  </span>
                </td>
                <td className="px-4 py-2 whitespace-nowrap">
                  {!r.alert_id && <span className="text-slate-600 text-xs">—</span>}
                  {r.alert_id && r.acknowledged_by && (
                    <span className="text-xs text-slate-400">
                      {r.is_false_positive ? "false positive" : "acknowledged"} · {r.acknowledged_by}
                    </span>
                  )}
                  {r.alert_id && !r.acknowledged_by && (
                    <div className="flex gap-1.5">
                      <button
                        onClick={() => handleAcknowledge(r, false)}
                        disabled={ackingId === r.alert_id}
                        className="text-xs px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-50"
                      >
                        Acknowledge
                      </button>
                      <button
                        onClick={() => handleAcknowledge(r, true)}
                        disabled={ackingId === r.alert_id}
                        className="text-xs px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-400 disabled:opacity-50"
                      >
                        False positive
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
            {!loading && rows.length === 0 && (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-slate-500">
                  No transactions scored yet — try the form above.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
