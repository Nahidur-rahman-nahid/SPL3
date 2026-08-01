"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  buildAlertsSocketUrl,
  getStreamStatus,
  isAuthenticated,
  logout,
  toggleStream,
} from "@/lib/api";

const MAX_TICKER_ITEMS = 20;
const RECONNECT_DELAY_MS = 3000;

const DECISION_STYLES = {
  ALLOW: "border-emerald-500/40 bg-emerald-500/10 text-emerald-400",
  REVIEW: "border-amber-500/40 bg-amber-500/10 text-amber-400",
  BLOCK: "border-rose-500/40 bg-rose-500/10 text-rose-400",
};

function StatTile({ label, value, accent = "text-white" }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
      <div className="text-xs text-slate-500 mb-1">{label}</div>
      <div className={`text-2xl font-semibold tabular-nums ${accent}`}>{value}</div>
    </div>
  );
}

export default function StreamMonitorPage() {
  const router = useRouter();
  const [ready, setReady] = useState(false);
  const [connected, setConnected] = useState(false);
  const [streamEnabled, setStreamEnabled] = useState(null); // null = not loaded yet
  const [toggling, setToggling] = useState(false);
  const [events, setEvents] = useState([]);
  const [sessionStats, setSessionStats] = useState({ total: 0, allow: 0, review: 0, block: 0, avgLatency: 0 });

  const startTimeRef = useRef(Date.now());
  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  useEffect(() => {
    if (!isAuthenticated()) {
      router.replace("/login");
      return;
    }
    setReady(true);
  }, [router]);

  useEffect(() => {
    if (!ready) return;
    getStreamStatus()
      .then((s) => setStreamEnabled(s.enabled))
      .catch(() => {});
  }, [ready]);

  useEffect(() => {
    if (!ready) return;
    let cancelled = false;

    function connect() {
      const ws = new WebSocket(buildAlertsSocketUrl());
      socketRef.current = ws;

      ws.onopen = () => setConnected(true);

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type !== "new_decision") return;

        setEvents((prev) => [msg, ...prev].slice(0, MAX_TICKER_ITEMS));
        setSessionStats((prev) => {
          const total = prev.total + 1;
          const latencySum = prev.avgLatency * prev.total + (msg.inference_latency_ms || 0);
          return {
            total,
            allow: prev.allow + (msg.decision === "ALLOW" ? 1 : 0),
            review: prev.review + (msg.decision === "REVIEW" ? 1 : 0),
            block: prev.block + (msg.decision === "BLOCK" ? 1 : 0),
            avgLatency: total ? latencySum / total : 0,
          };
        });
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
  }, [ready]);

  async function handleToggle() {
    setToggling(true);
    try {
      const res = await toggleStream();
      setStreamEnabled(res.enabled);
    } catch {
      // leave state as-is; button stays clickable so they can retry
    } finally {
      setToggling(false);
    }
  }

  function handleLogout() {
    logout();
    router.replace("/login");
  }

  if (!ready) return null;

  const elapsedMinutes = Math.max((Date.now() - startTimeRef.current) / 60000, 1 / 60);
  const perMinute = sessionStats.total / elapsedMinutes;

  return (
    <div className="min-h-screen flex-1 bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-4 flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-lg font-semibold">Live Stream Monitor</h1>
          <p className="text-xs text-slate-500">
            Every transaction here was scored automatically — nobody clicked a button
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/dashboard" className="text-sm text-slate-400 hover:text-white transition-colors">
            ← Dashboard
          </Link>
          <button onClick={handleLogout} className="text-sm text-slate-400 hover:text-white transition-colors">
            Log out
          </button>
        </div>
      </header>

      <main className="p-6 max-w-5xl mx-auto space-y-6">
        <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-lg px-4 py-3 flex items-start gap-2.5">
          <span className="text-indigo-400 text-sm leading-none mt-0.5">ⓘ</span>
          <p className="text-xs text-slate-400 leading-relaxed">
            <span className="text-slate-300 font-medium">Where this data comes from:</span> a built-in
            transaction simulator generates realistic scenarios (normal payments, suspicious drain
            patterns, ring-building activity) and sends each one through the same <code className="text-slate-300">/score</code> pipeline
            a real transaction would use — this is standing in for a live Kafka feed from the payment
            processor. The scoring logic itself (Redis lookup → GNN inference → decision) is identical
            either way; only the source of the transaction changes.
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg p-5 flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <span className={`w-2.5 h-2.5 rounded-full ${connected ? "bg-emerald-400 animate-pulse" : "bg-slate-600"}`} />
            <span className="text-sm text-slate-300">
              {connected ? "Connected — receiving live decisions" : "Reconnecting…"}
            </span>
          </div>
          <button
            onClick={handleToggle}
            disabled={toggling || streamEnabled === null}
            className={`text-sm font-medium px-4 py-2 rounded-md transition-colors disabled:opacity-50 ${
              streamEnabled
                ? "bg-rose-600 hover:bg-rose-500 text-white"
                : "bg-emerald-600 hover:bg-emerald-500 text-white"
            }`}
          >
            {streamEnabled === null ? "…" : streamEnabled ? "Pause Stream" : "Resume Stream"}
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
          <StatTile label="This session" value={sessionStats.total} />
          <StatTile label="Per minute" value={perMinute.toFixed(1)} />
          <StatTile label="Allow" value={sessionStats.allow} accent="text-emerald-400" />
          <StatTile label="Review" value={sessionStats.review} accent="text-amber-400" />
          <StatTile label="Block" value={sessionStats.block} accent="text-rose-400" />
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-800">
            <h2 className="text-sm font-medium text-slate-300">Incoming Transactions</h2>
          </div>
          <div className="px-4 py-2 flex items-center justify-between gap-4 flex-wrap text-[11px] uppercase tracking-wide text-slate-500 border-b border-slate-800">
            <span>Decision · Route · Source</span>
            <div className="flex items-center gap-4 shrink-0">
              <span className="w-20 text-right">Amount</span>
              <span className="w-14 text-right">Probability</span>
              <span className="w-16 text-right">Time</span>
            </div>
          </div>
          <div className="divide-y divide-slate-800 max-h-[520px] overflow-y-auto">
            {events.length === 0 && (
              <p className="px-4 py-10 text-center text-sm text-slate-500">
                Waiting for the next transaction to stream in…
              </p>
            )}
            {events.map((e) => (
              <div
                key={e.transaction_id}
                className="animate-slide-in px-4 py-3 flex items-center justify-between gap-4 flex-wrap"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <span className={`shrink-0 text-xs font-semibold px-2 py-1 rounded border ${DECISION_STYLES[e.decision] || ""}`}>
                    {e.decision} {e.risk_level}
                  </span>
                  <span className="text-sm text-slate-300 font-mono truncate">
                    {e.sender_account} → {e.receiver_account}
                  </span>
                  <span className="text-xs text-slate-500 uppercase">{e.source}</span>
                </div>
                <div className="flex items-center gap-4 text-xs text-slate-400 shrink-0">
                  <span className="w-20 text-right">{Number(e.amount).toLocaleString()}</span>
                  <span className="w-14 text-right">{(e.fraud_probability * 100).toFixed(1)}%</span>
                  <span className="w-16 text-right">{new Date(e.decided_at).toLocaleTimeString()}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
