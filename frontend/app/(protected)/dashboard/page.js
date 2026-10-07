"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { getStats } from "@/lib/api";
import StatsPanel from "@/components/StatsPanel";
import LiveFeed from "@/components/LiveFeed";

// Fallback safety-net poll in case the WebSocket drops silently — the
// primary refresh path is onEvent(), fired immediately when LiveFeed's
// WebSocket delivers a new decision, not this interval.
const STATS_FALLBACK_POLL_MS = 15000;

export default function DashboardPage() {
  const [stats, setStats] = useState(null);

  const loadStats = useCallback(async () => {
    try {
      setStats(await getStats());
    } catch {
      // apiFetch already redirects to /login on repeated 401s; other errors just retry next tick
    }
  }, []);

  useEffect(() => {
    loadStats();
    const interval = setInterval(loadStats, STATS_FALLBACK_POLL_MS);
    return () => clearInterval(interval);
  }, [loadStats]);

  return (
    <div className="flex-1">
      <main className="p-6 max-w-6xl mx-auto space-y-6 w-full">
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-xl font-semibold text-white">Fraud Detection Dashboard</h1>
            <p className="text-sm text-slate-500 mt-1">Graph Neural Network — real-time transaction scoring</p>
          </div>
          <Link
            href="/scoring"
            className="rounded-md bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium px-4 py-2 transition-colors"
          >
            Score a transaction
          </Link>
        </div>
        <StatsPanel stats={stats} />
        <LiveFeed onEvent={loadStats} />
      </main>
    </div>
  );
}
