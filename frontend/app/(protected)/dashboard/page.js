"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { getStats } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import StatsPanel from "@/components/StatsPanel";
import ScoreForm from "@/components/ScoreForm";
import LiveFeed from "@/components/LiveFeed";

// Fallback safety-net poll in case the WebSocket drops silently — the
// primary refresh path is onEvent(), fired immediately when LiveFeed's
// WebSocket delivers a new decision, not this interval.
const STATS_FALLBACK_POLL_MS = 15000;

export default function DashboardPage() {
  const router = useRouter();
  const { user, role, logout } = useAuth();
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

  async function handleLogout() {
    await logout();
    router.replace("/login");
  }

  return (
    <div className="min-h-screen flex-1 bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold">Fraud Detection Dashboard</h1>
          <p className="text-xs text-slate-500">Graph Neural Network — real-time transaction scoring</p>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/alerts" className="text-sm text-slate-400 hover:text-white transition-colors">
            Alerts & Review Queue →
          </Link>
          <Link href="/stream" className="text-sm text-slate-400 hover:text-white transition-colors">
            Live Stream Monitor →
          </Link>
          {role === "ADMIN" && (
            <Link href="/admin/users" className="text-sm text-slate-400 hover:text-white transition-colors">
              Manage Users →
            </Link>
          )}
          <Link href="/account" className="text-sm text-slate-400 hover:text-white transition-colors">
            {user?.username}
          </Link>
          {role && <span className="text-xs px-2 py-1 rounded bg-slate-800 text-slate-300">{role}</span>}
          <button onClick={handleLogout} className="text-sm text-slate-400 hover:text-white transition-colors">
            Log out
          </button>
        </div>
      </header>

      <main className="p-6 max-w-6xl mx-auto space-y-6">
        <StatsPanel stats={stats} />
        <ScoreForm onScored={loadStats} />
        <LiveFeed onEvent={loadStats} />
      </main>
    </div>
  );
}
