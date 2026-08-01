"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { getRole, isAuthenticated, logout } from "@/lib/api";
import AlertQueue from "@/components/AlertQueue";

export default function AlertsPage() {
  const router = useRouter();
  const [ready, setReady] = useState(false);
  const [role, setRole] = useState("");

  useEffect(() => {
    if (!isAuthenticated()) {
      router.replace("/login");
      return;
    }
    setRole(getRole() || "");
    setReady(true);
  }, [router]);

  function handleLogout() {
    logout();
    router.replace("/login");
  }

  if (!ready) return null;

  return (
    <div className="min-h-screen flex-1 bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-4 flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-lg font-semibold">Alerts & Review Queue</h1>
          <p className="text-xs text-slate-500">
            The Alert &amp; Decision Engine — why each transaction was flagged, and what happens if nobody reviews it
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/dashboard" className="text-sm text-slate-400 hover:text-white transition-colors">
            ← Dashboard
          </Link>
          <Link href="/stream" className="text-sm text-slate-400 hover:text-white transition-colors">
            Live Stream Monitor →
          </Link>
          {role && <span className="text-xs px-2 py-1 rounded bg-slate-800 text-slate-300">{role}</span>}
          <button onClick={handleLogout} className="text-sm text-slate-400 hover:text-white transition-colors">
            Log out
          </button>
        </div>
      </header>

      <main className="p-6 max-w-5xl mx-auto space-y-6">
        <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-lg px-4 py-3 flex items-start gap-2.5">
          <span className="text-indigo-400 text-sm leading-none mt-0.5">ⓘ</span>
          <p className="text-xs text-slate-400 leading-relaxed">
            <span className="text-slate-300 font-medium">How this differs from the Detection Engine:</span> the GNN
            (<code className="text-slate-300">ml_service.py</code>) only outputs a raw fraud probability. Everything
            below — the ALLOW/REVIEW/BLOCK tier, the explanation of which relationships drove the score, whether an
            alert gets raised, and SLA escalation if it sits unacknowledged — is decided separately, in{" "}
            <code className="text-slate-300">decision_engine.py</code>. Click a row to see exactly which neighbour
            relationships contributed to its score.
          </p>
        </div>

        <AlertQueue />
      </main>
    </div>
  );
}
