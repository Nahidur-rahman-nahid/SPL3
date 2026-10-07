"use client";

import AlertQueue from "@/components/AlertQueue";

export default function AlertsPage() {
  return (
    <div className="flex-1">
      <main className="p-6 max-w-5xl mx-auto space-y-6 w-full">
        <div>
          <h1 className="text-xl font-semibold text-white">Alerts & Review Queue</h1>
          <p className="text-sm text-slate-500 mt-1">The Alert &amp; Decision Engine — why each transaction was flagged, and what happens if nobody reviews it</p>
        </div>
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
