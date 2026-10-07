"use client";

import Link from "next/link";
import ScoreForm from "@/components/ScoreForm";

// Manual / on-demand scoring, split out of the dashboard into its own page.
// Lives at /scoring, NOT /score — /score is proxied straight to the
// backend's POST /score endpoint (next.config.mjs), so a page there would
// swallow the API call ScoreForm makes.

const PIPELINE_STEPS = [
  {
    title: "Build the subgraph",
    body: "Recent transactions of the sender and receiver are pulled from Redis and linked as SAME_SENDER / SAME_RECEIVER neighbours.",
  },
  {
    title: "GNN inference",
    body: "The GraphSAGE pathway of FraudGNN scores the transaction together with its neighbours and returns a fraud probability.",
  },
  {
    title: "Rules & limits",
    body: "Bangladesh Bank daily limits and the night-time high-value rule run on top of the model and can block or hold it for review.",
  },
  {
    title: "Decision & explanation",
    body: "The probability is tiered into ALLOW / REVIEW / BLOCK, explained per relationship, and raised as an alert when needed.",
  },
];

export default function ScoringPage() {
  return (
    <div className="flex-1">
      <main className="p-6 max-w-6xl mx-auto space-y-6 w-full">
        <div>
          <h1 className="text-xl font-semibold text-white">Score a Transaction</h1>
          <p className="text-sm text-slate-500 mt-1">
            Run a single transaction through the full detection pipeline and see the decision, its explanation and the
            subgraph it was scored on
          </p>
        </div>

        <ScoreForm />

        <section className="bg-slate-900 border border-slate-800 rounded-lg p-5">
          <h2 className="text-sm font-medium text-slate-300 mb-4">What happens when you score</h2>
          <ol className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {PIPELINE_STEPS.map((step, i) => (
              <li key={step.title} className="flex gap-3">
                <span className="shrink-0 w-6 h-6 rounded-full bg-indigo-500/15 text-indigo-400 text-xs font-semibold flex items-center justify-center">
                  {i + 1}
                </span>
                <div>
                  <p className="text-sm text-slate-200">{step.title}</p>
                  <p className="text-xs text-slate-500 mt-1 leading-relaxed">{step.body}</p>
                </div>
              </li>
            ))}
          </ol>
          <p className="text-xs text-slate-500 mt-5 pt-4 border-t border-slate-800">
            Every result is saved and appears on the{" "}
            <Link href="/dashboard" className="text-indigo-400 hover:text-indigo-300">dashboard feed</Link>
            ; anything flagged also lands in the{" "}
            <Link href="/alerts" className="text-indigo-400 hover:text-indigo-300">alerts queue</Link>.
          </p>
        </section>
      </main>
    </div>
  );
}
