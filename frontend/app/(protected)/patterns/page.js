"use client";

import { useEffect, useRef, useState } from "react";
import { getPatterns, runPattern } from "@/lib/api";

// Fraud Patterns demo: replays each injected fraud pattern — and its
// legitimate look-alike — through the real scoring pipeline one
// transaction at a time (backend/pattern_demo.py). The point to see: the
// fraud group's scores climb as related transactions accumulate, while the
// look-alike (for five patterns: the exact same transactions between
// unrelated accounts) stays low. Only the graph differs.

const DECISION_STYLES = {
  ALLOW: { bar: "bg-emerald-500", badge: "bg-emerald-500/15 text-emerald-400" },
  REVIEW: { bar: "bg-amber-500", badge: "bg-amber-500/15 text-amber-400" },
  BLOCK: { bar: "bg-rose-500", badge: "bg-rose-500/15 text-rose-400" },
};

const REVEAL_MS = 450; // one transaction at a time, so the build-up is visible

function pct(p) {
  return `${(p * 100).toFixed(1)}%`;
}

function ScoreBar({ probability, decision }) {
  return (
    <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-500 ${DECISION_STYLES[decision]?.bar || "bg-slate-500"}`}
        style={{ width: `${Math.max(probability * 100, 1.5)}%` }}
      />
    </div>
  );
}

function StepRow({ step }) {
  return (
    <li className="py-2.5 border-t border-slate-800 first:border-t-0">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="text-slate-400 truncate">
          <span className="text-slate-500 tabular-nums mr-1.5">{step.index}.</span>
          {step.note}
        </span>
        <span className={`shrink-0 px-1.5 py-0.5 rounded font-medium ${DECISION_STYLES[step.decision]?.badge || ""}`}>
          {step.decision}
        </span>
      </div>
      <div className="mt-1.5 flex items-center gap-3">
        <div className="flex-1">
          <ScoreBar probability={step.probability} decision={step.decision} />
        </div>
        <span className="w-14 text-right text-sm font-semibold text-white tabular-nums">{pct(step.probability)}</span>
      </div>
      <div className="mt-1 flex items-center justify-between text-[11px] text-slate-500">
        <span>
          {step.type} BDT {step.amount.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          {step.share_of_balance != null && ` · ${Math.round(step.share_of_balance * 100)}% of balance`}
          {step.receiver_was_empty && " · new account"}
        </span>
        <span>
          linked to <span className="text-slate-300">{step.neighbour_count}</span> · alone {pct(step.alone_probability)}
        </span>
      </div>
    </li>
  );
}

function RunColumn({ tone, title, summary, result, shown, running, onRun, disabled }) {
  const accent = tone === "fraud" ? "border-rose-500/40" : "border-emerald-500/40";
  const steps = result ? result.steps.slice(0, shown) : [];
  const done = result && shown >= result.steps.length;

  return (
    <div className={`bg-slate-900 border border-slate-800 border-t-2 ${accent} rounded-lg p-4 flex flex-col`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className={`text-[11px] uppercase tracking-wider ${tone === "fraud" ? "text-rose-400" : "text-emerald-400"}`}>
            {tone === "fraud" ? "Fraud pattern" : "Legitimate look-alike"}
          </p>
          <h3 className="text-sm font-medium text-white mt-0.5">{title}</h3>
          <p className="text-xs text-slate-500 mt-1">{summary}</p>
        </div>
        <button
          onClick={onRun}
          disabled={disabled}
          className="shrink-0 rounded-md bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-medium px-3 py-1.5 transition-colors"
        >
          {running ? "Running…" : result ? "Run again" : "Run"}
        </button>
      </div>

      {done && (
        <div className="mt-3 grid grid-cols-3 gap-2 text-center">
          <div className="rounded-md bg-slate-950/60 py-2">
            <p className="text-lg font-semibold text-white tabular-nums">
              {result.flagged}/{result.steps.length}
            </p>
            <p className="text-[11px] text-slate-500">flagged</p>
          </div>
          <div className="rounded-md bg-slate-950/60 py-2">
            <p className="text-lg font-semibold text-white tabular-nums">{pct(result.mean_probability)}</p>
            <p className="text-[11px] text-slate-500">avg score</p>
          </div>
          <div className="rounded-md bg-slate-950/60 py-2">
            <p className="text-lg font-semibold text-slate-300 tabular-nums">{pct(result.mean_alone_probability)}</p>
            <p className="text-[11px] text-slate-500">avg if alone</p>
          </div>
        </div>
      )}

      {result && !result.redis_available && (
        <p className="mt-3 text-xs text-amber-400">
          Redis is unavailable, so no related transactions could be found. Every transaction was scored on its own.
        </p>
      )}

      <ul className="mt-2">
        {steps.map((s) => (
          <StepRow key={s.transaction_id} step={s} />
        ))}
      </ul>
      {!result && !running && (
        <p className="mt-6 mb-4 text-center text-xs text-slate-600">Press Run to send these transactions through the live system.</p>
      )}
    </div>
  );
}

export default function PatternsPage() {
  const [patterns, setPatterns] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [results, setResults] = useState({}); // `${id}:${variant}` -> result
  const [shown, setShown] = useState({}); // same key -> steps revealed so far
  const [running, setRunning] = useState(null);
  const [error, setError] = useState("");
  const timers = useRef([]);

  useEffect(() => {
    getPatterns()
      .then((data) => {
        setPatterns(data.patterns);
        setModelInfo(data.model);
        setSelectedId(data.patterns[0]?.id ?? null);
      })
      .catch((err) => setError(err.message || "Failed to load patterns."));
    const pending = timers.current;
    return () => pending.forEach(clearTimeout);
  }, []);

  async function handleRun(variant) {
    const key = `${selectedId}:${variant}`;
    setRunning(key);
    setError("");
    try {
      const result = await runPattern(selectedId, variant);
      setResults((r) => ({ ...r, [key]: result }));
      setShown((s) => ({ ...s, [key]: 0 }));
      result.steps.forEach((_, i) => {
        timers.current.push(setTimeout(() => setShown((s) => ({ ...s, [key]: i + 1 })), (i + 1) * REVEAL_MS));
      });
    } catch (err) {
      setError(err.message || "Run failed.");
    } finally {
      setRunning(null);
    }
  }

  const selected = patterns.find((p) => p.id === selectedId);
  const retrained = modelInfo?.trained_on && modelInfo.trained_on !== "PaySim";

  return (
    <div className="flex-1">
      <main className="p-6 max-w-6xl mx-auto space-y-6 w-full">
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-xl font-semibold text-white">Fraud Patterns</h1>
            <p className="text-sm text-slate-500 mt-1">
              Fraud that only shows up in the graph: each transaction looks normal on its own, the group gives it away
            </p>
          </div>
          {modelInfo && (
            <span
              className={`text-xs px-2.5 py-1 rounded-full border ${
                retrained ? "border-emerald-500/40 text-emerald-400" : "border-amber-500/40 text-amber-400"
              }`}
              title={retrained ? "" : "This model was trained on PaySim only; retrain with the injected patterns to see them detected."}
            >
              Model: {modelInfo.trained_on}
            </span>
          )}
        </div>

        {error && <p className="text-sm text-rose-400">{error}</p>}

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {patterns.map((p, i) => (
            <button
              key={p.id}
              onClick={() => setSelectedId(p.id)}
              className={`text-left rounded-lg border p-4 transition-colors ${
                p.id === selectedId
                  ? "border-indigo-500 bg-indigo-500/10"
                  : "border-slate-800 bg-slate-900 hover:border-slate-700"
              }`}
            >
              <p className="text-[11px] text-slate-500 tabular-nums">F{i + 1}</p>
              <p className="text-sm font-medium text-white mt-0.5">{p.title}</p>
              <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">{p.summary}</p>
            </button>
          ))}
        </div>

        {selected && (
          <section className="space-y-3">
            <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-lg px-4 py-3 text-xs text-slate-300">
              <span className="font-medium text-indigo-300">What gives it away: </span>
              {selected.signal} Each transaction is scored through the real pipeline (Redis neighbours → GNN →
              decision); <span className="text-slate-400">linked to</span> is how many related transactions the
              system found, and <span className="text-slate-400">alone</span> is the score the same transaction would
              get with no neighbours.
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              {["fraud", "lookalike"].map((variant) => {
                const key = `${selected.id}:${variant}`;
                return (
                  <RunColumn
                    key={key}
                    tone={variant}
                    title={variant === "fraud" ? selected.title : selected.lookalike_title}
                    summary={variant === "fraud" ? selected.summary : selected.lookalike_summary}
                    result={results[key]}
                    shown={shown[key] ?? 0}
                    running={running === key}
                    disabled={running !== null}
                    onRun={() => handleRun(variant)}
                  />
                );
              })}
            </div>
            <p className="text-xs text-slate-600">
              Demo transactions are saved like any other (ids starting with <code>pattern-</code>). Remove them any
              time with Admin → Data Cleanup → Manually scored data. Bangladesh Bank daily limits are not applied to
              demo runs, so the GNN&apos;s own decision is visible.
            </p>
          </section>
        )}
      </main>
    </div>
  );
}
