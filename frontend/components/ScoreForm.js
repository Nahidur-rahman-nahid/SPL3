"use client";

import { useState } from "react";
import { scoreTransaction } from "@/lib/api";
import SubgraphViz from "@/components/SubgraphViz";

// Three ready-made scenarios so a demo doesn't need manual data entry —
// click a preset, hit Score, watch the decision come back live.
const PRESETS = {
  normal: {
    label: "Normal Payment",
    sender_account: "C_CUSTOMER_100",
    receiver_account: "M_MERCHANT_200",
    features: {
      step: 5, type: "TRANSFER", amount: 5000, oldbalanceOrg: 25000,
      newbalanceOrig: 20000, oldbalanceDest: 10000, newbalanceDest: 15000,
    },
    neighbours: [],
  },
  // Both fraud presets are graph patterns: the transaction alone looks
  // normal (retrained model: 0.1% and 12.6%), the supplied related
  // transactions give it away (63% REVIEW, 95% BLOCK). Amounts stay under
  // the BDT 50,000 daily limit so the GNN's decision is what shows.
  suspicious: {
    label: "Suspicious (structuring: near-limit splits)",
    sender_account: "C_SPLITTER_701",
    receiver_account: "C_RECEIVER_702",
    features: {
      step: 11, type: "TRANSFER", amount: 47500, oldbalanceOrg: 302700,
      newbalanceOrig: 255200, oldbalanceDest: 187300, newbalanceDest: 234800,
    },
    neighbours: [
      { step: 10, type: "TRANSFER", amount: 48200, oldbalanceOrg: 400000, newbalanceOrig: 351800, oldbalanceDest: 90000, newbalanceDest: 138200 },
      { step: 10, type: "TRANSFER", amount: 49100, oldbalanceOrg: 351800, newbalanceOrig: 302700, oldbalanceDest: 138200, newbalanceDest: 187300 },
    ],
  },
  highrisk: {
    label: "High-Risk (mule hub: victims drained into one account)",
    sender_account: "C_VICTIM_801",
    receiver_account: "C_MULE_802",
    features: {
      step: 15, type: "TRANSFER", amount: 47000, oldbalanceOrg: 47000,
      newbalanceOrig: 0, oldbalanceDest: 125000, newbalanceDest: 172000,
    },
    neighbours: [
      { step: 15, type: "TRANSFER", amount: 38000, oldbalanceOrg: 38000, newbalanceOrig: 0, oldbalanceDest: 0, newbalanceDest: 38000 },
      { step: 15, type: "TRANSFER", amount: 45000, oldbalanceOrg: 45000, newbalanceOrig: 0, oldbalanceDest: 38000, newbalanceDest: 83000 },
      { step: 15, type: "TRANSFER", amount: 42000, oldbalanceOrg: 42000, newbalanceOrig: 0, oldbalanceDest: 83000, newbalanceDest: 125000 },
    ],
  },
};

const RESULT_STYLES = {
  ALLOW: "bg-emerald-500/10 border-emerald-500/30 text-emerald-400",
  REVIEW: "bg-amber-500/10 border-amber-500/30 text-amber-400",
  BLOCK: "bg-rose-500/10 border-rose-500/30 text-rose-400",
};

export default function ScoreForm({ onScored }) {
  const [presetKey, setPresetKey] = useState("suspicious");
  const [form, setForm] = useState(PRESETS.suspicious);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function applyPreset(key) {
    setPresetKey(key);
    setForm(PRESETS[key]);
    setResult(null);
    setError("");
  }

  // Editing any field manually moves the form out of "this exactly matches
  // a preset" state, so none of the preset buttons stay highlighted.
  function updateField(key, value) {
    setPresetKey(null);
    setForm((f) => ({ ...f, [key]: value }));
  }

  function updateAmount(value) {
    setPresetKey(null);
    setForm((f) => ({ ...f, features: { ...f.features, amount: Number(value), oldbalanceOrg: Number(value), newbalanceDest: Number(value) } }));
  }

  // Guaranteed-never-seen account ID (base36 timestamp) -- the point is to
  // demonstrate the model scoring an account it (and Redis) have literally
  // never encountered before, which is exactly what "inductive" means.
  function generateNewAccountId(field) {
    const id = `C_NEW_${Date.now().toString(36).toUpperCase()}`;
    updateField(field, id);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const payload = {
        transaction_id: `demo-${Date.now()}`,
        sender_account: form.sender_account,
        receiver_account: form.receiver_account,
        features: form.features,
        neighbours: form.neighbours,
      };
      const res = await scoreTransaction(payload);
      setResult(res);
      onScored?.();
    } catch (err) {
      setError(err.message || "Scoring failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-5">
      <h2 className="text-sm font-medium text-slate-300 mb-3">Transaction details</h2>

      <div className="flex flex-wrap gap-2 mb-4">
        {Object.entries(PRESETS).map(([key, preset]) => (
          <button
            key={key}
            type="button"
            onClick={() => applyPreset(key)}
            className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
              presetKey === key
                ? "bg-indigo-600 border-indigo-500 text-white"
                : "border-slate-700 text-slate-400 hover:border-slate-500"
            }`}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-4">
        <div>
          <label className="block text-xs text-slate-500 mb-1">Sender</label>
          <div className="flex items-center gap-1.5">
            <input
              type="text"
              value={form.sender_account}
              onChange={(e) => updateField("sender_account", e.target.value)}
              className="w-40 rounded-md bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm text-slate-300 font-mono"
            />
            <button
              type="button"
              onClick={() => generateNewAccountId("sender_account")}
              title="Insert a guaranteed-never-seen account ID — demonstrates inductive scoring on a brand-new account"
              className="text-xs px-2 py-1.5 rounded-md border border-slate-700 text-slate-400 hover:border-indigo-500 hover:text-indigo-400 transition-colors whitespace-nowrap"
            >
              New ID
            </button>
          </div>
        </div>
        <div>
          <label className="block text-xs text-slate-500 mb-1">Receiver</label>
          <div className="flex items-center gap-1.5">
            <input
              type="text"
              value={form.receiver_account}
              onChange={(e) => updateField("receiver_account", e.target.value)}
              className="w-40 rounded-md bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm text-slate-300 font-mono"
            />
            <button
              type="button"
              onClick={() => generateNewAccountId("receiver_account")}
              title="Insert a guaranteed-never-seen account ID — demonstrates inductive scoring on a brand-new account"
              className="text-xs px-2 py-1.5 rounded-md border border-slate-700 text-slate-400 hover:border-indigo-500 hover:text-indigo-400 transition-colors whitespace-nowrap"
            >
              New ID
            </button>
          </div>
        </div>
        <div>
          <label className="block text-xs text-slate-500 mb-1">Amount</label>
          <input
            type="number"
            value={form.features.amount}
            onChange={(e) => updateAmount(e.target.value)}
            className="w-32 rounded-md bg-slate-800 border border-slate-700 px-2 py-1.5 text-sm text-white"
          />
        </div>
        <div>
          <label className="block text-xs text-slate-500 mb-1">Related transactions</label>
          <div className="text-sm text-slate-300">{form.neighbours.length}</div>
        </div>
        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 text-white text-sm font-medium px-5 py-2 transition-colors"
        >
          {loading ? "Scoring…" : "Score Transaction"}
        </button>
      </form>

      {error && <p className="text-sm text-rose-400 mt-4">{error}</p>}

      {result && (
        <div className={`mt-4 rounded-lg border p-4 ${RESULT_STYLES[result.decision] || ""}`}>
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="text-lg font-semibold">
              {result.decision} {result.risk_level}
            </div>
            <div className="text-sm opacity-80">
              fraud probability: {(result.fraud_probability * 100).toFixed(1)}% · scored in {result.inference_latency_ms}ms
            </div>
          </div>
          <p className="text-sm mt-2 opacity-90">{result.explanation?.summary}</p>
          {result.limit_usage && (
            <p className="text-xs mt-2 opacity-80">
              Daily {result.limit_usage.type} limit: BDT {result.limit_usage.used.toLocaleString()} of BDT{" "}
              {result.limit_usage.limit.toLocaleString()} used · BDT {result.limit_usage.remaining.toLocaleString()} remaining
            </p>
          )}

          <div className="mt-4 pt-4 border-t border-white/10">
            <SubgraphViz
              center={form.features}
              neighbours={result.explanation?.neighbours}
              decision={result.decision}
            />
          </div>
        </div>
      )}
    </div>
  );
}
