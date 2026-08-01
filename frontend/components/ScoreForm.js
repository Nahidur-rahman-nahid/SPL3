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
  suspicious: {
    label: "Suspicious Transfer (drain-to-mule)",
    sender_account: "C_VICTIM_301",
    receiver_account: "C_MULE_402",
    features: {
      step: 10, type: "TRANSFER", amount: 181000, oldbalanceOrg: 181000,
      newbalanceOrig: 0, oldbalanceDest: 0, newbalanceDest: 181000,
    },
    neighbours: [
      { step: 10, type: "CASH_OUT", amount: 90000, oldbalanceOrg: 90000, newbalanceOrig: 0, oldbalanceDest: 5000, newbalanceDest: 95000 },
    ],
  },
  highrisk: {
    label: "High-Risk (agent collusion pattern)",
    sender_account: "C_VICTIM_501",
    receiver_account: "C_MULE_602",
    features: {
      step: 15, type: "TRANSFER", amount: 420000, oldbalanceOrg: 420000,
      newbalanceOrig: 0, oldbalanceDest: 0, newbalanceDest: 420000,
    },
    neighbours: [
      { step: 15, type: "CASH_OUT", amount: 200000, oldbalanceOrg: 200000, newbalanceOrig: 0, oldbalanceDest: 0, newbalanceDest: 200000 },
      { step: 14, type: "TRANSFER", amount: 150000, oldbalanceOrg: 150000, newbalanceOrig: 0, oldbalanceDest: 0, newbalanceDest: 150000 },
      { step: 15, type: "CASH_OUT", amount: 300000, oldbalanceOrg: 300000, newbalanceOrig: 0, oldbalanceDest: 0, newbalanceDest: 300000 },
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
      <h2 className="text-sm font-medium text-slate-300 mb-3">Score a Transaction</h2>

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
