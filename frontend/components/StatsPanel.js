function StatCard({ label, value, accent }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`text-2xl font-semibold mt-1 ${accent || "text-white"}`}>{value}</div>
    </div>
  );
}

export default function StatsPanel({ stats }) {
  if (!stats) {
    return <div className="text-slate-500 text-sm">Loading stats…</div>;
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
      <StatCard label="Total Scored" value={stats.total_scored} />
      <StatCard label="Allowed" value={stats.allow_count} accent="text-emerald-400" />
      <StatCard label="Review" value={stats.review_count} accent="text-amber-400" />
      <StatCard label="Blocked" value={stats.block_count} accent="text-rose-400" />
      <StatCard
        label="Avg Latency"
        value={stats.avg_inference_latency_ms != null ? `${Math.round(stats.avg_inference_latency_ms)} ms` : "—"}
      />
    </div>
  );
}
