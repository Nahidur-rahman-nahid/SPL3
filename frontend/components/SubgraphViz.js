"use client";

// Simple radial node-link diagram — deliberately NOT a D3 force simulation.
// This renders the EXACT star-graph structure that was fed to RealTimeNet
// for scoring (center = the transaction just scored, spokes = the
// neighbours found via Redis/manual input), so it's a faithful picture of
// what the model actually saw, not an illustration.

const SIZE = 320;
const CENTER = SIZE / 2;
const NEIGHBOUR_RADIUS = 110;
const CENTER_NODE_R = 28;
const NEIGHBOUR_NODE_R = 16;

// CSS variables (not hex), so the graph follows the light / dark theme.
// SVG attributes don't accept var(), hence the style={{...}} below.
const DECISION_COLORS = {
  ALLOW: "var(--color-emerald-400)",
  REVIEW: "var(--color-amber-400)",
  BLOCK: "var(--color-rose-400)",
};

// Past this many neighbours the per-node labels would overlap: nodes shrink
// to fit the ring and the figures move to the hover tooltip only.
const MAX_LABELLED = 12;

export default function SubgraphViz({ center, neighbours, decision }) {
  if (!neighbours || neighbours.length === 0) {
    return (
      <div className="flex items-center justify-center h-[220px] text-sm text-slate-500 border border-dashed border-slate-800 rounded-lg">
        Scored in isolation — no related transactions found nearby.
      </div>
    );
  }

  const centerColor = DECISION_COLORS[decision] || "var(--color-slate-400)";
  const n = neighbours.length;
  const labelled = n <= MAX_LABELLED;
  // Space each node gets on the ring, so crowded rings don't overlap.
  const nodeR = Math.max(5, Math.min(NEIGHBOUR_NODE_R, (Math.PI * NEIGHBOUR_RADIUS) / n - 2));
  const priorShown = neighbours.some((nb) => nb.prior_decision);

  const positions = neighbours.map((_, i) => {
    const angle = (2 * Math.PI * i) / n - Math.PI / 2; // start at top, go clockwise
    return {
      x: CENTER + NEIGHBOUR_RADIUS * Math.cos(angle),
      y: CENTER + NEIGHBOUR_RADIUS * Math.sin(angle),
    };
  });

  return (
    <div>
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full max-w-[320px] mx-auto">
        {/* edges first, so nodes draw on top */}
        {positions.map((p, i) => (
          <line
            key={`edge-${neighbours[i].transaction_id}`}
            x1={CENTER}
            y1={CENTER}
            x2={p.x}
            y2={p.y}
            style={{ stroke: "var(--color-slate-600)" }}
            strokeWidth="1.5"
          />
        ))}

        {/* neighbour nodes — native SVG <title> gives a free browser tooltip
            on hover with the exact figures, so labels don't need to stay
            permanently visible (and legible) at any node count. */}
        {positions.map((p, i) => (
          <g key={neighbours[i].transaction_id} className="cursor-default">
            <title>
              {`${neighbours[i].edge_type ? neighbours[i].edge_type + " · " : ""}${neighbours[i].type} · amount ${Number(neighbours[i].amount).toLocaleString()}${neighbours[i].step != null ? ` · step ${neighbours[i].step}` : ""}${neighbours[i].prior_decision ? ` · was ${neighbours[i].prior_decision}` : ""}`}
            </title>
            {/* outline = the decision this neighbour itself got when it was scored */}
            <circle cx={p.x} cy={p.y} r={nodeR} strokeWidth="1.5"
              style={{ fill: "var(--color-slate-800)", stroke: DECISION_COLORS[neighbours[i].prior_decision] || "var(--color-slate-500)" }} />
            {nodeR >= 12 && (
              <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize="9" style={{ fill: "var(--color-slate-300)" }}>
                {neighbours[i].type === "CASH_OUT" ? "CO" : "TR"}
              </text>
            )}
            {labelled && (
              <text x={p.x} y={p.y + nodeR + 13} textAnchor="middle" fontSize="9" style={{ fill: "var(--color-slate-500)" }}>
                {Number(neighbours[i].amount).toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </text>
            )}
          </g>
        ))}

        {/* center node (the transaction just scored) drawn last, on top.
            Same "type code inside the circle" style as neighbour nodes for
            consistency; the decision label goes below, like the amount
            label on neighbour nodes, since two lines never fit inside a
            circle this size. */}
        <g className="cursor-default">
          <title>
            {`${center?.type || "transaction"} · amount ${center?.amount != null ? Number(center.amount).toLocaleString() : "?"} · ${decision}`}
          </title>
          <circle cx={CENTER} cy={CENTER} r={CENTER_NODE_R} fillOpacity="0.15" strokeWidth="2.5" style={{ fill: centerColor, stroke: centerColor }} />
          <text x={CENTER} y={CENTER + 4} textAnchor="middle" fontSize="12" fontWeight="600" style={{ fill: centerColor }}>
            {center?.type === "CASH_OUT" ? "CO" : "TR"}
          </text>
          <text x={CENTER} y={CENTER + CENTER_NODE_R + 16} textAnchor="middle" fontSize="11" fontWeight="600" style={{ fill: centerColor }}>
            {decision}
          </text>
        </g>
      </svg>
      <p className="text-center text-xs text-slate-500 mt-1">
        the actual subgraph scored — {n} related transaction{n === 1 ? "" : "s"} connected to this one · hover a node for exact figures
      </p>
      {priorShown && (
        <div className="text-center text-[11px] text-slate-500 mt-2">
          <p>node outline = that transaction&apos;s own earlier decision</p>
          <p className="mt-1 flex flex-wrap items-center justify-center gap-x-3 gap-y-1">
            {Object.entries(DECISION_COLORS).map(([label, color]) => (
              <span key={label} className="inline-flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full border-2" style={{ borderColor: color }} />
                {label}
              </span>
            ))}
          </p>
        </div>
      )}
    </div>
  );
}
