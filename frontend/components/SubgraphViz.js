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

const DECISION_COLORS = {
  ALLOW: "#34d399", // emerald-400
  REVIEW: "#fbbf24", // amber-400
  BLOCK: "#fb7185", // rose-400
};

export default function SubgraphViz({ center, neighbours, decision }) {
  if (!neighbours || neighbours.length === 0) {
    return (
      <div className="flex items-center justify-center h-[220px] text-sm text-slate-500 border border-dashed border-slate-800 rounded-lg">
        Scored in isolation — no related transactions found nearby.
      </div>
    );
  }

  const centerColor = DECISION_COLORS[decision] || "#94a3b8";
  const n = neighbours.length;

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
            key={`edge-${neighbours[i].id}`}
            x1={CENTER}
            y1={CENTER}
            x2={p.x}
            y2={p.y}
            stroke="#475569"
            strokeWidth="1.5"
          />
        ))}

        {/* neighbour nodes */}
        {positions.map((p, i) => (
          <g key={neighbours[i].id}>
            <circle cx={p.x} cy={p.y} r={NEIGHBOUR_NODE_R} fill="#1e293b" stroke="#64748b" strokeWidth="1.5" />
            <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize="9" fill="#cbd5e1">
              {neighbours[i].type === "CASH_OUT" ? "CO" : "TR"}
            </text>
            <text x={p.x} y={p.y + NEIGHBOUR_NODE_R + 13} textAnchor="middle" fontSize="9" fill="#64748b">
              {Number(neighbours[i].amount).toLocaleString()}
            </text>
          </g>
        ))}

        {/* center node (the transaction just scored) drawn last, on top.
            Same "type code inside the circle" style as neighbour nodes for
            consistency; the decision label goes below, like the amount
            label on neighbour nodes, since two lines never fit inside a
            circle this size. */}
        <circle cx={CENTER} cy={CENTER} r={CENTER_NODE_R} fill={centerColor} fillOpacity="0.15" stroke={centerColor} strokeWidth="2.5" />
        <text x={CENTER} y={CENTER + 4} textAnchor="middle" fontSize="12" fontWeight="600" fill={centerColor}>
          {center?.type === "CASH_OUT" ? "CO" : "TR"}
        </text>
        <text x={CENTER} y={CENTER + CENTER_NODE_R + 16} textAnchor="middle" fontSize="11" fontWeight="600" fill={centerColor}>
          {decision}
        </text>
      </svg>
      <p className="text-center text-xs text-slate-500 mt-1">
        the actual subgraph scored — {n} related transaction{n === 1 ? "" : "s"} connected to this one
      </p>
    </div>
  );
}
