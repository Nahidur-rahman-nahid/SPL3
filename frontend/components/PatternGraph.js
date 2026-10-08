"use client";

// Account-level picture of a fraud pattern demo run, built up one
// transaction at a time: every account is a dot, every transaction an arrow
// from sender to receiver, coloured by the decision it got. Senders sit on
// the left, receivers on the right, so a fraud ring shows up as many arrows
// converging on (or spreading from) one account, while the look-alike stays
// a set of separate, unconnected pairs.

// CSS variables so the graph follows the light / dark theme (applied via
// style, since SVG attributes don't accept var()).
const COLORS = { ALLOW: "var(--color-emerald-400)", REVIEW: "var(--color-amber-400)", BLOCK: "var(--color-rose-400)" };
const SEVERITY = { ALLOW: 0, REVIEW: 1, BLOCK: 2 };
const NEUTRAL = "var(--color-slate-500)";

const WIDTH = 380;
const LEFT_X = 120;
const RIGHT_X = 260;
const ROW = 26;
const PAD = 22;
const NODE_R = 8;

// "C_DEMO_3F9A21_VICTIM4" -> "Victim 4"
export function accountLabel(account) {
  const raw = account.replace(/^C_DEMO_[0-9A-F]+_/, "");
  const rules = [
    [/^ALONE(\d+)_FROM$/, "Sender $1"],
    [/^ALONE(\d+)_TO$/, "Receiver $1"],
    [/^VICTIM(\d+)$/, "Victim $1"],
    [/^MULE(\d+)$/, "Mule $1"],
    [/^NEW(\d+)$/, "New account $1"],
    [/^CUST(\d+)$/, "Customer $1"],
    [/^TENANT(\d+)$/, "Tenant $1"],
    [/^PAYEE(\d+)$/, "Payee $1"],
  ];
  for (const [pattern, replacement] of rules) {
    if (pattern.test(raw)) return raw.replace(pattern, replacement);
  }
  return raw.charAt(0) + raw.slice(1).toLowerCase();
}

// Positions come from ALL steps of the run, so nothing moves as more
// transactions are revealed; only what has been revealed is drawn.
function layout(allSteps) {
  const side = new Map();
  const left = [];
  const right = [];
  for (const s of allSteps) {
    if (!side.has(s.sender)) {
      side.set(s.sender, "L");
      left.push(s.sender);
    }
  }
  for (const s of allSteps) {
    if (!side.has(s.receiver)) {
      side.set(s.receiver, "R");
      right.push(s.receiver);
    }
  }
  const rows = Math.max(left.length, right.length, 1);
  const height = PAD * 2 + (rows - 1) * ROW;
  const pos = new Map();
  const place = (column, x) => {
    const offset = ((rows - column.length) * ROW) / 2;
    column.forEach((account, i) => pos.set(account, { x, y: PAD + offset + i * ROW, side: x === LEFT_X ? "L" : "R" }));
  };
  place(left, LEFT_X);
  place(right, RIGHT_X);
  return { pos, height };
}

function edgePath(from, to, offset) {
  // quadratic curve, bent sideways by `offset` so repeated transfers between
  // the same two accounts stay visible as separate arrows
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len;
  const ny = dx / len;
  const cx = (from.x + to.x) / 2 + nx * offset;
  const cy = (from.y + to.y) / 2 + ny * offset;
  // stop short of the target circle so the arrowhead stays visible
  const tx = cx - to.x;
  const ty = cy - to.y;
  const tl = Math.hypot(tx, ty) || 1;
  const ex = to.x + (tx / tl) * (NODE_R + 2);
  const ey = to.y + (ty / tl) * (NODE_R + 2);
  return `M ${from.x} ${from.y} Q ${cx} ${cy} ${ex} ${ey}`;
}

export default function PatternGraph({ allSteps, shown }) {
  if (!allSteps?.length) {
    return (
      <div className="h-36 flex items-center justify-center rounded-lg border border-dashed border-slate-800 text-xs text-slate-600">
        Press Run: the account graph builds here as each transaction goes through the live system.
      </div>
    );
  }

  const { pos, height } = layout(allSteps);
  const visible = allSteps.slice(0, shown);

  // Repeated sender→receiver pairs (structuring) get fanned-out curves.
  const pairTotals = new Map();
  allSteps.forEach((s) => {
    const key = `${s.sender}>${s.receiver}`;
    pairTotals.set(key, (pairTotals.get(key) || 0) + 1);
  });
  const pairSeen = new Map();

  // A dot takes the worst decision among the transactions it took part in.
  const nodeSeverity = new Map();
  visible.forEach((s) => {
    for (const account of [s.sender, s.receiver]) {
      const prev = nodeSeverity.get(account);
      if (prev === undefined || SEVERITY[s.decision] > SEVERITY[prev]) nodeSeverity.set(account, s.decision);
    }
  });

  return (
    <div>
      <svg viewBox={`0 0 ${WIDTH} ${height}`} className="w-full" role="img" aria-label="Account graph of this run">
        <defs>
          {Object.entries(COLORS).map(([decision, color]) => (
            <marker key={decision} id={`arrow-${decision}`} viewBox="0 0 10 10" refX="8" refY="5"
              markerWidth="5" markerHeight="5" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" style={{ fill: color }} />
            </marker>
          ))}
        </defs>

        {visible.map((s, i) => {
          const key = `${s.sender}>${s.receiver}`;
          const k = pairSeen.get(key) || 0;
          pairSeen.set(key, k + 1);
          const total = pairTotals.get(key);
          const offset = (k - (total - 1) / 2) * 18;
          const latest = i === visible.length - 1;
          return (
            <path
              key={s.transaction_id}
              d={edgePath(pos.get(s.sender), pos.get(s.receiver), offset)}
              fill="none"
              style={{ stroke: COLORS[s.decision] || NEUTRAL }}
              strokeWidth={latest ? 2.6 : 1.4}
              strokeOpacity={latest ? 1 : 0.75}
              markerEnd={`url(#arrow-${s.decision})`}
              className={latest ? "animate-pulse" : ""}
            >
              <title>
                {`${s.index}. ${accountLabel(s.sender)} → ${accountLabel(s.receiver)} · ${s.type} BDT ${Number(s.amount).toLocaleString()} · ${(s.probability * 100).toFixed(1)}% ${s.decision}`}
              </title>
            </path>
          );
        })}

        {[...nodeSeverity.keys()].map((account) => {
          const p = pos.get(account);
          const color = COLORS[nodeSeverity.get(account)] || NEUTRAL;
          return (
            <g key={account}>
              <title>{accountLabel(account)}</title>
              <circle cx={p.x} cy={p.y} r={NODE_R} strokeWidth="2" style={{ fill: "var(--color-slate-900)", stroke: color }} />
              <text
                x={p.side === "L" ? p.x - NODE_R - 5 : p.x + NODE_R + 5}
                y={p.y + 3.5}
                textAnchor={p.side === "L" ? "end" : "start"}
                fontSize="10"
                style={{ fill: "var(--color-slate-300)" }}
              >
                {accountLabel(account)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="mt-1 flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-[11px] text-slate-500">
        <span>dot = account · arrow = transaction, coloured by its decision:</span>
        {Object.entries(COLORS).map(([decision, color]) => (
          <span key={decision} className="inline-flex items-center gap-1">
            <span className="w-3 h-0.5 rounded" style={{ background: color }} />
            {decision}
          </span>
        ))}
      </div>
    </div>
  );
}
