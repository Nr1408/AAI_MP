import { RUNS, HORIZONS, RISK_COLORS, REC_LABELS, REC_COLORS } from '../lib/constants.js'
import { clsx } from '../lib/utils.js'

function Toggle({ label, active, color, onClick }) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'flex items-center gap-2 w-full text-left px-3 py-2 rounded-lg text-sm transition-all',
        active
          ? 'bg-slate-700/80 text-white'
          : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-300'
      )}
    >
      <span
        className="w-2.5 h-2.5 rounded-full flex-shrink-0"
        style={{ background: active ? color : '#475569' }}
      />
      {label}
    </button>
  )
}

function Section({ title, children }) {
  return (
    <div className="mb-6">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-2 px-1">{title}</p>
      <div className="flex flex-col gap-1">{children}</div>
    </div>
  )
}

export function Sidebar({ filters, setFilters }) {
  const toggle = (key, val) =>
    setFilters(f => ({
      ...f,
      [key]: f[key].includes(val) ? f[key].filter(x => x !== val) : [...f[key], val],
    }))

  return (
    <aside className="w-56 flex-shrink-0 bg-slate-950/80 border-r border-slate-800 p-4 flex flex-col h-full overflow-y-auto">
      {/* Logo */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-2xl">🔋</span>
          <div>
            <p className="text-sm font-bold text-white leading-tight">FreeTwinEV</p>
            <p className="text-xs text-slate-500">Thermal Dashboard</p>
          </div>
        </div>
        <div className="mt-2 px-2 py-1.5 bg-indigo-500/10 border border-indigo-500/20 rounded-lg">
          <p className="text-xs text-indigo-400">Phase 11 · Read-only</p>
        </div>
      </div>

      <Section title="Run">
        {RUNS.map(r => (
          <Toggle
            key={r}
            label={r}
            active={filters.runs.includes(r)}
            color={r === 'ID18' ? '#818cf8' : '#34d399'}
            onClick={() => toggle('runs', r)}
          />
        ))}
      </Section>

      <Section title="Horizon">
        {HORIZONS.map(h => (
          <Toggle
            key={h}
            label={h === '60s' ? '60 s  — Operational' : '300 s — Strategic'}
            active={filters.horizons.includes(h)}
            color={h === '60s' ? '#818cf8' : '#34d399'}
            onClick={() => toggle('horizons', h)}
          />
        ))}
      </Section>

      <Section title="Risk">
        {Object.entries(RISK_COLORS).map(([r, c]) => (
          <Toggle key={r} label={r} active={filters.risks.includes(r)} color={c}
            onClick={() => toggle('risks', r)} />
        ))}
      </Section>

      <Section title="Recommendation">
        {Object.entries(REC_LABELS).map(([k, label]) => (
          <Toggle key={k} label={label} active={filters.recs.includes(k)}
            color={REC_COLORS[k]} onClick={() => toggle('recs', k)} />
        ))}
      </Section>

      <div className="mt-auto pt-4 border-t border-slate-800">
        <p className="text-xs text-slate-600 leading-relaxed">
          Frozen Phase 6–9 outputs.<br />
          No model training performed.<br />
          70 °C = documented cutoff.
        </p>
      </div>
    </aside>
  )
}
