import { clsx } from '../lib/utils.js'

export function KpiCard({ label, value, unit = '', sub, accent = 'indigo', warning = false, critical = false }) {
  const color = critical
    ? 'text-red-400'
    : warning
    ? 'text-amber-400'
    : accent === 'green'
    ? 'text-emerald-400'
    : accent === 'blue'
    ? 'text-blue-400'
    : 'text-indigo-400'

  return (
    <div className={clsx(
      'bg-slate-900/80 border rounded-xl p-4 flex flex-col gap-1',
      critical ? 'border-red-500/50 bg-red-950/20' :
      warning  ? 'border-amber-500/40' :
                 'border-slate-700/60'
    )}>
      <p className="text-xs text-slate-500 uppercase tracking-wider truncate">{label}</p>
      <p className={`text-2xl font-bold font-mono ${color}`}>
        {value}<span className="text-sm font-normal text-slate-400 ml-1">{unit}</span>
      </p>
      {sub && <p className="text-xs text-slate-500">{sub}</p>}
    </div>
  )
}
