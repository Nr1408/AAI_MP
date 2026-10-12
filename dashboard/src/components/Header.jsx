import { fmt2 } from '../lib/utils.js'

export function Header({ filtered, thresholds }) {
  const latestByHorizon = {}
  for (const row of filtered) {
    const k = row.horizon
    if (!latestByHorizon[k] || row.time_s > latestByHorizon[k].time_s) {
      latestByHorizon[k] = row
    }
  }
  const latest = latestByHorizon['60s'] || latestByHorizon['300s'] || null

  const riskColor = {
    Normal:   'text-emerald-400',
    Elevated: 'text-amber-400',
    High:     'text-red-400',
    Critical: 'text-red-300',
  }

  return (
    <header className="bg-slate-950/90 border-b border-slate-800 px-6 py-3 flex items-center justify-between flex-shrink-0">
      <div>
        <h1 className="text-lg font-bold text-white tracking-tight">
          FreeTwinEV · Thermal Forecast Dashboard
        </h1>
        <p className="text-xs text-slate-500 mt-0.5">
          P45B 1s4p Battery Module · KJS-CES-02 · Held-out test runs ID18 &amp; ID19
        </p>
      </div>
      {latest && (
        <div className="flex items-center gap-6">
          <div className="text-right">
            <p className="text-xs text-slate-500">T_cell_max</p>
            <p className="text-lg font-mono font-bold text-white">{fmt2(latest.T_cell_max)} °C</p>
          </div>
          <div className="text-right">
            <p className="text-xs text-slate-500">Risk (latest)</p>
            <p className={`text-lg font-bold ${riskColor[latest.risk_category] || 'text-slate-300'}`}>
              {latest.risk_category}
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs text-slate-500">Recommendation</p>
            <p className="text-sm font-semibold text-indigo-300">{latest.recommendation?.replace(/_/g, ' ')}</p>
          </div>
          <div className="hidden xl:block text-right">
            <p className="text-xs text-slate-500">70 °C cutoff</p>
            <p className="text-sm font-mono text-red-400 font-bold">DOCUMENTED</p>
          </div>
        </div>
      )}
    </header>
  )
}
