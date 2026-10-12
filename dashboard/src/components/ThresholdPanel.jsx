import { Card, CardTitle } from './Card.jsx'
import { ENGINEERING_CUTOFF } from '../lib/constants.js'
import { ShieldAlert } from 'lucide-react'

export function ThresholdPanel({ thresholds, recConfig }) {
  const byH = thresholds?.thresholds_by_horizon || {}

  return (
    <Card>
      <CardTitle icon={<ShieldAlert size={14} />}>Frozen Thresholds &amp; Rules</CardTitle>

      {/* Engineering cutoff */}
      <div className="flex items-center gap-3 mb-4 p-2.5 bg-red-950/30 border border-red-800/40 rounded-lg">
        <span className="text-red-400 text-xl">⚠️</span>
        <div>
          <p className="text-sm font-bold text-red-400">{ENGINEERING_CUTOFF} °C — Documented Engineering Cutoff</p>
          <p className="text-xs text-slate-500">Critical risk boundary. Not a certified safety limit.</p>
        </div>
      </div>

      {/* Percentile bands */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        {Object.entries(byH).map(([horizon, t]) => (
          <div key={horizon} className="bg-slate-800/50 rounded-lg p-3">
            <p className="text-xs font-bold text-indigo-400 mb-2">{horizon} Horizon</p>
            {[
              ['P75', t.future_temperature_p75_c, 'text-amber-400'],
              ['P90', t.future_temperature_p90_c, 'text-orange-400'],
              ['P99', t.future_temperature_p99_c, 'text-red-400'],
            ].map(([label, val, cls]) => (
              <div key={label} className="flex justify-between text-xs mb-1">
                <span className="text-slate-500">{label}</span>
                <span className={`font-mono font-bold ${cls}`}>{val?.toFixed(2)} °C</span>
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* Recommendation rules */}
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Decision-Support Rules</p>
      {recConfig?.rules && Object.entries(recConfig.rules).map(([key, rule]) => {
        const colors = {
          MAINTAIN: 'border-emerald-800/60 text-emerald-400',
          INCREASE_COOLING: 'border-blue-800/60 text-blue-400',
          HIGH_COOLING_ATTENTION: 'border-amber-800/60 text-amber-400',
          CRITICAL_PROTECTIVE_ACTION: 'border-red-800/60 text-red-400',
        }
        return (
          <div key={key} className={`border rounded-lg p-2 mb-1.5 ${colors[key] || 'border-slate-700'}`}>
            <p className={`text-xs font-bold mb-0.5 ${colors[key]?.split(' ')[1] || 'text-slate-300'}`}>
              {key.replace(/_/g, ' ')}
            </p>
            <p className="text-xs text-slate-500">{rule}</p>
          </div>
        )
      })}

      <p className="text-xs text-slate-600 mt-3">
        P75/P90/P99 = validation-derived descriptive bands (ID16 + ID22 only). Not certified safety limits.
      </p>
    </Card>
  )
}
