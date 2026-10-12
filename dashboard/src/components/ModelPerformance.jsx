import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell
} from 'recharts'
import { Card, CardTitle } from './Card.jsx'
import { fmt4, fmt2 } from '../lib/utils.js'
import { BarChart2, Target } from 'lucide-react'

const METRIC_COLORS = {
  '60s':  '#818cf8',
  '300s': '#34d399',
}

export function ModelMetricsTable({ modelPerf, valMetrics }) {
  const valByH = {}
  for (const v of (valMetrics || [])) valByH[v.horizon] = v

  return (
    <Card className="col-span-2">
      <CardTitle icon={<Target size={14} />}>Frozen Model Performance — Validation vs Held-out Test</CardTitle>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-500 border-b border-slate-800">
              {['Horizon', 'Model', 'Features', 'Val MAE', 'Val RMSE', 'Val R²', 'Test MAE', 'Test RMSE', 'Test R²', 'N', 'Abs R²'].map(h => (
                <th key={h} className="text-left py-2 pr-4 font-medium whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(modelPerf || []).map((r, i) => {
              const v = valByH[r.horizon] || {}
              return (
                <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                  <td className="py-2 pr-4 font-mono font-bold" style={{ color: METRIC_COLORS[r.horizon] }}>{r.horizon}</td>
                  <td className="py-2 pr-4 text-slate-300">{r.model}</td>
                  <td className="py-2 pr-4 text-slate-400">{r.feature_set}</td>
                  <td className="py-2 pr-4 font-mono text-slate-400">{v.val_mae?.toFixed(4) ?? '—'}</td>
                  <td className="py-2 pr-4 font-mono text-slate-400">{v.val_rmse?.toFixed(4) ?? '—'}</td>
                  <td className="py-2 pr-4 font-mono text-slate-400">{v.val_r2?.toFixed(4) ?? '—'}</td>
                  <td className="py-2 pr-4 font-mono font-bold text-indigo-300">{fmt4(r.mae)}</td>
                  <td className="py-2 pr-4 font-mono font-bold text-indigo-300">{fmt4(r.rmse)}</td>
                  <td className="py-2 pr-4 font-mono font-bold text-indigo-300">{fmt4(r.r2)}</td>
                  <td className="py-2 pr-4 font-mono text-slate-400">{r.n?.toLocaleString()}</td>
                  <td className="py-2 pr-4 font-mono text-emerald-400">{fmt4(r.abs_r2)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-slate-600 mt-3">
        Validation = ID16 + ID22 (development). Test = ID18 + ID19 (held-out, opened once after model freeze).
        Abs R² measures absolute temperature tracking (near 1.0 expected as target is ΔT + current temp).
      </p>
    </Card>
  )
}

export function PerRunTable({ modelPerfPerRun }) {
  return (
    <Card className="col-span-2">
      <CardTitle icon={<BarChart2 size={14} />}>Per-Run Held-out Test Results</CardTitle>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-500 border-b border-slate-800">
              {['Horizon', 'Run', 'N', 'MAE (°C)', 'RMSE (°C)', 'R²', 'Bias (°C)', 'Max AE (°C)'].map(h => (
                <th key={h} className="text-left py-2 pr-4 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(modelPerfPerRun || []).map((r, i) => (
              <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                <td className="py-2 pr-4 font-mono font-bold" style={{ color: METRIC_COLORS[r.horizon] }}>{r.horizon}</td>
                <td className="py-2 pr-4 font-mono text-indigo-300">{r.run}</td>
                <td className="py-2 pr-4 font-mono text-slate-400">{r.n?.toLocaleString()}</td>
                <td className="py-2 pr-4 font-mono text-white">{fmt4(r.mae)}</td>
                <td className="py-2 pr-4 font-mono text-white">{fmt4(r.rmse)}</td>
                <td className="py-2 pr-4 font-mono text-white">{fmt4(r.r2)}</td>
                <td className={`py-2 pr-4 font-mono ${Math.abs(r.bias) > 0.5 ? 'text-amber-400' : 'text-slate-300'}`}>{fmt4(r.bias)}</td>
                <td className="py-2 pr-4 font-mono text-red-400">{fmt2(r.max_ae)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

export function MAEComparisonChart({ modelPerf, valMetrics }) {
  const valByH = {}
  for (const v of (valMetrics || [])) valByH[v.horizon] = v

  const data = (modelPerf || []).map(r => ({
    horizon: r.horizon,
    'Val MAE': valByH[r.horizon]?.val_mae,
    'Test MAE': r.mae,
    'Test RMSE': r.rmse,
  }))

  return (
    <Card>
      <CardTitle icon={<BarChart2 size={14} />}>MAE: Validation vs Test</CardTitle>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 16, left: -12, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
          <XAxis dataKey="horizon" tick={{ fontSize: 11, fill: '#64748b' }} />
          <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            formatter={v => [`${v?.toFixed(4)} °C`]}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="Val MAE" fill="#64748b" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Test MAE" fill="#818cf8" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Test RMSE" fill="#f87171" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </Card>
  )
}
