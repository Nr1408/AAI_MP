import {
  ComposedChart, Line, ReferenceLine, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area
} from 'recharts'
import { Card, CardTitle } from './Card.jsx'
import { ENGINEERING_CUTOFF, RISK_COLORS, RUN_COLORS } from '../lib/constants.js'
import { downsample, groupBy, fmt2 } from '../lib/utils.js'
import { TrendingUp } from 'lucide-react'

const HORIZON_DASH = { '60s': '0', '300s': '5 3' }

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-xl p-3 shadow-2xl text-xs">
      <p className="text-slate-400 mb-2 font-mono">{fmt2(label)} s</p>
      {payload.map(p => (
        <div key={p.dataKey} className="flex items-center justify-between gap-4">
          <span style={{ color: p.color }} className="truncate max-w-32">{p.name}</span>
          <span className="font-mono font-bold text-white">{fmt2(p.value)} °C</span>
        </div>
      ))}
    </div>
  )
}

export function ForecastChart({ filtered, thresholds }) {
  // Build per-run, per-horizon series — downsample to keep chart fast
  const byRunHorizon = groupBy(filtered, r => `${r.exp_id}__${r.horizon}`)

  // Merge all timestamps for the x-axis using 60s as primary
  const seriesMap = {}
  for (const [key, rows] of Object.entries(byRunHorizon)) {
    const [run, horizon] = key.split('__')
    const sampled = downsample([...rows].sort((a, b) => a.time_s - b.time_s), 400)
    for (const row of sampled) {
      const t = row.time_s
      if (!seriesMap[t]) seriesMap[t] = { time_s: t }
      seriesMap[t][`obs_${run}`] = row.T_cell_max
      seriesMap[t][`pred_${run}_${horizon}`] = row.predicted_future_temperature
    }
  }

  const chartData = Object.values(seriesMap).sort((a, b) => a.time_s - b.time_s)

  const runs = [...new Set(filtered.map(r => r.exp_id))]
  const horizons = [...new Set(filtered.map(r => r.horizon))]

  // P90 band thresholds — pick whichever horizon is active
  const firstHorizon = horizons[0] || '60s'
  const tBand = thresholds?.thresholds_by_horizon?.[firstHorizon]
  const p75 = tBand?.future_temperature_p75_c
  const p90 = tBand?.future_temperature_p90_c

  return (
    <Card className="col-span-2">
      <CardTitle icon={<TrendingUp size={14} />}>Forecast Trajectories — Observed vs Predicted</CardTitle>
      <p className="text-xs text-slate-500 mb-3">
        Solid lines = observed T_cell_max · Dashed lines = predicted future temperature · Red dashed = 70 °C engineering cutoff
      </p>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
          <XAxis
            dataKey="time_s"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={v => `${(v / 60).toFixed(0)}m`}
            tick={{ fontSize: 10, fill: '#64748b' }}
            label={{ value: 'Time (min)', position: 'insideBottomRight', offset: -4, fontSize: 10, fill: '#64748b' }}
          />
          <YAxis
            tick={{ fontSize: 10, fill: '#64748b' }}
            label={{ value: 'Temp (°C)', angle: -90, position: 'insideLeft', offset: 12, fontSize: 10, fill: '#64748b' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 11, color: '#94a3b8' }}
            formatter={v => v.replace(/_/g, ' ')}
          />

          {/* 70 °C engineering cutoff */}
          <ReferenceLine
            y={ENGINEERING_CUTOFF}
            stroke="#ef4444"
            strokeDasharray="6 3"
            strokeWidth={2}
            label={{ value: '70°C cutoff', position: 'insideTopRight', fill: '#ef4444', fontSize: 10 }}
          />

          {/* P75 / P90 band lines */}
          {p75 && <ReferenceLine y={p75} stroke="#f59e0b" strokeDasharray="3 6" strokeWidth={1}
            label={{ value: 'P75', position: 'insideTopLeft', fill: '#f59e0b', fontSize: 9 }} />}
          {p90 && <ReferenceLine y={p90} stroke="#ef4444" strokeDasharray="3 6" strokeWidth={1}
            label={{ value: 'P90', position: 'insideTopLeft', fill: '#ef4444', fontSize: 9 }} />}

          {/* Observed temp per run */}
          {runs.map(run => (
            <Line
              key={`obs_${run}`}
              dataKey={`obs_${run}`}
              name={`${run} observed`}
              stroke={RUN_COLORS[run] || '#94a3b8'}
              strokeWidth={2}
              dot={false}
              connectNulls
            />
          ))}

          {/* Predicted temp per run per horizon */}
          {runs.flatMap(run =>
            horizons.map(h => (
              <Line
                key={`pred_${run}_${h}`}
                dataKey={`pred_${run}_${h}`}
                name={`${run} pred ${h}`}
                stroke={RUN_COLORS[run] || '#94a3b8'}
                strokeWidth={1.5}
                strokeDasharray={HORIZON_DASH[h]}
                dot={false}
                connectNulls
                opacity={0.75}
              />
            ))
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </Card>
  )
}
