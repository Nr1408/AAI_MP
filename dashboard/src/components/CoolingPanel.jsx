import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line, Legend
} from 'recharts'
import { Card, CardTitle } from './Card.jsx'
import { REC_COLORS, REC_LABELS } from '../lib/constants.js'
import { downsample } from '../lib/utils.js'
import { Droplets, Zap } from 'lucide-react'

export function RecommendationBar({ recMetrics, filters }) {
  const visible = recMetrics.filter(
    r => filters.runs.includes(r.run) && filters.horizons.includes(r.horizon)
  )
  const chartData = visible.map(r => ({
    name: `${r.run} ${r.horizon}`,
    MAINTAIN: r.maintain,
    INCREASE_COOLING: r.increase_cooling,
    HIGH_COOLING_ATTENTION: r.high_attention,
    CRITICAL_PROTECTIVE_ACTION: r.critical_action,
  }))

  return (
    <Card>
      <CardTitle icon={<Droplets size={14} />}>Recommendation Distribution</CardTitle>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
          <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }} />
          <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            labelStyle={{ color: '#94a3b8' }}
          />
          <Legend
            wrapperStyle={{ fontSize: 10 }}
            formatter={v => REC_LABELS[v] || v}
          />
          {Object.entries(REC_COLORS).map(([key, color]) => (
            <Bar key={key} dataKey={key} stackId="a" fill={color} name={REC_LABELS[key]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </Card>
  )
}

export function FlowTemperatureChart({ filtered }) {
  const sampled = downsample(
    [...filtered].sort((a, b) => a.time_s - b.time_s),
    500
  )

  return (
    <Card className="col-span-2">
      <CardTitle icon={<Droplets size={14} />}>Cooling Conditions Over Time</CardTitle>
      <p className="text-xs text-slate-500 mb-3">Liquid flow (L/min) and inlet temperature — observational context for recommendations</p>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={sampled} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
          <XAxis
            dataKey="time_s"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={v => `${(v / 60).toFixed(0)}m`}
            tick={{ fontSize: 10, fill: '#64748b' }}
          />
          <YAxis yAxisId="flow" tick={{ fontSize: 10, fill: '#64748b' }}
            label={{ value: 'Flow (L/min)', angle: -90, position: 'insideLeft', offset: 14, fontSize: 9, fill: '#64748b' }} />
          <YAxis yAxisId="temp" orientation="right" tick={{ fontSize: 10, fill: '#64748b' }}
            label={{ value: 'Inlet °C', angle: 90, position: 'insideRight', offset: 8, fontSize: 9, fill: '#64748b' }} />
          <Tooltip
            labelFormatter={v => `${(v / 60).toFixed(1)} min`}
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            labelStyle={{ color: '#94a3b8' }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line yAxisId="flow" dataKey="liquid_flow" name="Liquid flow (L/min)"
            stroke="#3b82f6" strokeWidth={1.5} dot={false} connectNulls />
          <Line yAxisId="temp" dataKey="temp_inlet_C" name="Inlet temp (°C)"
            stroke="#f97316" strokeWidth={1.5} dot={false} connectNulls />
          <Line yAxisId="flow" dataKey="current_abs" name="Current |A|"
            stroke="#a78bfa" strokeWidth={1} dot={false} connectNulls strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  )
}

const REC_BADGE = {
  MAINTAIN:                  'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
  INCREASE_COOLING:          'bg-blue-500/20 text-blue-400 border-blue-500/40',
  HIGH_COOLING_ATTENTION:    'bg-amber-500/20 text-amber-400 border-amber-500/40',
  CRITICAL_PROTECTIVE_ACTION:'bg-red-700/30 text-red-300 border-red-600/60',
}

export function RecommendationTable({ filtered }) {
  // Show last 50 rows most recent first
  const rows = [...filtered]
    .sort((a, b) => b.time_s - a.time_s)
    .slice(0, 50)

  return (
    <Card className="col-span-2">
      <CardTitle icon={<Zap size={14} />}>Recent Recommendations (latest 50 rows)</CardTitle>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-500 border-b border-slate-800">
              {['Run', 'Time (min)', 'Horizon', 'Recommendation', 'Risk', 'Flow (L/min)', 'Current |A|', 'Inlet °C'].map(h => (
                <th key={h} className="text-left py-2 pr-4 font-medium whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                <td className="py-1.5 pr-4 font-mono text-indigo-300">{r.exp_id}</td>
                <td className="py-1.5 pr-4 font-mono">{(r.time_s / 60).toFixed(1)}</td>
                <td className="py-1.5 pr-4">{r.horizon}</td>
                <td className="py-1.5 pr-4">
                  <span className={`text-xs px-2 py-0.5 rounded-full border ${REC_BADGE[r.recommendation] || 'bg-slate-700 text-slate-300 border-slate-600'}`}>
                    {REC_LABELS[r.recommendation] || r.recommendation}
                  </span>
                </td>
                <td className="py-1.5 pr-4">
                  <span className="text-xs font-semibold" style={{ color: { Normal: '#22c55e', Elevated: '#f59e0b', High: '#ef4444', Critical: '#dc2626' }[r.risk_category] || '#94a3b8' }}>
                    {r.risk_category}
                  </span>
                </td>
                <td className="py-1.5 pr-4 font-mono">{r.liquid_flow?.toFixed(3) ?? '—'}</td>
                <td className="py-1.5 pr-4 font-mono">{r.current_abs?.toFixed(1) ?? '—'}</td>
                <td className="py-1.5 pr-4 font-mono">{r.temp_inlet_C?.toFixed(1) ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
