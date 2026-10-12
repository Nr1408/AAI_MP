import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, PieChart, Pie, Legend
} from 'recharts'
import { Card, CardTitle } from './Card.jsx'
import { RISK_COLORS } from '../lib/constants.js'
import { groupBy } from '../lib/utils.js'
import { Activity, PieChart as PieIcon } from 'lucide-react'

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-xl p-3 shadow-2xl text-xs">
      <p className="text-slate-400 mb-1">{label}</p>
      {payload.map(p => (
        <div key={p.name} className="flex justify-between gap-4">
          <span style={{ color: p.fill }}>{p.name}</span>
          <span className="font-mono font-bold text-white">{p.value}</span>
        </div>
      ))}
    </div>
  )
}

export function RiskDistributionBar({ riskMetrics, filters }) {
  const visible = riskMetrics.filter(
    r => filters.runs.includes(r.run) && filters.horizons.includes(r.horizon)
  )

  const chartData = visible.map(r => ({
    name: `${r.run} ${r.horizon}`,
    Normal: filters.risks.includes('Normal') ? r.normal : 0,
    Elevated: filters.risks.includes('Elevated') ? r.elevated : 0,
    High: filters.risks.includes('High') ? r.high : 0,
    Critical: filters.risks.includes('Critical') ? r.critical : 0,
  }))

  return (
    <Card>
      <CardTitle icon={<Activity size={14} />}>Risk Distribution</CardTitle>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
          <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }} />
          <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {Object.entries(RISK_COLORS).map(([risk, color]) => (
            <Bar key={risk} dataKey={risk} stackId="a" fill={color} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </Card>
  )
}

export function RiskTimeline({ filtered }) {
  // Build time-bucketed risk counts (bucket by 60s windows)
  const BUCKET = 60
  const buckets = {}
  for (const row of filtered) {
    const b = Math.floor(row.time_s / BUCKET) * BUCKET
    if (!buckets[b]) buckets[b] = { time_s: b, Normal: 0, Elevated: 0, High: 0, Critical: 0 }
    if (buckets[b][row.risk_category] !== undefined) buckets[b][row.risk_category]++
  }
  const data = Object.values(buckets).sort((a, b) => a.time_s - b.time_s)

  return (
    <Card>
      <CardTitle icon={<Activity size={14} />}>Risk Timeline (60 s buckets)</CardTitle>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
          <XAxis
            dataKey="time_s"
            tickFormatter={v => `${(v / 60).toFixed(0)}m`}
            tick={{ fontSize: 10, fill: '#64748b' }}
          />
          <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
          <Tooltip
            labelFormatter={v => `~${(v / 60).toFixed(0)} min`}
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            labelStyle={{ color: '#94a3b8' }}
            itemStyle={{ color: '#e2e8f0' }}
          />
          {Object.entries(RISK_COLORS).map(([risk, color]) => (
            <Bar key={risk} dataKey={risk} stackId="a" fill={color} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </Card>
  )
}

export function RiskPie({ filtered }) {
  const counts = {}
  for (const row of filtered) {
    counts[row.risk_category] = (counts[row.risk_category] || 0) + 1
  }
  const data = Object.entries(counts).map(([name, value]) => ({ name, value }))

  return (
    <Card>
      <CardTitle icon={<PieIcon size={14} />}>Risk Breakdown</CardTitle>
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={55}
            outerRadius={85}
            paddingAngle={3}
            dataKey="value"
            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            labelLine={{ stroke: '#475569' }}
          >
            {data.map(entry => (
              <Cell key={entry.name} fill={RISK_COLORS[entry.name] || '#64748b'} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            formatter={(v, n) => [v + ' rows', n]}
          />
        </PieChart>
      </ResponsiveContainer>
    </Card>
  )
}
