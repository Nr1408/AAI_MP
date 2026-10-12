import { KpiCard } from './KpiCard.jsx'
import { fmt2, fmt4 } from '../lib/utils.js'

export function KpiRow({ filtered, modelPerf }) {
  if (!filtered.length) return null

  // Latest state
  const latest60 = filtered.filter(r => r.horizon === '60s').sort((a, b) => b.time_s - a.time_s)[0]
  const latest300 = filtered.filter(r => r.horizon === '300s').sort((a, b) => b.time_s - a.time_s)[0]
  const latest = latest60 || latest300

  // Aggregate counts
  const criticalCount = filtered.filter(r => r.risk_category === 'Critical').length
  const totalRows = filtered.length

  // Model metrics
  const m60 = (modelPerf || []).find(r => r.horizon === '60s')
  const m300 = (modelPerf || []).find(r => r.horizon === '300s')

  return (
    <div className="grid grid-cols-4 gap-3 lg:grid-cols-8 mb-4">
      <KpiCard
        label="T_cell_max (latest)"
        value={fmt2(latest?.T_cell_max)}
        unit="°C"
        accent="blue"
        warning={latest?.T_cell_max > 40}
        critical={latest?.T_cell_max > 70}
      />
      <KpiCard
        label="Predicted (60s)"
        value={latest60 ? fmt2(latest60.predicted_future_temperature) : '—'}
        unit="°C"
        sub="60s horizon"
        accent="indigo"
      />
      <KpiCard
        label="Predicted (300s)"
        value={latest300 ? fmt2(latest300.predicted_future_temperature) : '—'}
        unit="°C"
        sub="300s horizon"
        accent="indigo"
      />
      <KpiCard
        label="Risk (latest)"
        value={latest?.risk_category || '—'}
        sub={latest?.exp_id}
        critical={latest?.risk_category === 'Critical'}
        warning={latest?.risk_category === 'High' || latest?.risk_category === 'Elevated'}
      />
      <KpiCard
        label="Critical rows"
        value={criticalCount}
        sub={`of ${totalRows.toLocaleString()} total`}
        critical={criticalCount > 0}
        accent="indigo"
      />
      <KpiCard
        label="Liquid flow (latest)"
        value={fmt2(latest?.liquid_flow)}
        unit="L/min"
        accent="blue"
      />
      <KpiCard
        label="60s Test MAE"
        value={m60 ? fmt4(m60.mae) : '—'}
        unit="°C"
        sub="Ridge α=1.0, Ext-21"
        accent="green"
      />
      <KpiCard
        label="300s Test MAE"
        value={m300 ? fmt4(m300.mae) : '—'}
        unit="°C"
        sub="OLS, Ext-21"
        accent="green"
      />
    </div>
  )
}
