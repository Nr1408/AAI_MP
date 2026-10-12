import { useState, useMemo } from 'react'
import { useData } from './hooks/useData.js'
import { Sidebar } from './components/Sidebar.jsx'
import { Header } from './components/Header.jsx'
import { KpiRow } from './components/KpiRow.jsx'
import { ForecastChart } from './components/ForecastChart.jsx'
import { RiskDistributionBar, RiskTimeline, RiskPie } from './components/RiskCharts.jsx'
import { RecommendationBar, FlowTemperatureChart, RecommendationTable } from './components/CoolingPanel.jsx'
import { ModelMetricsTable, PerRunTable, MAEComparisonChart } from './components/ModelPerformance.jsx'
import { ThresholdPanel } from './components/ThresholdPanel.jsx'
import { RUNS, HORIZONS } from './lib/constants.js'

const TABS = ['Overview', 'Forecast', 'Risk', 'Cooling', 'Model Performance']

function TabBar({ active, setActive }) {
  return (
    <div className="flex gap-1 px-6 pt-3 border-b border-slate-800 bg-slate-950/60">
      {TABS.map(t => (
        <button
          key={t}
          onClick={() => setActive(t)}
          className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-all ${
            active === t
              ? 'bg-slate-800 text-white border border-slate-700 border-b-slate-800 -mb-px'
              : 'text-slate-500 hover:text-slate-300'
          }`}
        >
          {t}
        </button>
      ))}
    </div>
  )
}

function LoadingScreen() {
  return (
    <div className="flex items-center justify-center h-screen bg-slate-950">
      <div className="text-center">
        <div className="text-5xl mb-4 animate-pulse">🔋</div>
        <p className="text-lg text-slate-400 font-semibold">Loading dashboard data…</p>
        <p className="text-sm text-slate-600 mt-1">Parsing frozen Phase 6–9 artifacts</p>
      </div>
    </div>
  )
}

function ErrorScreen({ error }) {
  return (
    <div className="flex items-center justify-center h-screen bg-slate-950">
      <div className="text-center max-w-md">
        <div className="text-5xl mb-4">⚠️</div>
        <p className="text-lg text-red-400 font-semibold mb-2">Failed to load data</p>
        <p className="text-sm text-slate-500 font-mono bg-slate-900 p-3 rounded-lg">{error}</p>
        <p className="text-xs text-slate-600 mt-3">Make sure the dev server is running from c:\AAI MP\dashboard</p>
      </div>
    </div>
  )
}

export default function App() {
  const { data, loading, error } = useData()
  const [tab, setTab] = useState('Overview')
  const [filters, setFilters] = useState({
    runs: [...RUNS],
    horizons: [...HORIZONS],
    risks: ['Normal', 'Elevated', 'High', 'Critical'],
    recs: ['MAINTAIN', 'INCREASE_COOLING', 'HIGH_COOLING_ATTENTION', 'CRITICAL_PROTECTIVE_ACTION'],
  })

  const filtered = useMemo(() => {
    if (!data) return []
    return data.timeseries.filter(r =>
      filters.runs.includes(r.exp_id) &&
      filters.horizons.includes(r.horizon) &&
      filters.risks.includes(r.risk_category) &&
      filters.recs.includes(r.recommendation)
    )
  }, [data, filters])

  if (loading) return <LoadingScreen />
  if (error) return <ErrorScreen error={error} />

  return (
    <div className="flex h-screen overflow-hidden bg-slate-950">
      <Sidebar filters={filters} setFilters={setFilters} />

      <div className="flex flex-col flex-1 overflow-hidden">
        <Header filtered={filtered} thresholds={data.thresholds} />
        <TabBar active={tab} setActive={setTab} />

        <main className="flex-1 overflow-y-auto p-4">
          {tab === 'Overview' && (
            <OverviewTab data={data} filtered={filtered} filters={filters} />
          )}
          {tab === 'Forecast' && (
            <ForecastTab data={data} filtered={filtered} />
          )}
          {tab === 'Risk' && (
            <RiskTab data={data} filtered={filtered} filters={filters} />
          )}
          {tab === 'Cooling' && (
            <CoolingTab data={data} filtered={filtered} filters={filters} />
          )}
          {tab === 'Model Performance' && (
            <ModelTab data={data} />
          )}
        </main>

        <footer className="border-t border-slate-800 px-6 py-2 flex items-center justify-between flex-shrink-0">
          <p className="text-xs text-slate-600">
            FreeTwinEV P45B 1s4p · Phase 11 · Read-only over frozen artifacts · {filtered.length.toLocaleString()} rows in view
          </p>
          <p className="text-xs text-slate-600">
            ⚠️ Decision-support only · Not a certified safety controller
          </p>
        </footer>
      </div>
    </div>
  )
}

function OverviewTab({ data, filtered, filters }) {
  return (
    <>
      <KpiRow filtered={filtered} modelPerf={data.modelPerf} />
      <div className="grid grid-cols-3 gap-4">
        <ForecastChart filtered={filtered} thresholds={data.thresholds} />
        <ThresholdPanel thresholds={data.thresholds} recConfig={data.recConfig} />
        <RiskDistributionBar riskMetrics={data.riskMetrics} filters={filters} />
        <RecommendationBar recMetrics={data.recMetrics} filters={filters} />
        <MAEComparisonChart modelPerf={data.modelPerf} valMetrics={data.valMetrics} />
      </div>
    </>
  )
}

function ForecastTab({ data, filtered }) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <ForecastChart filtered={filtered} thresholds={data.thresholds} />
      <ThresholdPanel thresholds={data.thresholds} recConfig={data.recConfig} />
      <FlowTemperatureChart filtered={filtered} />
    </div>
  )
}

function RiskTab({ data, filtered, filters }) {
  return (
    <div className="grid grid-cols-3 gap-4">
      <RiskDistributionBar riskMetrics={data.riskMetrics} filters={filters} />
      <RiskPie filtered={filtered} />
      <div className="col-span-3">
        <RiskTimeline filtered={filtered} />
      </div>
    </div>
  )
}

function CoolingTab({ data, filtered, filters }) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <RecommendationBar recMetrics={data.recMetrics} filters={filters} />
      <ThresholdPanel thresholds={data.thresholds} recConfig={data.recConfig} />
      <FlowTemperatureChart filtered={filtered} />
      <RecommendationTable filtered={filtered} />
    </div>
  )
}

function ModelTab({ data }) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <ModelMetricsTable modelPerf={data.modelPerf} valMetrics={data.valMetrics} />
      <PerRunTable modelPerfPerRun={data.modelPerfPerRun} />
      <MAEComparisonChart modelPerf={data.modelPerf} valMetrics={data.valMetrics} />
      <ThresholdPanel thresholds={data.thresholds} recConfig={data.recConfig} />
    </div>
  )
}
