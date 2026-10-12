import { useState, useEffect } from 'react'

const BASE = '/data'

async function fetchJSON(name) {
  const res = await fetch(`${BASE}/${name}`)
  if (!res.ok) throw new Error(`Failed to load ${name}: ${res.status}`)
  return res.json()
}

export function useData() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      fetchJSON('timeseries.json'),
      fetchJSON('model_performance.json'),
      fetchJSON('model_performance_per_run.json'),
      fetchJSON('thresholds.json'),
      fetchJSON('recommendation_config.json'),
      fetchJSON('risk_metrics.json'),
      fetchJSON('recommendation_metrics.json'),
      fetchJSON('validation_metrics.json'),
      fetchJSON('summary.json'),
    ])
      .then(([timeseries, modelPerf, modelPerfPerRun, thresholds, recConfig,
              riskMetrics, recMetrics, valMetrics, summary]) => {
        setData({ timeseries, modelPerf, modelPerfPerRun, thresholds, recConfig,
                  riskMetrics, recMetrics, valMetrics, summary })
        setLoading(false)
      })
      .catch(err => { setError(err.message); setLoading(false) })
  }, [])

  return { data, loading, error }
}
