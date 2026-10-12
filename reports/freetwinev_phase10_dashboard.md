# Phase 10: FreeTwinEV Dashboard

## Purpose

Phase 10 provides a read-only Streamlit and Plotly dashboard for visualizing the completed Phase 6-9 experiment. It is a visualization and decision-support interface, not a new model or controller.

## Architecture and data sources

`src/freetwinev_dashboard.py` loads the persisted Phase 6 final metrics and predictions, Phase 7 temporal predictions, Phase 8 risk predictions and metrics, and Phase 9 recommendations and metrics. The 70 deg C cutoff and P75/P90/P99 bands are loaded from `phase8_thresholds.json`; the validation low-flow references and recommendation context are loaded from `phase9_recommendation_config.json`. No training, fitting, threshold tuning, or prediction generation occurs.

`src/run_freetwinev_dashboard.py` performs the data-contract validation and writes `results/freetwinev/phase10_dashboard_audit.json`. It also compares hashes of prior persisted artifacts before and after validation.

## Views and interaction

- Run selector for ID18 and ID19.
- Horizon selector for 60s, 300s, or both.
- Exact persisted `time_s` range filter.
- Risk-category and recommendation filters.
- Current-state KPIs: observed maximum cell temperature, both future forecasts when selected, risk, recommendation, flow, current magnitude, and temperature spread.
- Forecast trajectories for observed `T_cell_max`, predicted future temperature, and the documented 70 deg C boundary.
- Risk distribution, timeline, critical-event summary, and validation-derived descriptive bands.
- Cooling conditions and recommendation transitions.
- Frozen overall and per-run held-out model metrics.

## Validation

The dashboard validator checks artifact existence, required columns, exact run and horizon membership, duplicate composite keys, required-data missingness, exact timestamp alignment, persisted threshold/config coverage, and PASS status for Phase 6-9 audits. The Phase 10 runner additionally checks the frozen published metrics and prior-artifact hashes. Initial rolling-window NaNs are not silently filled; required dashboard fields are checked explicitly.

Run locally from the project root:

```powershell
python src/run_freetwinev_dashboard.py
streamlit run src/freetwinev_dashboard.py
```

## Limitations

Forecasts are model predictions, not guarantees. The 70 deg C boundary is the documented project cutoff, not a new limit established by this dashboard. P75/P90/P99 are validation-derived descriptive forecast bands, not certified safety limits. Cooling recommendations are decision-support guidance; Phase 9 does not establish causal cooling effectiveness. ID18 and ID19 are held-out evaluation observations. Dashboard interaction changes filtering only and cannot change the frozen model, risk, or recommendation methodology.