# Phase 11 Presentation

## Slide 1: AI-enabled battery thermal forecasting

FreeTwinEV P45B 1s4p battery module  
AAI Mini Project, KJS-CES-02  
Forecasting, risk interpretation, cooling guidance, and dashboard

Speaker note: This deck summarizes frozen Phase 0-10 evidence. No model or threshold changes are introduced in Phase 11.

## Slide 2: Project objective

- Predict future maximum cell-surface temperature from measurements available at time t.
- Use 60-second and 300-second horizons.
- Preserve train, validation, and held-out test separation.
- Convert forecasts into interpretable risk and cooling decision-support outputs.

Speaker note: The project emphasizes scientific validity over headline metrics.

## Slide 3: Dataset replacement and provenance

- Previous 500-row EV battery CSV was rejected.
- FreeTwinEV is a physical laboratory dataset with documented protocol.
- 14 independent runs, 22,247 observations, 5-second sampling.
- 25 temperature channels, electrical telemetry, and cooling measurements.

Speaker note: The replacement dataset fixed the earlier leakage and temporal-continuity problems.

## Slide 4: Forecasting formulation

Target:

`DeltaT_h(t) = T_cell_max(t + h) - T_cell_max(t)`

Absolute reconstruction:

`T_future_pred = T_cell_max(t) + predicted_DeltaT_h(t)`

Horizon choices:

- 60s: operational response.
- 300s: strategic thermal envelope.

Speaker note: Training on DeltaT makes persistence explicit and avoids confusing autocorrelation with dynamic skill.

## Slide 5: Frozen split and leakage controls

| Split | Runs |
|---|---|
| Train | ID01, ID02, ID11, ID12, ID13, ID14, ID15, ID17, ID20, ID21 |
| Validation | ID16, ID22 |
| Held-out test | ID18, ID19 |

Controls:

- Run-level split, no row shuffle.
- StandardScaler fitted on training only.
- Backward-only rolling features.
- No test-derived model, threshold, or recommendation decisions.

## Slide 6: Feature representation

Final selected models use the Extended 21-feature set:

- Electrical state: current, voltage, SOC.
- Cooling state: flow, inlet temperature, ambient temperature.
- Thermal state: current hotspot, cell spread, plate mean.
- Derived heating and history terms: current squared, power, rolling current and current-squared means.
- Thermal velocity and gradients: recent T change and cell-to-plate/inlet differences.

Speaker note: Ablation outlet-temperature features were not promoted into the primary selected model.

## Slide 7: Model selection evidence

| Horizon | Selected model | Validation MAE | Validation RMSE | Validation R2 |
|---|---|---:|---:|---:|
| 60s | Extended Ridge, alpha = 1.0 | 0.1598 | 0.7228 | 0.7890 |
| 300s | Extended OLS | 2.2582 | 4.5396 | 0.5027 |

Tree models did not beat the selected Extended linear models on validation.

Speaker note: Model selection used validation only, before loading ID18 or ID19.

## Slide 8: Held-out test performance

| Horizon | Model | Test MAE | Test RMSE | Test R2 | Max AE |
|---|---|---:|---:|---:|---:|
| 60s | Extended Ridge | 0.2271 | 0.8256 | 0.8060 | 12.2504 |
| 300s | Extended OLS | 1.9302 | 4.9649 | 0.5836 | 43.3812 |

Key interpretation:

- 60s remains tight on held-out stress runs.
- 300s has larger tail errors despite useful pooled R2.

## Slide 9: Temporal behavior

| Split | Horizon | MAE | RMSE | P95 AE | Max rolling MAE |
|---|---|---:|---:|---:|---:|
| Validation | 60s | 0.1598 | 0.7228 | 0.2873 | 8.1580 |
| Validation | 300s | 2.2582 | 4.5396 | 5.7027 | 41.9898 |
| Test post-hoc | 60s | 0.2271 | 0.8256 | 0.5376 | 6.3786 |
| Test post-hoc | 300s | 1.9302 | 4.9649 | 9.7084 | 40.4867 |

Speaker note: The 300-second horizon concentrates large errors in local temporal windows.

## Slide 10: Thermal-risk layer

Risk inputs:

- Predicted future temperature.
- Validation-derived P75/P90/P99 forecast bands.
- Recent thermal trend, predicted DeltaT, and cell spread.
- Documented 70 deg C cutoff.

Risk categories:

- Normal.
- Elevated.
- High.
- Critical.

Held-out observation: ID19 produced 9 critical rows at 60s.

## Slide 11: Cooling recommendation layer

Recommendation states:

- `MAINTAIN`
- `INCREASE_COOLING`
- `HIGH_COOLING_ATTENTION`
- `CRITICAL_PROTECTIVE_ACTION`

Validation-only associations:

| Horizon | Flow Spearman | Inlet-temperature Spearman |
|---|---:|---:|
| 60s | -0.4325 | 0.4195 |
| 300s | -0.4865 | 0.5451 |

Speaker note: These are observational associations, not causal control claims.

## Slide 12: Dashboard

Streamlit and Plotly dashboard over frozen Phase 6-9 artifacts:

- Run and horizon selectors.
- Forecast trajectory views.
- Risk distribution and timeline.
- Cooling recommendation transitions.
- Final held-out model metrics.

The dashboard does not train models or generate new predictions.

## Slide 13: Limitations

- 14 runs from one laboratory module setup.
- ID18 and ID19 are only two held-out stress-test runs.
- 300s errors are much larger than 60s errors.
- Risk bands are descriptive validation percentiles.
- Cooling guidance is decision support, not causal proof or certified control.
- Dashboard is not a production safety interface.

## Slide 14: Final conclusion

The final project demonstrates a reproducible, leakage-controlled thermal forecasting and decision-support pipeline.

Main result:

- 60s Extended Ridge achieved 0.2271 deg C held-out MAE and 0.8060 held-out R2.

Best framing:

- A validated academic prototype for thermal forecasting and decision support.
- Not a certified EV battery safety controller.

