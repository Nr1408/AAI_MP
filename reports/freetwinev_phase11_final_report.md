# Phase 11: Final Technical Report

**Project:** KJS-CES-02, AI-Enabled Micro-channel Battery Thermal Management System  
**Dataset:** FreeTwinEV P45B 1s4p Battery Module Measurement  
**Status:** Phase 11 complete from frozen Phase 0-10 evidence  
**Date:** 2026-10-05

## Executive Summary

This project developed a validation-first battery thermal forecasting and decision-support pipeline using the FreeTwinEV P45B 1s4p battery module dataset. The final system predicts future maximum cell-surface temperature change at 60-second and 300-second horizons, reconstructs absolute future temperature, converts the forecasts into an interpretable thermal-risk layer, and presents the frozen outputs through a Streamlit and Plotly dashboard.

The final selected forecasting models were frozen before held-out test evaluation:

| Horizon | Selected model | Feature set | Validation MAE | Validation RMSE | Validation R2 | Test MAE | Test RMSE | Test R2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 60s | Ridge Regression, alpha = 1.0 | Extended 21 | 0.1598 deg C | 0.7228 deg C | 0.7890 | 0.2271 deg C | 0.8256 deg C | 0.8060 |
| 300s | Linear Regression / OLS | Extended 21 | 2.2582 deg C | 4.5396 deg C | 0.5027 | 1.9302 deg C | 4.9649 deg C | 0.5836 |

The 60-second model provides tighter temporal tracking. The 300-second model remains useful but shows larger tail errors, especially in local rolling-error windows. Risk and cooling outputs are decision-support layers, not certified safety or control systems.

## Dataset and Provenance

The project uses the FreeTwinEV P45B 1s4p Battery Module Measurement dataset. Phase 0 classified it as a Category A candidate because it is a physical laboratory dataset with documented provenance, a liquid-cooled 1S4P module, 14 independent experimental runs, uniform 5-second sampling, and raw sensor channels rather than synthetic risk labels.

Important dataset facts:

| Item | Finding |
|---|---|
| Module | 1S4P battery module using 4 Molicel INR21700-P45B cells |
| Raw observations used in later phases | 22,247 rows across 14 runs |
| Sampling | Approximately 5 seconds |
| Temperature channels | 25 physical thermal sensors, including 12 cell-surface sensors |
| Electrical channels | Current, voltage, and SOC |
| Cooling channels | Liquid flow, inlet temperature, outlet temperature |
| License | CC BY 4.0 |

The previous 500-row EV battery CSV was rejected because it contained leakage-prone and physically discontinuous behavior. The project therefore pivoted to FreeTwinEV before the frozen experiment was built.

## Problem Formulation

The forecasting task is:

```text
Given measurements available at time t, predict future maximum cell-surface temperature at t + h.
```

The model target is future temperature change:

```text
DeltaT_h(t) = T_cell_max(t + h) - T_cell_max(t)
```

Absolute future temperature is reconstructed as:

```text
T_future_pred = T_cell_max(t) + predicted_DeltaT_h(t)
```

Two horizons were frozen:

| Horizon | Steps | Purpose |
|---|---:|---|
| 60s | 12 | Operational response horizon |
| 300s | 60 | Strategic thermal envelope horizon |

This formulation avoids future leakage and makes persistence a meaningful baseline.

## Feature Engineering

The final selected models use the frozen Extended feature representation with 21 features. The features are derived only from information available at time t or earlier.

The feature groups are:

| Group | Examples | Purpose |
|---|---|---|
| Electrical state | `current`, `voltage`, `SOC` | Operating condition and load state |
| Cooling state | `liquid_flow`, `temp_inlet_C`, `temp_ambient_C` | Thermal boundary and cooling condition |
| Thermal state | `T_cell_max`, `T_cell_spread`, `T_plate_mean` | Current module hotspot and spatial gradient |
| Load history | `current_mean_60s`, `current_sq_mean_300s` | Backward-only heating memory |
| Derived heat proxies | `current_squared`, `current_abs`, `power_gross` | Joule-heating and power terms |
| Thermal gradients | `T_cell_max_diff_30s`, `T_cell_to_plate_diff`, `T_cell_to_inlet_diff` | Recent thermal velocity and heat-transfer gradients |
| Cooling interaction | `flow_x_inlet_temp` | Simple flow and inlet-temperature interaction |

The ablation-only features `temp_outlet_C` and `DeltaT_coolant_C` were not promoted into the primary selected model.

## Preprocessing and Split Design

The preprocessing pipeline converts Kelvin temperatures to Celsius where needed, constructs targets per run, enforces run boundaries, removes only terminal rows without future targets, and fits `StandardScaler` only on the training partition. Validation and test rows use the train-fitted scaler.

The frozen split is run-level, not row-level:

| Split | Runs |
|---|---|
| Train | ID01, ID02, ID11, ID12, ID13, ID14, ID15, ID17, ID20, ID21 |
| Validation | ID16, ID22 |
| Held-out test | ID18, ID19 |

The split protects the final stress-test runs from model selection. ID18 and ID19 were not used for feature selection, hyperparameter selection, threshold derivation, risk calibration, or recommendation design.

## Baselines and Model Selection

Phase 4 established persistence, mean, OLS, Ridge, and Extended linear baselines. Extended linear models improved substantially over the 9-feature baseline, especially at 60 seconds.

Phase 5 tested Random Forest, Gradient Boosting, HistGradientBoosting, and XGBoost using grouped cross-validation on training runs and validation evaluation on ID16 and ID22. The nonlinear models did not displace the simpler Extended linear models.

Final model selection used validation evidence only:

1. Lowest validation MAE.
2. Lowest validation RMSE as tie-breaker.
3. Highest validation R2 as secondary tie-breaker.

The selected models were:

| Horizon | Frozen selected model | Reason |
|---|---|---|
| 60s | Extended Ridge, alpha = 1.0 | Best validation MAE/RMSE/R2 among selected candidates |
| 300s | Extended OLS | Slightly stronger frozen validation evidence than 300s Ridge |

## Final Held-Out Test Evaluation

After model selection was frozen, Phase 6 evaluated the selected models once on ID18 and ID19.

| Horizon | Model | Test N | Test MAE | Test RMSE | Test R2 | Bias | Max AE |
|---|---|---:|---:|---:|---:|---:|---:|
| 60s | Extended Ridge | 3,449 | 0.2271 deg C | 0.8256 deg C | 0.8060 | 0.0383 deg C | 12.2504 deg C |
| 300s | Extended OLS | 3,353 | 1.9302 deg C | 4.9649 deg C | 0.5836 | -0.4013 deg C | 43.3812 deg C |

Per-run held-out results:

| Horizon | Run | N | MAE | RMSE | R2 | Bias | Max AE |
|---|---|---:|---:|---:|---:|---:|---:|
| 60s | ID18 | 1,242 | 0.3045 | 0.9191 | 0.7769 | -0.0308 | 11.0222 |
| 60s | ID19 | 2,207 | 0.1835 | 0.7681 | 0.8244 | 0.0772 | 12.2504 |
| 300s | ID18 | 1,194 | 2.2479 | 4.6361 | 0.6632 | -0.7394 | 22.0166 |
| 300s | ID19 | 2,159 | 1.7545 | 5.1377 | 0.5340 | -0.2143 | 43.3812 |

The 300-second test MAE is lower than validation MAE, but the test RMSE is higher than validation RMSE. This indicates larger extreme errors despite lower average absolute error.

## Temporal Forecasting Findings

Phase 7 reused persisted model predictions and did not refit any model. It fixed a 12-sample trailing window, equal to 60 seconds at the established sampling interval.

| Split | Horizon | N | MAE | RMSE | P95 AE | Mean rolling MAE | Max rolling MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| Validation | 60s | 4,513 | 0.1598 | 0.7228 | 0.2873 | 0.1603 | 8.1580 |
| Validation | 300s | 4,417 | 2.2582 | 4.5396 | 5.7027 | 2.2613 | 41.9898 |
| Test post-hoc | 60s | 3,449 | 0.2271 | 0.8256 | 0.5376 | 0.2281 | 6.3786 |
| Test post-hoc | 300s | 3,353 | 1.9302 | 4.9649 | 9.7084 | 1.9405 | 40.4867 |

The main temporal conclusion is that 60-second forecasting is substantially tighter. The 300-second horizon has meaningful signal but larger local error excursions.

## Thermal-Risk Layer

Phase 8 converts frozen forecasts into interpretable risk categories. It uses:

| Source | Use |
|---|---|
| 70 deg C documented cutoff | Critical boundary |
| Validation P75/P90/P99 forecast bands | Descriptive intermediate bands |
| Predicted DeltaT, recent trend, cell spread, and persistence | Secondary escalation signals |

Risk categories:

| Category | Meaning |
|---|---|
| Normal | Below validation P75 and no escalation signal |
| Elevated | At/above validation P75 or one secondary signal |
| High | At/above validation P90, sustained elevated state, or multiple secondary signals |
| Critical | Predicted future temperature at or above 70 deg C |

Validation produced 11 critical rows for ID16 at 60s. Held-out post-hoc test application produced 9 critical rows for ID19 at 60s and none for ID18.

These categories are decision-support indicators. They are not certified safety labels.

## Cooling Recommendation Layer

Phase 9 maps risk and current cooling context into conservative recommendation states:

| Recommendation | Rule summary |
|---|---|
| `MAINTAIN` | Normal risk |
| `INCREASE_COOLING` | Elevated risk, active load at or above 20 A, and flow below validation P75 reference |
| `HIGH_COOLING_ATTENTION` | High risk, or Elevated risk without validation-supported increase-flow evidence |
| `CRITICAL_PROTECTIVE_ACTION` | Critical risk |

Validation-only flow and inlet-temperature associations with predicted future temperature were:

| Horizon | Flow Spearman | Inlet-temperature Spearman |
|---|---:|---:|
| 60s | -0.4325 | 0.4195 |
| 300s | -0.4865 | 0.5451 |

These are observational associations and may be confounded by operating regime. The recommendation layer is guidance, not causal proof and not a controller.

## Dashboard

Phase 10 implemented a read-only Streamlit and Plotly dashboard. It consumes persisted Phase 6-9 CSV/JSON artifacts, exposes filters for run, horizon, time, risk, and recommendation state, and visualizes forecast trajectories, risk distributions, cooling conditions, recommendation transitions, and final model metrics.

The dashboard does not train models, tune thresholds, or generate new predictions. The Phase 10 audit reports `PASS`.

## Audit and Reproducibility

Completed audit files show the frozen experiment is reproducible and protected:

| Phase | Audit status observed |
|---|---|
| Phase 5 | PASS |
| Phase 6 | PASS |
| Phase 7 | PASS |
| Phase 8 | `status = COMPLETE`, `overall_status = PASS` |
| Phase 9 | `status = COMPLETE`, `overall_status = PASS` |
| Phase 10 | PASS |

The prior Phase 11 audit file expected Phase 8 and Phase 9 top-level `status` to equal `PASS`, which caused a stale Phase 11 failure. The Phase 8 and Phase 9 audit files themselves report `overall_status = PASS`; this report treats that as the substantive audit result and records the top-level wording mismatch as an audit-format issue.

## Limitations

1. The dataset contains 14 physical runs from one laboratory setup, not a broad fleet or pack-level deployment.
2. The held-out test set contains only ID18 and ID19, both starved-cooling stress runs.
3. The 300-second model has much larger RMSE and tail errors than the 60-second model.
4. The risk bands P75/P90/P99 are descriptive validation percentiles, not certified limits.
5. The 70 deg C cutoff is documented for this project, but the system is not a certified safety controller.
6. Cooling recommendations are based on observational associations and transparent rules, not causal closed-loop validation.
7. The dashboard is a visualization layer over frozen outputs, not a production safety interface.

## Conclusion

The project demonstrates a defensible end-to-end thermal forecasting workflow: documented dataset replacement, leakage-controlled target and feature engineering, run-level validation, frozen model selection, held-out stress evaluation, temporal diagnostics, interpretable risk classification, conservative cooling decision support, and a read-only dashboard.

The strongest result is the 60-second Extended Ridge model, which achieved 0.2271 deg C MAE and 0.8060 R2 on the held-out test runs. The 300-second Extended OLS model remains useful but has materially larger tail-error risk. The final system should be presented as a validated academic decision-support prototype, not as a certified EV battery safety controller.

