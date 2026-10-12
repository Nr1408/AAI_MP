# Phase 7: Temporal Forecasting

**Project:** KJS-CES-02, AI-Enabled Micro-channel Battery Thermal Management System  
**Dataset:** FreeTwinEV P45B 1s4p Battery Module Measurement  
**Status:** Complete; Phase 0-6 artifacts remain frozen  
**Model changes:** None

## 1. Objective

Phase 7 evaluates the temporal behavior of the already-selected models rather than introducing another model search:

- 60s: Extended 21-feature Ridge Regression, `alpha=1.0`.
- 300s: Extended 21-feature Linear Regression / OLS.

The analysis examines trajectories, local rolling error, per-run stability, horizon differences, and relationships between absolute error and frozen operating features. No model was refit or changed.

## 2. Data and validation methodology

Development analysis used persisted Phase 4 selected-model validation predictions for `ID16` and `ID22`. The validation prediction table was joined to the frozen Extended 21-feature validation artifacts using `exp_id` and a rounded 1 ms timestamp key to accommodate harmless floating-point timestamp representations. No feature values were changed.

The temporal sampling interval is approximately 5 seconds. A trailing window of 12 observations was selected because it corresponds to exactly 60 seconds and the primary forecast horizon. Rolling MAE and RMSE are computed within each physical run; no window crosses a run boundary. The initial incomplete window is left missing rather than padded.

Regime analysis uses descriptive Spearman correlations between absolute forecast error and available frozen features: current magnitude, absolute gross power, SOC, flow, inlet temperature, current cell temperature, cell-temperature spread, and absolute 30s thermal change. An upper-quartile current-magnitude subset is reported descriptively using the validation distribution; it is not a tuning threshold.

ID18 and ID19 were not used for any methodological decision. Their temporal rows and frozen feature values are reported only after the validation methodology was fixed, and are labeled `test_posthoc` throughout the artifacts.

## 3. Validation findings

### Pooled validation temporal metrics

| Horizon |     N | MAE (°C) | RMSE (°C) | Bias (°C) | P95 absolute error (°C) | Mean rolling MAE (°C) | Maximum rolling MAE (°C) |
| ------- | ----: | -------: | --------: | --------: | ----------------------: | --------------------: | -----------------------: |
| 60s     | 4,513 |   0.1598 |    0.7228 |    0.0200 |                  0.2873 |                0.1603 |                   8.1580 |
| 300s    | 4,417 |   2.2582 |    4.5396 |    0.6367 |                  5.7027 |                2.2613 |                  41.9898 |

The 60s model has much tighter typical temporal error. The 300s model has a larger error scale and a much larger worst local-error window, demonstrating that long-horizon errors can concentrate in specific temporal segments even when pooled metrics remain interpretable.

### Validation per-run behavior

| Horizon | Run  | MAE (°C) | RMSE (°C) |     R² | Bias (°C) | Maximum AE (°C) |
| ------- | ---- | -------: | --------: | -----: | --------: | --------------: |
| 60s     | ID16 |   0.1847 |    0.8537 | 0.7501 |    0.0655 |         14.7154 |
| 60s     | ID22 |   0.1150 |    0.3879 | 0.9105 |   -0.0622 |          5.6665 |
| 300s    | ID16 |   2.6944 |    5.1104 | 0.4366 |    1.4053 |         45.0174 |
| 300s    | ID22 |   1.4609 |    3.2465 | 0.6745 |   -0.7681 |         21.6911 |

ID16 is harder for both horizons, with the largest 300s bias and maximum error. This is a measured regime-specific pattern, not a basis for changing the frozen model.

### Temporal regime relationships

Validation Spearman correlations with absolute error were:

| Horizon | Variable with notable association | Spearman correlation |
| ------- | --------------------------------- | -------------------: |
| 60s     | Absolute 30s thermal change       |               0.4894 |
| 60s     | Current cell temperature          |               0.4365 |
| 300s    | Inlet temperature                 |               0.4228 |
| 300s    | Cell-temperature spread           |              -0.3782 |
| 300s    | Absolute 30s thermal change       |               0.0934 |

Absolute current had weak negative correlations with absolute error in validation (`-0.0230` at 60s and `-0.2304` at 300s). Therefore these artifacts do not support a simple claim that error monotonically increases with instantaneous current magnitude. The stronger 60s association with recent thermal change supports a temporal-error interpretation, but correlations are descriptive and not causal.

## 4. Forecast trajectories and stability

Figures 36 and 38 show validation trajectories and trailing rolling error. The selected predictions follow the broad temporal shape of actual DeltaT, while local error spikes occur in limited portions of runs. The 300s trajectory is visibly less locally stable than the 60s trajectory, consistent with its higher pooled MAE/RMSE and larger maximum rolling error.

The generated artifacts also record prediction step variability and prediction-step MAE per run. These are descriptive stability measures, not new objectives used to modify the model.

## 5. Held-out test observations, post-hoc only

The following observations were generated after the validation methodology was frozen. They were not used for temporal-window choice, feature selection, threshold selection, calibration, model selection, or error-driven model changes.

| Split         | Horizon |     N | MAE (°C) | RMSE (°C) | Bias (°C) | P95 absolute error (°C) | Mean rolling MAE (°C) | Maximum rolling MAE (°C) |
| ------------- | ------- | ----: | -------: | --------: | --------: | ----------------------: | --------------------: | -----------------------: |
| Test post-hoc | 60s     | 3,449 |   0.2271 |    0.8256 |    0.0383 |                  0.5376 |                0.2281 |                   6.3786 |
| Test post-hoc | 300s    | 3,353 |   1.9302 |    4.9649 |   -0.4013 |                  9.7084 |                1.9405 |                  40.4867 |

Per-run test behavior remains available in `phase7_temporal_metrics_per_run.csv`. At 60s, ID18 has MAE 0.3045°C versus 0.1835°C for ID19. At 300s, ID18 has MAE 2.2479°C versus 1.7545°C for ID19, while ID19 has the larger RMSE and maximum error. These are post-hoc observations only.

## 6. Artifacts and diagnostics

- `results/freetwinev/phase7_temporal_predictions_validation.csv`
- `results/freetwinev/phase7_temporal_predictions_test_posthoc.csv`
- `results/freetwinev/phase7_temporal_metrics_per_run.csv`
- `results/freetwinev/phase7_rolling_error_summary.csv`
- `results/freetwinev/phase7_regime_error_summary.csv`
- `results/freetwinev/phase7_horizon_comparison.csv`
- `results/freetwinev/phase7_audit.json`

Figures 36-40 under `results/freetwinev/figures/` cover validation trajectories, post-hoc test trajectories, rolling error, regime correlations, and horizon comparison.

## 7. Integrity audit

`results/freetwinev/phase7_audit.json` reports **PASS**. It verifies:

- Phase 3-6 artifact hashes are unchanged before and after Phase 7.
- The Extended feature contract remains exactly 21 features with no target/future columns.
- Train, validation, and test partition definitions remain unchanged.
- Validation decisions precede any test post-hoc access.
- No test tuning, calibration, model selection, or error-driven modification occurred.
- Temporal ordering is strictly increasing within each run.
- Both horizons are represented.
- There are no duplicate temporal prediction rows or missing predictions.
- Required Phase 7 outputs exist.

## 8. Limitations

Only two validation runs and two held-out runs are available. Rolling statistics depend on the established approximately 5-second sampling interval and describe local behavior rather than creating independent samples. Regime correlations can reflect confounding between operating variables and should not be interpreted causally. The post-hoc test analysis cannot be used to claim development-time forecasting improvement.

## 9. Conclusion and phase boundary

Phase 7 demonstrates that the frozen 60s model is temporally tighter and more stable than the frozen 300s model, while the 300s model shows larger local error excursions and stronger regime sensitivity. No model was changed, and no new forecasting method was introduced. The project is ready to proceed to **Phase 8 - Thermal-Risk Layer** only after explicit authorization.
