# Phase 8: Interpretable Thermal-Risk Layer

**Project:** KJS-CES-02, AI-Enabled Micro-channel Battery Thermal Management System  
**Dataset:** FreeTwinEV P45B 1s4p Battery Module Measurement  
**Status:** Complete; Phases 0-7 remain frozen  
**Model changes:** None

## 1. Purpose and frozen inputs

Phase 8 transforms frozen forecasts into interpretable thermal-state and risk indicators. The central derived quantity is:

`predicted_future_temperature = current_T_cell_max + predicted_DeltaT`

The selected forecasting models were not refit or changed:

- 60s: Extended 21-feature Ridge, `alpha=1.0`.
- 300s: Extended 21-feature OLS.

Phase 8 reuses persisted Phase 7 validation and post-hoc test predictions. It does not perform model selection, calibration, feature selection, or hyperparameter tuning.

## 2. Threshold methodology

The project documentation contains a documented maximum cell-surface cutoff of **70 °C** in [freetwinev_dataset_assessment.md](freetwinev_dataset_assessment.md) and [freetwinev_eda.md](freetwinev_eda.md). This is retained as the only engineering cutoff used by the risk layer. It is not newly inferred by Phase 8.

No certified engineering limits were found for the intermediate categories. Therefore, intermediate temperature bands are explicitly **validation-derived descriptive thresholds**, not safety guarantees:

| Horizon | P75 future temperature | P90 future temperature | P99 future temperature | P90 predicted DeltaT | P90 recent 30s trend | P90 cell spread |
| ------- | ---------------------: | ---------------------: | ---------------------: | -------------------: | -------------------: | --------------: |
| 60s     |             20.3288 °C |             24.0607 °C |             54.7323 °C |            0.1736 °C |            0.0472 °C |       6.4343 °C |
| 300s    |             21.0309 °C |             23.1461 °C |             62.6034 °C |            2.2727 °C |            0.0498 °C |       6.4908 °C |

Thresholds were calculated only from validation runs ID16 and ID22 and persisted in `results/freetwinev/phase8_thresholds.json`. The persistence window is 12 samples, equal to 60 seconds at the established 5-second sampling interval.

## 3. Risk logic

The category is transparent and rule-based:

- **Normal:** predicted future temperature below validation P75, with no escalation signal.
- **Elevated:** predicted future temperature at or above validation P75, or one secondary escalation signal.
- **High:** predicted future temperature at or above validation P90, or at least two secondary signals, or a sustained elevated-temperature period.
- **Critical:** predicted future temperature at or above the documented 70 °C engineering cutoff.

Secondary signals are retained explicitly rather than hidden in a weighted score:

- predicted DeltaT at or above the validation P90 predicted-rise threshold;
- recent 30-second thermal change at or above its validation P90 threshold;
- cell-temperature spread at or above its validation P90 threshold;
- all observations in a trailing 60-second window remaining at or above the validation P75 temperature band.

These categories describe predicted thermal state and trajectory. They do not describe prediction error, and “Normal” does not mean certified safe.

## 4. Validation results

Risk metrics are calculated separately by physical run and horizon.

| Split      | Horizon | Run  | Max predicted future T | Max predicted DeltaT | Normal | Elevated | High | Critical | Longest non-normal period |
| ---------- | ------- | ---- | ---------------------: | -------------------: | -----: | -------: | ---: | -------: | ------------------------: |
| Validation | 60s     | ID16 |             74.2785 °C |           13.3236 °C |  2,054 |       98 |  740 |       11 |                   1,605 s |
| Validation | 60s     | ID22 |             49.0723 °C |            6.5029 °C |  1,016 |       64 |  530 |        0 |                   1,505 s |
| Validation | 300s    | ID16 |             64.0653 °C |           39.5067 °C |  1,971 |      241 |  643 |        0 |                   2,025 s |
| Validation | 300s    | ID22 |             38.9228 °C |           21.1460 °C |    997 |      133 |  432 |        0 |                   1,495 s |

ID16 produces the highest validation predicted future temperature and the only validation critical rows, all at the 60s horizon. The 300s ID16 forecast reaches a large predicted DeltaT but remains below the 70 °C cutoff in the frozen forecast output.

## 5. Held-out test observations

ID18 and ID19 were loaded only after validation thresholds and risk logic were persisted and the pre-test audit passed. These results are held-out evaluation, not development evidence.

| Split         | Horizon | Run  | Max predicted future T | Max predicted DeltaT | Normal | Elevated |  High | Critical | Longest non-normal period |
| ------------- | ------- | ---- | ---------------------: | -------------------: | -----: | -------: | ----: | -------: | ------------------------: |
| Test post-hoc | 60s     | ID18 |             68.8675 °C |            6.5570 °C |    682 |       10 |   550 |        0 |                   1,870 s |
| Test post-hoc | 60s     | ID19 |             73.5601 °C |           13.1593 °C |  1,043 |       81 | 1,074 |        9 |                   3,170 s |
| Test post-hoc | 300s    | ID18 |             50.6887 °C |           23.0309 °C |    734 |        0 |   460 |        0 |                   1,535 s |
| Test post-hoc | 300s    | ID19 |             62.5451 °C |           38.6763 °C |  1,379 |       39 |   741 |        0 |                   1,440 s |

ID19 has the only post-hoc test critical rows: 9 rows at 60s because predicted future temperature exceeds 70 °C. ID18 remains below the documented cutoff at both horizons. These observations were not used to alter thresholds or logic.

## 6. Temporal interpretation

The risk layer preserves run boundaries and calculates persistence within each `(split, exp_id, horizon)` sequence. A brief secondary signal can produce an Elevated indicator, while sustained elevated temperature or multiple simultaneous secondary signals can produce High. This distinguishes a temperature-state category from an isolated residual or forecast error.

The Phase 8 figures are:

- `41_phase8_temperature_risk_overview.png`
- `42_phase8_validation_risk_timeline.png`
- `43_phase8_risk_duration.png`
- `44_phase8_test_risk_timeline_posthoc.png`

## 7. Integrity audit

`results/freetwinev/phase8_audit.json` reports **PASS**. It verifies:

- Phase 3-7 artifact hashes are unchanged.
- Frozen model, feature, scaler, and partition contracts remain intact.
- Thresholds are derived only from ID16 and ID22.
- No test threshold tuning, feature selection, calibration, or model refitting occurred.
- Validation and test run memberships are exact.
- Both horizons are represented.
- Required risk inputs are complete.
- Temporal ordering is monotonic within each run and horizon.
- There are no duplicate temporal rows.
- Predicted future-temperature reconstruction matches the persisted Phase 7 absolute prediction.

## 8. Limitations

The P75/P90/P99 bands are descriptive summaries of only two validation runs, not certified safety limits. Prediction uncertainty is not modeled, and forecast temperature is a proxy for future thermal state rather than a direct measurement. The 70 °C boundary is documented in the project’s dataset assessment, but this Phase 8 layer does not establish regulatory or battery-manufacturer safety certification. Dataset coverage is limited, and extrapolation outside the observed operating envelope remains uncertain. Risk categories should therefore support interpretation and monitoring, not autonomous safety decisions.

## 9. Phase boundary

Phase 8 is complete. No Phase 9 cooling recommendation, Phase 10 dashboard, or Phase 11 reporting work was started.
