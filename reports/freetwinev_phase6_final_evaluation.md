# Phase 6: Final Held-Out Test Evaluation

**Project:** KJS-CES-02, AI-Enabled Micro-channel Battery Thermal Management System  
**Dataset:** FreeTwinEV P45B 1s4p Battery Module Measurement  
**Execution:** 2026-10-01  
**Status:** Complete; no post-test tuning performed

## A. Frozen model-selection evidence

Model selection was frozen before the test CSVs were loaded. The objective rule was:

1. Lowest validation MAE.
2. Lowest validation RMSE as tie-breaker.
3. Highest validation R² as secondary tie-breaker.

Both selected configurations use the frozen Extended 21-feature representation and the Phase 3 train-fitted `StandardScaler` representation.

| Horizon | Selected model    | Specification        | Validation MAE (°C) | Validation RMSE (°C) | Validation R² |
| ------- | ----------------- | -------------------- | ------------------: | -------------------: | ------------: |
| 60s     | Ridge Regression  | `Ridge(alpha=1.0)`   |              0.1598 |               0.7228 |        0.7890 |
| 300s    | Linear Regression | `LinearRegression()` |              2.2582 |               4.5396 |        0.5027 |

The selection evidence was read from the persisted Phase 4 baseline results. No ID18 or ID19 data, predictions, or metrics entered this decision.

## B. Final held-out test results

The test set contains 3,449 valid rows at 60s and 3,353 valid rows at 300s. Metrics are pooled across ID18 and ID19.

| Horizon | Model          | Test MAE (°C) | Test RMSE (°C) | Test R² | Bias (°C) | Max AE (°C) | Absolute MAE (°C) | Absolute RMSE (°C) | Absolute R² |
| ------- | -------------- | ------------: | -------------: | ------: | --------: | ----------: | ----------------: | -----------------: | ----------: |
| 60s     | Extended Ridge |        0.2271 |         0.8256 |  0.8060 |    0.0383 |     12.2504 |            0.2271 |             0.8256 |      0.9923 |
| 300s    | Extended OLS   |        1.9302 |         4.9649 |  0.5836 |   -0.4013 |     43.3812 |            1.9302 |             4.9649 |      0.7284 |

Absolute-temperature MAE and RMSE equal the corresponding delta-temperature errors because the same current `T_cell_max` is added to both prediction and target reconstruction.

## C. Per-run test results

| Horizon | Run  |     N | MAE (°C) | RMSE (°C) |     R² | Bias (°C) | Max AE (°C) | Absolute R² |
| ------- | ---- | ----: | -------: | --------: | -----: | --------: | ----------: | ----------: |
| 60s     | ID18 | 1,242 |   0.3045 |    0.9191 | 0.7769 |   -0.0308 |     11.0222 |      0.9929 |
| 60s     | ID19 | 2,207 |   0.1835 |    0.7681 | 0.8244 |    0.0772 |     12.2504 |      0.9914 |
| 300s    | ID18 | 1,194 |   2.2479 |    4.6361 | 0.6632 |   -0.7394 |     22.0166 |      0.8259 |
| 300s    | ID19 | 2,159 |   1.7545 |    5.1377 | 0.5340 |   -0.2143 |     43.3812 |      0.6250 |

ID18 has higher MAE at both horizons. ID19 has the larger 300s RMSE and maximum error, indicating a more severe tail-error contribution at that horizon. These are measured differences; no causal explanation is asserted from the held-out results alone.

## D. Validation versus test comparison

This is a generalization comparison, not a new selection step.

| Horizon   | Validation MAE | Test MAE |  Change | Validation RMSE | Test RMSE |  Change | Validation R² | Test R² |  Change |
| --------- | -------------: | -------: | ------: | --------------: | --------: | ------: | ------------: | ------: | ------: |
| 60s Ridge |         0.1598 |   0.2271 | +0.0673 |          0.7228 |    0.8256 | +0.1028 |        0.7890 |  0.8060 | +0.0170 |
| 300s OLS  |         2.2582 |   1.9302 | -0.3280 |          4.5396 |    4.9649 | +0.4253 |        0.5027 |  0.5836 | +0.0809 |

The 60s model retains strong held-out performance with a modest MAE/RMSE increase. At 300s, test MAE is lower than validation MAE, while RMSE is higher, so the pooled error distribution is not uniformly improved. The test results do not justify changing the frozen models.

## E. Absolute-temperature tracking

For each prediction, absolute future temperature was reconstructed as:

`T_future_pred = T_cell_max + predicted_DeltaT`

The generated tracking figure is [34_phase6_absolute_temperature_tracking.png](../results/freetwinev/figures/34_phase6_absolute_temperature_tracking.png). Absolute R² is higher than delta-temperature R² because current thermal state and autocorrelation explain a large portion of absolute future-temperature variance. It should not be interpreted as independent dynamic-prediction skill.

## F. Test-set integrity audit

The automated audit is stored in `results/freetwinev/phase6_audit.json` and has overall status **PASS**. It verifies:

- Exact evaluated membership is `ID18` and `ID19` only.
- Train, validation, and test run sets are disjoint.
- Exactly 21 Extended features were used in the frozen manifest order.
- No target, future, or lead columns entered `X`.
- `StandardScaler` was frozen and applied from the existing train-fitted representation.
- Model specifications were frozen before test access.
- No test hyperparameter tuning occurred.
- Test metrics were not used for model selection.
- Prediction counts match the test datasets: 3,449 at 60s and 3,353 at 300s.
- There are no missing predictions or duplicate `(exp_id, time_s, horizon)` rows.
- Metrics were computed after predictions were generated.

Figures 31-35 provide actual-versus-predicted DeltaT, residual distributions, residuals versus target, absolute-temperature tracking, and per-run MAE comparisons.

## G. Limitations

Only two held-out physical runs were evaluated, so these metrics are not universal performance guarantees. ID18 and ID19 represent starved-cooling regimes and should be interpreted as a targeted stress evaluation. The evaluation used only the frozen Extended feature pipeline; Ablation 23 was not introduced. No post-test tuning, feature changes, model search, or retraining was performed after metrics were generated.

## Final boundary

Phase 6 is complete. The held-out test results are final for this frozen experiment. No Phase 7 or post-test optimization was started.
