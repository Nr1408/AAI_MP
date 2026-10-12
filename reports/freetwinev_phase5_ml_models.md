# Phase 5: ML Model Exploration and Cross-Validation

**Project:** KJS-CES-02, AI-Enabled Micro-channel Battery Thermal Management System  
**Dataset:** FreeTwinEV P45B 1s4p Battery Module Measurement  
**Status:** Complete; final test runs remain untouched  
**Execution:** 2026-10-01, seed 42

## Objective

Phase 5 tests whether nonlinear tree ensembles improve on the Phase 4 linear Extended Ridge benchmark. The evaluated models are Random Forest, Gradient Boosting, HistGradientBoosting, and XGBoost. No neural networks or Phase 6 work was performed.

## Methodology

The locked run-level partition was preserved:

- Training: `ID01`, `ID02`, `ID11`, `ID12`, `ID13`, `ID14`, `ID15`, `ID17`, `ID20`, `ID21`
- Validation: `ID16`, `ID22`
- Final test, not loaded or evaluated: `ID18`, `ID19`

For each horizon (`target_DeltaT_60s`, `target_DeltaT_300s`) and each feature set (Baseline 9 and Extended 21), hyperparameters were selected with five-fold `GroupKFold`, grouping by `exp_id`. Rows from one physical run therefore remain in one fold. Model selection used training-fold mean absolute error only. The selected estimator was refit on all training rows and evaluated once on the two validation runs.

Phase 3 preprocessing was consumed as unscaled CSV artifacts. No imputation, scaling change, row-level random split, or raw-data modification was introduced. Targets remain in degrees Celsius.

## Search spaces

The compact, reproducible grids are recorded in `results/freetwinev/phase5_config.json`; each model has eight configurations:

- Random Forest: `n_estimators=[100,200]`, `max_depth=[8,None]`, `min_samples_leaf=[5]`, `max_features=[sqrt,0.5]`.
- Gradient Boosting: `n_estimators=[100,200]`, `learning_rate=[0.05]`, `max_depth=[3,5]`, `min_samples_leaf=[5]`, `subsample=[0.8,1.0]`.
- HistGradientBoosting: `max_iter=[100,200]`, `learning_rate=[0.05]`, `max_depth=[4,7]`, `min_samples_leaf=[30]`, `l2_regularization=[0,1]`.
- XGBoost: `n_estimators=[100,200]`, `learning_rate=[0.05,0.1]`, `max_depth=[4,6]`, `subsample=[0.8]`, `colsample_bytree=[0.7]`, `reg_lambda=[5]`.

## Pooled results

Values are degrees Celsius except R², seconds, and feature counts. CV values are mean ± standard deviation across five grouped folds. Validation is pooled `ID16 + ID22`.

| Horizon | Features    | Model                |              CV MAE |             CV RMSE |               CV R² |    Val MAE | Val RMSE |  Val R² |   Bias |  Max AE |
| ------- | ----------- | -------------------- | ------------------: | ------------------: | ------------------: | ---------: | -------: | ------: | -----: | ------: |
| 60s     | Baseline 9  | Random Forest        |     0.2970 ± 0.1122 |     0.6893 ± 0.3939 |    -0.0146 ± 1.1286 |     0.9727 |   1.4580 |  0.1417 | 0.8789 | 12.7677 |
| 60s     | Baseline 9  | Gradient Boosting    |     0.3468 ± 0.1024 |     0.7411 ± 0.3533 |    -0.0873 ± 0.9126 |     0.7557 |   1.2327 |  0.3865 | 0.6224 | 14.1362 |
| 60s     | Baseline 9  | HistGradientBoosting |     0.2698 ± 0.0825 |     0.6457 ± 0.3283 |     0.4697 ± 0.1859 |     0.4863 |   0.9123 |  0.6639 | 0.3077 | 14.7230 |
| 60s     | Baseline 9  | XGBoost              |     0.3096 ± 0.0877 |     0.6829 ± 0.3245 |     0.0879 ± 0.8668 |     0.9839 |   1.4421 |  0.1603 | 0.8297 | 14.8854 |
| 60s     | Extended 21 | Random Forest        | **0.1681 ± 0.0499** | **0.4971 ± 0.3136** |     0.5097 ± 0.5607 |     0.3311 |   0.7861 |  0.7505 | 0.2170 | 13.6253 |
| 60s     | Extended 21 | Gradient Boosting    |     0.2630 ± 0.1802 |     0.7276 ± 0.3737 |     0.1123 ± 0.8510 | **0.2911** |   0.7889 |  0.7487 | 0.1886 | 13.8727 |
| 60s     | Extended 21 | HistGradientBoosting |     0.2134 ± 0.0977 |     0.5915 ± 0.2945 |     0.5561 ± 0.2243 |     0.4752 |   0.9351 |  0.6470 | 0.3198 | 13.5397 |
| 60s     | Extended 21 | XGBoost              |     0.2078 ± 0.1025 |     0.5926 ± 0.3041 | **0.5572 ± 0.2415** |     0.5650 |   1.0572 |  0.5487 | 0.4589 | 14.5348 |
| 300s    | Baseline 9  | Random Forest        | **1.9221 ± 0.5547** | **3.3755 ± 1.4090** |    -0.4901 ± 1.5571 |     8.5645 |  12.1677 | -2.5727 | 7.8581 | 26.0293 |
| 300s    | Baseline 9  | Gradient Boosting    |     2.5460 ± 1.3668 |     4.7140 ± 2.2389 |    -2.0968 ± 3.1498 |     7.0776 |  10.6824 | -1.7537 | 6.2394 | 27.2505 |
| 300s    | Baseline 9  | HistGradientBoosting |     2.0825 ± 0.7129 |     3.7111 ± 1.6435 |    -0.5032 ± 1.2564 |     9.7640 |  13.7721 | -3.5770 | 9.1300 | 27.2639 |
| 300s    | Baseline 9  | XGBoost              |     2.2865 ± 0.9994 |     3.9664 ± 1.7688 |    -1.0578 ± 1.8535 |     8.0680 |  11.2944 | -2.0783 | 7.3167 | 25.9132 |
| 300s    | Extended 21 | Random Forest        | **1.3541 ± 0.5243** | **2.7671 ± 1.3644** | **0.3201 ± 0.4194** |     4.3577 |   6.3872 |  0.0155 | 3.4057 | 36.6807 |
| 300s    | Extended 21 | Gradient Boosting    |     2.3847 ± 1.9650 |     4.7687 ± 3.0850 |    -2.9610 ± 5.7070 |     5.8940 |   8.3759 | -0.6929 | 5.0218 | 31.9733 |
| 300s    | Extended 21 | HistGradientBoosting |     2.2076 ± 1.4685 |     4.3305 ± 2.5379 |    -2.0034 ± 3.8786 |     7.7215 |  10.5744 | -1.6983 | 6.8649 | 27.7827 |
| 300s    | Extended 21 | XGBoost              |     2.0859 ± 1.2817 |     4.0146 ± 2.2512 |    -1.2308 ± 2.8290 |     5.6274 |   8.2520 | -0.6432 | 4.7390 | 34.9917 |

The complete machine-readable table, including parameter dictionaries, training time, training metrics, and absolute-temperature metrics, is `results/freetwinev/phase5_results.csv`.

## Per-run validation

The operating regimes are not interchangeable. At 60s with Extended 21, the best validation MAE by run was:

| Model                |   ID16 MAE |   ID22 MAE |
| -------------------- | ---------: | ---------: |
| Random Forest        |     0.4456 | **0.1246** |
| Gradient Boosting    | **0.3758** |     0.1383 |
| HistGradientBoosting |     0.6518 |     0.1568 |
| XGBoost              |     0.8042 |     0.1336 |

At 300s with Extended 21, every model failed to generalize cleanly to the extreme-load ID16 run. Random Forest had ID16 MAE 6.1508 and ID22 MAE 1.0804; XGBoost had 7.9754 and 1.3359 respectively. The full per-run table is `results/freetwinev/phase5_results_per_run.csv`.

## Comparison with Phase 4

Phase 4 Extended Ridge achieved 0.1598 MAE, 0.7228 RMSE, and 0.7890 R² at 60s. None of the Phase 5 tree models improved all three metrics. The lowest tree validation MAE was Gradient Boosting at 0.2911, while the CV-selected Random Forest was 0.3311 on validation. Therefore the nonlinear models do not displace Extended Ridge at 60s.

At 300s, Phase 4 Extended Ridge achieved 2.2629 MAE, 4.5423 RMSE, and 0.5021 R². None of the tree models improved all metrics. The best tree validation RMSE was Random Forest at 6.3872 and the best tree R² was 0.0155, both substantially below Ridge. Persistence remains a relevant comparator because it had lower MAE than the Phase 4 linear models at this horizon.

## Generalization gaps and failure cases

The central finding is a large CV-to-validation gap. For example, CV-selected Extended Random Forest has 300s CV MAE 1.3541 but pooled validation MAE 4.3577; its ID16 bias is +5.3500°C. Baseline tree models are worse: their 300s validation R² values range from -1.7537 to -3.5770. At 60s, Extended models are much more stable on ID22 than ID16, but still do not match Ridge pooled performance.

The ID16 errors are physically consistent with an out-of-distribution regime: approximately -120 A, 15°C coolant, and 0.5 L/min. ID22, approximately -80 A, 8°C coolant, and 2.4 L/min, is easier for most models. Positive 300s bias on ID16 indicates underestimation of long-horizon heating under severe load, not a reason to conceal the result.

## Absolute-temperature reconstruction

For each selected estimator, absolute predictions were reconstructed as `T_cell_max + predicted_DeltaT`. The persisted validation table reports `Val_Abs_MAE`, `Val_Abs_RMSE`, and `Val_Abs_R2`. Absolute MAE and RMSE equal the corresponding delta-temperature errors algebraically because the same current temperature is added to prediction and target residuals. Absolute R² is higher at short horizons because thermal autocorrelation explains much of absolute temperature variance; it must not be interpreted as equivalent dynamic forecasting skill.

## Feature importance and diagnostics

Impurity-based feature importance was recorded for all tree models; the XGBoost measure is its native gain-style `feature_importances_` output. Validation permutation importance is stored separately and is explicitly diagnostic, not used for feature selection. In Extended models, `power_gross`, `T_cell_max_diff_30s`, current terms, and thermal gradients frequently receive substantial importance, but these associations are not causal evidence.

Figures 21-30 under `results/freetwinev/figures/` provide actual-versus-predicted plots, residual distributions, residual-versus-target plots, per-run comparisons, CV-versus-validation comparisons, feature-importance heatmaps, error-over-time, and absolute-temperature reconstruction. The figures label horizons, feature sets, models, validation runs, and units where applicable.

## Leakage and reproducibility audit

`results/freetwinev/phase5_audit.json` is an automated document with overall status **PASS**. It confirms:

- ID18 and ID19 are absent from train and validation inputs and are never loaded by the Phase 5 pipeline.
- Train and validation run sets are disjoint.
- Features contain neither targets nor future/lead columns, with expected 9/21 feature counts.
- GroupKFold with `exp_id` is used for all CV and validation is not used for model selection.
- Missing-value checks pass on the consumed preprocessed inputs.
- Seed, library versions, grids, partitions, and result tables are persisted.

No row-level random splitting was introduced. The audit is a code/data-path audit; it does not claim that feature importance is causal or that validation performance estimates final-test performance.

## Limitations

Only ten training runs and two validation runs are available for cross-run generalization. GroupKFold estimates are consequently variable, as shown by the large fold standard deviations. The compact grids are intentionally not exhaustive. Validation remains a single locked operating-regime comparison and must not be used to retune the reported configurations. Ablation 23 was not run because the main Baseline/Extended experiment already addressed the prioritized Phase 5 scope; downstream coolant telemetry therefore remains an open controlled experiment.

## Conclusion and phase boundary

The measured results do not support the claim that nonlinear tree ensembles outperform Extended Ridge. Extended Ridge remains the strongest Phase 4 reference at both horizons under the pooled validation criteria, while Phase 5 demonstrates regime-sensitive degradation, especially at 300s on ID16. The evidence supports retaining the simpler linear model as the current model-selection reference and treating tree models as diagnostic alternatives rather than replacements.

Phase 5 is complete. The exact recommended next phase is **Phase 6: final evaluation on ID18 and ID19**, only after explicit authorization. No final-test evaluation was performed here.
