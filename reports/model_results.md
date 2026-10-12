# Model Results

## Status

The experiment was run with `python -m src.train`. Random Forest was selected using development-period validation MAE only. The final 100 observations were not used for preprocessing fitting, model selection, or parameter choices.

## Methodology

- Target: `MaxTemp_C`.
- Features: the 11 approved Phase B fields implemented in `src/data_preprocessing.py`.
- Development period: first 400 chronological observations (08:00-14:39).
- Final test period: final 100 chronological observations (14:40-16:19).
- Validation: five-fold expanding-window `TimeSeriesSplit` on the development period.
- Metrics: MAE (average absolute temperature error), RMSE (error that emphasizes larger misses), and R2 (variance explained relative to predicting the validation mean).
- Pipelines: numerical median imputation plus scaling for linear regression; numerical median imputation without scaling for tree models; most-frequent categorical imputation and one-hot encoding for both.

## Initial Models

- Linear Regression baseline.
- Random Forest Regressor with 300 trees and `min_samples_leaf=2`.
- XGBoost Regressor with fixed, conservative initial settings (`n_estimators=300`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`).

## Feature Set Used

The models used these Phase B-approved fields: `PackVoltage_V`, `CellVoltage_V`, `ChargeCurrent_A`, `SOC_%`, `AmbientTemp_C`, `InternalResistance_mOhm`, `StateOfHealth_%`, `VibrationLevel_mg`, `Pressure_kPa`, `ChargingStage`, and `MoistureDetected`.

Excluded fields were not passed to any pipeline: `MinTemp_C`, `AvgTemp_C`, `TR_Probability`, `BMS_Status`, `EventFlag`, `Notes`, `Timestamp`, `ChargerID`, `CellID`, `DemandVoltage_V`, `DemandCurrent_A`, and `ChargePower_kW`.

## Development Validation Results

Values below are means across five expanding chronological validation folds. Lower MAE/RMSE is better; higher R2 is better. Standard deviations are across validation folds.

| Model | Train MAE | Validation MAE | Validation MAE std. | Train RMSE | Validation RMSE | Train R2 | Validation R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 5.884 | 14.427 | 0.779 | 6.906 | 16.566 | 0.800 | -0.089 |
| Linear Regression | 12.914 | 14.580 | 0.728 | 14.932 | 16.869 | 0.068 | -0.130 |
| XGBoost | 3.013 | 14.622 | 0.841 | 3.707 | 17.070 | 0.929 | -0.161 |

Random Forest had the lowest validation MAE by a narrow margin (0.153 C versus Linear Regression) and was therefore selected before final-test metrics were examined. This small gap is far below the fold-to-fold MAE variation, so it is not strong evidence that Random Forest is meaningfully better for this dataset.

## Final Chronological Test Results

All pipelines were then fitted once on the complete 400-row development period and evaluated on the fixed final 100 rows. These test metrics were not used to select a model.

| Model | MAE (C) | RMSE (C) | R2 |
| --- | ---: | ---: | ---: |
| Random Forest (selected by validation) | 14.275 | 16.564 | -0.106 |
| Linear Regression | 13.768 | 15.902 | -0.019 |
| XGBoost | 14.922 | 17.505 | -0.235 |

For context only, predicting the development-period mean temperature for every final-test row gives MAE 13.696 C, RMSE 15.762 C, and R2 -0.001. Every tested model is slightly worse than this naive baseline on this test period. Therefore, no model should be described as practically predictive or superior on this dataset and test setup.

## Overfitting Observations

- XGBoost has the strongest overfitting signal: mean training MAE 3.013 C versus validation MAE 14.622 C, and mean training R2 0.929 versus validation R2 -0.161.
- Random Forest also shows substantial overfitting: training MAE 5.884 C versus validation MAE 14.427 C, and training R2 0.800 versus validation R2 -0.089.
- Linear Regression has a much smaller train/validation gap, but its validation and final-test R2 values remain negative. This indicates weak usable signal rather than good generalization.

## Figures and Explainability

- `results/figures/actual_vs_predicted.png` and `results/figures/residuals.png` show the selected Random Forest on the chronological final test period.
- `results/figures/feature_importance.png` contains fitted tree-model feature importances. These values are descriptive associations within this dataset, not causal effects and not a substitute for predictive validation.

## Limitations and Next Step

The dataset is a single 500-minute sequence and, after removing legitimate leakage fields, its approved real-time candidate features have little apparent association with `MaxTemp_C`. The test target distribution is similar to the development distribution (development mean 52.95 C and test mean 52.38 C), so the negative R2 values are not explained by a large mean shift alone.

Do not tune models against the current final test period. The recommended next step is to review the data-generation and timestamp-alignment process with the dataset owner: determine whether the candidate inputs are simultaneous readings, lagged readings, commands, or aggregates; then, if a causal prediction horizon can be established, evaluate strictly past-only lag features or collect additional time sequences. Preserve the existing final test period untouched for any future comparison.
