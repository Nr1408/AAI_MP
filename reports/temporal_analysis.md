# Temporal Prediction Investigation

## Scope and Data Boundary

This Phase D investigation uses only the first 400 chronological observations (08:00-14:39) for temporal feature analysis and persistence evaluation. The final 100 observations (14:40-16:19) are preserved as an untouched holdout and are not used for feature-engineering decisions, lag-correlation calculations, or baseline scoring.

## Current and Proposed Formulations

- Current Phase C formulation: `X(t) -> MaxTemp_C(t)`. The target and electrical/environmental features have the same timestamp; it is a simultaneous estimation task, not a future forecast.
- Persistence baseline: `MaxTemp_C(t) -> MaxTemp_C(t+h)`.
- Proposed causal forecast: `X(t), X(t-1), X(t-2), X(t-5), X(t-10) -> MaxTemp_C(t+5)`.

For the proposed forecast, `MaxTemp_C(t+5)` is the shifted target. It is never a feature. The current temperature is permitted only as `MaxTemp_C(t)` because it is available at the prediction timestamp; all other temperature and sensor lags are from t or earlier.

## Dataset Temporal Structure

The source contains 500 sequential, one-minute observations. For this investigation, only the 400-row development period was used: it remains evenly spaced at exactly one minute per row from 08:00 through 14:39. The source `MaxTemp_C` value is recorded at the same timestamp as the Phase C inputs; therefore, the original formulation is a simultaneous estimate, not a future forecast.

The 5-minute causal formulation is mechanically feasible. There are 395 development pairs for `X(t) -> MaxTemp_C(t+5)` when using current values only. With the full proposed lag set (t, t-1, t-2, t-5, t-10), 385 rows remain after removing the initial 10 rows and final 5 rows that lack required past/future values.

## Temperature Persistence

Autocorrelation was calculated as the Pearson correlation between `MaxTemp_C(t)` and `MaxTemp_C(t+h)` on the development period only.

| Horizon | Correlation with current `MaxTemp_C` |
| --- | ---: |
| 1 minute | 0.014 |
| 2 minutes | -0.046 |
| 5 minutes | -0.102 |
| 10 minutes | 0.104 |

These values are close to zero and do not indicate useful temperature persistence. In particular, current temperature does not predict the temperature five minutes later in this sequence.

## Causal Lag Relationships for a Five-Minute Horizon

For each permitted numeric feature at lags 0, 1, 2, 5, and 10 minutes, the absolute correlation with `MaxTemp_C(t+5)` was at most 0.119. The largest observed value was `InternalResistance_mOhm(t)` at -0.119. Current temperature itself had correlation -0.102 with the five-minute future target. This is weak exploratory association, not credible predictive evidence.

Categorical states also provide no clear result without a proper later validation. At time t, the mean future temperature ranged from 51.57 C to 54.22 C across charging stages and was 51.98 C/53.78 C for moisture `True`/`False`. These small descriptive differences are not sufficient to claim a useful future signal, especially in a single sequence.

## Persistence Baseline

The no-training persistence rule predicts `MaxTemp_C(t+h) = MaxTemp_C(t)`. It was evaluated using five expanding-window validation folds within the development period only.

| Horizon | MAE mean (C) | MAE std. | RMSE mean (C) | R2 mean |
| --- | ---: | ---: | ---: | ---: |
| 1 minute | 18.033 | 1.481 | 22.426 | -0.998 |
| 2 minutes | 18.382 | 1.094 | 22.846 | -1.079 |
| 5 minutes | 19.302 | 2.420 | 23.745 | -1.210 |
| 10 minutes | 17.266 | 1.634 | 21.557 | -0.841 |

The five-minute persistence baseline is poor and should be the minimum reference point for any later 5-minute ML forecast. At present, there is no evidence that a lagged-feature model would beat it.

## Comparison of Formulations

| Formulation | Evidence from this project | Interpretation |
| --- | --- | --- |
| A. `X(t) -> MaxTemp_C(t)` | Best Phase C validation result: Random Forest MAE 14.427 C, R2 -0.089 | Simultaneous inputs did not provide useful target signal after leakage exclusion. |
| B. `MaxTemp_C(t) -> MaxTemp_C(t+5)` | Development-only persistence MAE 19.302 C, R2 -1.210 | Current temperature has no usable five-minute persistence in this sequence. |
| C. Past/current features -> `MaxTemp_C(t+5)` | No lagged ML model fitted; all numeric causal lag correlations have absolute value at most 0.119 | There is no exploratory basis to claim that this future formulation works. |

## Why the Simultaneous Formulation Performed Poorly

The target-adjacent fields removed for leakage prevention (`MinTemp_C`, `AvgTemp_C`, and risk/status variables) contained the only strong observed associations with `MaxTemp_C`. The remaining approved sensor and operating variables had weak same-time correlations in Phase A, and their causal lagged values also have weak future-target correlations here. The near-zero temperature autocorrelation suggests the sequence may have been assembled from observations without physically continuous thermal evolution, despite its minute-by-minute timestamps. This is an inference from the observed pattern, not a claim about how the data was generated.

## Limitations and Recommendation

The dataset has only one 500-minute session, leaving 400 rows for development and no independent charging sessions for generalization checks. A five-minute horizon is technically constructible but is not supported by meaningful persistence or lagged-feature signal in this sample. Do not tune or fit a lagged ML pipeline yet.

The recommended next step is data provenance review: confirm whether adjacent records genuinely describe the same battery/cell under continuous charging, whether `MaxTemp_C` aligns causally with the timestamps, and whether another sequence contains slower thermal dynamics. Preserve the existing final 100-row holdout for any future, pre-specified temporal-model evaluation.
