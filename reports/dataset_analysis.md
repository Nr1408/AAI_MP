# Dataset Inspection Report

## Scope

This Phase A report inspects `EV_Battery_Charging_TR_Dataset_with_Notes (1).csv` only. No data was modified, no synthetic observations were created, and no model was trained.

## Dataset Overview

- File: `EV_Battery_Charging_TR_Dataset_with_Notes (1).csv`
- Shape: 500 rows x 24 columns
- Exact duplicate rows: 0
- Missing values: 0 in every column (0.0%)
- Time coverage: 03 September 2025, 08:00 to 16:19, at one-minute intervals; timestamps parse successfully and are monotonically increasing.

## Columns and Data Types

| Column | Data type |
| --- | --- |
| Timestamp | string (parseable datetime) |
| ChargerID | string |
| CellID | string |
| ChargingStage | string |
| PackVoltage_V | float64 |
| CellVoltage_V | float64 |
| DemandVoltage_V | float64 |
| ChargeCurrent_A | float64 |
| DemandCurrent_A | float64 |
| SOC_% | float64 |
| MaxTemp_C | float64 (target) |
| MinTemp_C | float64 |
| AvgTemp_C | float64 |
| AmbientTemp_C | float64 |
| InternalResistance_mOhm | float64 |
| StateOfHealth_% | float64 |
| VibrationLevel_mg | float64 |
| MoistureDetected | bool |
| BMS_Status | string |
| ChargePower_kW | float64 |
| Pressure_kPa | float64 |
| TR_Probability | float64 |
| EventFlag | string |
| Notes | string |

## Numerical Summary

| Feature | Mean | Std. dev. | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| PackVoltage_V | 346.387 | 28.731 | 300.070 | 346.065 | 399.970 |
| CellVoltage_V | 3.903 | 0.169 | 3.603 | 3.906 | 4.199 |
| DemandVoltage_V | 346.391 | 28.728 | 295.600 | 345.920 | 402.560 |
| ChargeCurrent_A | 58.928 | 33.950 | 0.230 | 58.640 | 119.900 |
| DemandCurrent_A | 59.117 | 34.001 | -3.320 | 57.855 | 124.010 |
| SOC_% | 50.851 | 29.040 | 0.120 | 51.640 | 99.930 |
| MaxTemp_C | 52.837 | 15.874 | 25.090 | 52.940 | 79.920 |
| MinTemp_C | 50.478 | 15.950 | 20.370 | 50.650 | 79.370 |
| AvgTemp_C | 51.658 | 15.896 | 22.820 | 51.740 | 79.380 |
| AmbientTemp_C | 27.370 | 4.247 | 20.000 | 27.525 | 34.990 |
| InternalResistance_mOhm | 7.741 | 4.088 | 1.000 | 7.595 | 14.980 |
| StateOfHealth_% | 90.085 | 5.890 | 80.010 | 90.225 | 99.990 |
| VibrationLevel_mg | 24.801 | 14.733 | 0.090 | 25.375 | 49.970 |
| ChargePower_kW | 20.387 | 11.836 | 0.090 | 19.470 | 44.630 |
| Pressure_kPa | 105.019 | 8.807 | 90.070 | 105.480 | 119.910 |
| TR_Probability | 0.766 | 0.191 | 0.247 | 0.781 | 1.000 |

No numerical column has a value outside its conventional 1.5 x IQR bounds. This does not prove that all values are physically valid; it only means there are no statistical IQR outliers in this sample.

## Categorical Values

- `Timestamp`: 500 unique values; each occurs once.
- `ChargerID`: 10 values (`CH1` through `CH10`).
- `CellID`: 50 values (`CELL1` through `CELL50`).
- `ChargingStage`: `Handshake` (144), `Parameter_Config` (128), `Recharge` (123), `End_Of_Charge` (105).
- `MoistureDetected`: `True` (262), `False` (238).
- `BMS_Status`: `OK` (350), `Warning` (92), `Critical` (58).
- `EventFlag`: `Alarm` (179), `Runaway` (152), `Warning` (145), `Watch` (24).
- `Notes`: 16 distinct free-text combinations, including phrases such as `Critical thermal runaway risk`, `Rapid temp rise`, and `High internal resistance`.

## Target: `MaxTemp_C`

`MaxTemp_C` is present, numeric, complete, and has a useful spread: 25.09 to 79.92 C (mean 52.84 C, median 52.94 C, standard deviation 15.87 C, skewness -0.042). Its quartiles are 39.34 C, 52.94 C, and 65.85 C. There are 182 observations in 15-45 C, 82 in 45-55 C, and 236 above 55 C.

It is a reasonable target for the proposed prototype because it is directly measured in the dataset and varies substantially. The prevalence of temperatures above the prototype's 45 C operating-range ceiling should be described as a property of this dataset, not as a claim about fleet-level EV operation.

## Correlation With `MaxTemp_C`

| Numerical feature | Pearson correlation |
| --- | ---: |
| AvgTemp_C | 0.9990 |
| MinTemp_C | 0.9960 |
| TR_Probability | 0.7242 |
| Pressure_kPa | 0.0923 |
| InternalResistance_mOhm | 0.0721 |
| CellVoltage_V | 0.0186 |
| VibrationLevel_mg | 0.0180 |
| SOC_% | 0.0037 |
| DemandVoltage_V | -0.0074 |
| PackVoltage_V | -0.0084 |
| AmbientTemp_C | -0.0284 |
| StateOfHealth_% | -0.0357 |
| ChargeCurrent_A | -0.0849 |
| DemandCurrent_A | -0.0852 |
| ChargePower_kW | -0.0897 |

Correlation describes association only. It neither establishes a usable causal relationship nor makes a field available before a temperature prediction.

## Leakage Assessment and Feature Decision

The initial real-time model should use only features that are plausibly available at prediction time. The following candidates are suitable for a first conservative model: `PackVoltage_V`, `CellVoltage_V`, `DemandVoltage_V`, `ChargeCurrent_A`, `DemandCurrent_A`, `SOC_%`, `AmbientTemp_C`, `InternalResistance_mOhm`, `StateOfHealth_%`, `VibrationLevel_mg`, `MoistureDetected`, `ChargePower_kW`, `Pressure_kPa`, and optionally `ChargingStage`. Each is a direct operating, environmental, battery-health, or charge-state measurement. Their availability must still be confirmed against the actual data-collection sequence.

Do not use the following in the initial model:

| Field | Decision | Reason |
| --- | --- | --- |
| `MinTemp_C`, `AvgTemp_C` | Exclude | Directly related same-interval thermal outcomes and near-perfect correlation with `MaxTemp_C`; strong target leakage. |
| `TR_Probability` | Exclude | Likely a derived thermal-risk score; correlation 0.7242 and not demonstrably known before prediction. |
| `BMS_Status`, `EventFlag`, `Notes` | Exclude | Risk/status labels and text mention thermal events; likely assigned after or using temperature. |
| `Timestamp` | Do not use raw | Every value is unique and rows are chronological. It is useful for time-aware splitting, but raw timestamp would encourage memorization. |
| `ChargerID`, `CellID` | Exclude initially | Identifiers can enable memorization and do not represent direct thermal conditions. Reconsider only with a defined generalization policy. |

## Data Quality Observations

- The file is complete and has no exact duplicate records.
- `DemandCurrent_A` has a minimum of -3.32 A. This may be valid measurement noise or a sign convention, but it needs domain confirmation before modelling.
- `EventFlag` has no `Normal` value despite 99 `Notes` entries labelled `Normal operation`; this is a label-semantics inconsistency worth documenting.
- The high proportion of `MoistureDetected=True` (52.4%), high-risk flags, and temperatures above 45 C may indicate a deliberately stress-oriented dataset. Do not generalize its distribution to production EVs.
- The data is ordered minute-by-minute, so a random train/test split could leak adjacent temporal patterns. A chronological split or time-series cross-validation should be evaluated in a later phase.

## Phase B: Timestamp Inspection and Preprocessing Plan

### Timestamp Findings

- Column name: `Timestamp`.
- Stored CSV type: string; it parses without errors to `datetime64[us]`.
- Chronological coverage: 03 September 2025 08:00:00 through 16:19:00.
- All 499 adjacent differences are exactly one minute. There are no gaps greater than one minute, duplicate timestamps, non-positive intervals, or out-of-order rows.
- The rows therefore appear to represent one continuous 500-minute sequence. This supports a chronological split and rules out a shuffled split for the initial evaluation.

### Initial Feature Selection

The first model should use a compact, conservative set of features that are plausibly measurable or known at the instant of prediction. This availability assumption must be confirmed against the eventual BMS/sensor data flow.

| Candidate | Decision | Availability and rationale |
| --- | --- | --- |
| `PackVoltage_V`, `CellVoltage_V` | Include | Direct electrical measurements plausibly available from the BMS before inference. |
| `ChargeCurrent_A` | Include | Direct charging-current measurement plausibly available at prediction time. |
| `SOC_%` | Include | BMS state estimate normally available before inference. |
| `AmbientTemp_C` | Include | Environmental measurement plausibly available from a sensor. |
| `InternalResistance_mOhm` | Include, with caveat | Useful health/thermal indicator if it is a contemporaneous BMS estimate rather than a value calculated after the event. |
| `StateOfHealth_%` | Include, with caveat | Battery-health estimate plausibly available from the BMS; should be versioned and known before inference. |
| `VibrationLevel_mg` | Include, with caveat | Direct sensor signal if the hardware provides it; retain as an optional deployment input. |
| `Pressure_kPa` | Include, with caveat | Direct environmental/enclosure signal if its sensor context is confirmed. |
| `ChargingStage` | Include | Charging-controller state known before inference; one-hot encode it. |
| `MoistureDetected` | Include | Boolean sensor state plausibly known before inference; one-hot encode it. |
| `DemandVoltage_V` | Exclude initially | Plausibly a pre-inference command/setpoint, but it is almost identical to `PackVoltage_V` (r=0.995). |
| `DemandCurrent_A` | Exclude initially, preserve | Plausibly a command/setpoint, but near-duplicates `ChargeCurrent_A` (r=0.997) and includes five small negative readings requiring domain confirmation. |
| `ChargePower_kW` | Exclude initially | Effectively derived from voltage and current (correlation 1.000 with `PackVoltage_V * ChargeCurrent_A / 1000`); avoid redundant input. |

### Excluded Fields

`MinTemp_C`, `AvgTemp_C`, `TR_Probability`, `BMS_Status`, `EventFlag`, and `Notes` remain excluded because they are thermal outcomes, risk-derived fields, status/event labels, or text that describes thermal events. `Timestamp` controls ordering only; raw time is excluded to avoid memorization. `ChargerID` and `CellID` are identifiers, excluded initially to avoid learning device-specific identity rather than physical conditions.

### Preprocessing Pipeline

- Parse `Timestamp` strictly and preserve source row order.
- Do not remove rows: the dataset has no missing values and no duplicates.
- Preserve the five negative `DemandCurrent_A` readings exactly as recorded. They occur at low `ChargeCurrent_A` (0.23-1.66 A) across `Handshake`, `Recharge`, and `Parameter_Config`, so there is insufficient justification to alter them.
- Use median imputation for selected numerical fields and most-frequent imputation plus one-hot encoding for `ChargingStage` and `MoistureDetected`. The source has no missing values; these steps make a later inference pipeline robust.
- Fit all preprocessing steps on training data only. Use scaling only in the later linear-regression pipeline; tree models should use the unscaled numerical transformer.
- Do not remove IQR outliers automatically; none were detected, and statistical flags are not a domain-validity rule.

### Proposed Train/Test Split

Use the first 400 observations (80%; 08:00-14:39) for training and the final 100 observations (20%; 14:40-16:19) as the untouched test period. Do not shuffle. Later cross-validation should use an expanding-window time-series splitter inside the training period, not random K-fold CV.

### EDA Figures

The preprocessing module creates these read-only visualizations in `results/figures/`: target distribution, maximum temperature over time, correlations for conservative numeric inputs, and temperature by charging stage/moisture state.

## Suitability and Recommended Next Step

The dataset is suitable for an academic temperature-prediction prototype, provided that leakage-prone thermal and risk fields are excluded and its narrow, single-session time coverage is stated as a limitation. The Phase B pipeline and EDA figures are now prepared, but neither a baseline nor a final model has been trained. Confirm the project direction before fitting any preprocessing transformer or model.
