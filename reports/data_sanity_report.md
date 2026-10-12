# Data Provenance and Sanity Report

## Scope and Boundary

This Phase E investigation examines only the first 400 chronological source rows (08:00-14:39). The final 100-row holdout is preserved and excluded from all diagnostic calculations, plots, and decisions. The original CSV was not modified, and no predictive model was trained.

## Local Documentation Inventory

The workspace contains the CSV and project artifacts created in earlier phases. No supplier README, data dictionary, metadata file, collection protocol, sensor specification, or source citation was provided locally. The CSV `Notes` column contains row-level operational/risk phrases, but it does not document dataset provenance, collection hardware, units, vehicle/battery identity, or whether rows come from one experiment.

## 1. Timestamp and Consecutive-Row Relationship

The 400 development rows are ordered at one-minute timestamps from 08:00 through 14:39. However, ordering by timestamp does not make them a continuous physical measurement series:

- `ChargerID` changes on 359 of 399 adjacent transitions (90.0%).
- `CellID` changes on 394 of 399 transitions (98.7%).
- There are 10 distinct chargers, 50 distinct cells, and 276 distinct charger-cell pairs in 400 rows.
- `ChargingStage` changes on 301 transitions (75.4%). `MoistureDetected`, `BMS_Status`, and `EventFlag` change on 206, 193, and 265 transitions respectively.
- Lag-one correlations are close to zero for all examined numerical fields: `MaxTemp_C` 0.014, `PackVoltage_V` 0.013, `ChargeCurrent_A` -0.033, `SOC_%` -0.010, and `AmbientTemp_C` -0.040.

The evidence does not support interpreting adjacent rows as successive observations of one battery/cell under one continuous charging process.

## 2. Temperature Behavior

For development rows, the signed one-minute `delta_MaxTemp_C = MaxTemp_C(t) - MaxTemp_C(t-1)` has mean 0.010 C, median -0.310 C, standard deviation 22.343 C, minimum -51.280 C, and maximum 53.630 C. Its absolute change has mean 17.988 C, median 14.880 C, 95th percentile 43.026 C, and 99th percentile 51.045 C.

| Absolute one-minute temperature change | Rows | Percent |
| --- | ---: | ---: |
| <1 C | 16 | 4.01% |
| 1-2 C | 17 | 4.26% |
| 2-5 C | 36 | 9.02% |
| 5-10 C | 67 | 16.79% |
| >10 C | 263 | 65.91% |

These are diagnostic bins only. A temperature trace with 65.91% of adjacent one-minute changes above 10 C, including jumps of about 50 C, does not behave like a strongly continuous thermal signal. This agrees with the near-zero temporal autocorrelation from Phase D.

## 3. Thermal Field Consistency

At the row level, all 400 development records satisfy `MinTemp_C <= AvgTemp_C <= MaxTemp_C`. The max-min spread ranges from 0.00 C to 5.00 C, with mean 2.333 C and median 2.230 C.

`AvgTemp_C` is equal to `(MinTemp_C + MaxTemp_C) / 2` for every row to floating-point precision (maximum absolute midpoint difference approximately `1.4e-14`). This means `AvgTemp_C` is arithmetically derived from the min and max fields in this file rather than behaving as an independent average measurement. The fields are internally consistent but this exact construction is an important provenance caveat.

## 4. Electrical Consistency

The supplied electrical values are internally coherent in one key arithmetic sense:

`ChargePower_kW` is almost exactly `PackVoltage_V * ChargeCurrent_A / 1000`.

Across the development period, their correlation is 0.99999997; mean absolute difference is 0.002467 kW and maximum absolute difference is 0.004994 kW. These discrepancies are compatible with rounding. Therefore `ChargePower_kW` is a derived/redundant field, not independent evidence of a physical measurement process.

`DemandVoltage_V - PackVoltage_V` and `DemandCurrent_A - ChargeCurrent_A` each vary roughly between -5 and +5 in their named units. Five raw `DemandCurrent_A` values in the complete CSV are negative; they were preserved and are not used as model features. The pack-to-cell voltage ratio ranges from 72.24 to 109.43 in the development rows. Without a documented pack configuration, this changing ratio cannot be deemed physically wrong, but it is inconsistent with treating every row as one fixed-pack topology.

## 5. Battery-Thermal Relationships

The diagnostic correlations with `MaxTemp_C` are weak for approved physical candidates: `ChargeCurrent_A` -0.099, `PackVoltage_V` -0.003, `CellVoltage_V` 0.007, `SOC_%` 0.002, `AmbientTemp_C` -0.017, `InternalResistance_mOhm` 0.104, `StateOfHealth_%` 0.006, `ChargePower_kW` -0.104, `Pressure_kPa` 0.063, and `VibrationLevel_mg` 0.023.

Charging-stage means are narrowly grouped (51.94-53.67 C) relative to their approximately 16 C within-stage standard deviations. Moisture-state means are 53.81 C (`False`) and 52.16 C (`True`). These descriptive relationships do not establish a reliable physical temperature mechanism.

`TR_Probability` correlates 0.725 with `MaxTemp_C`. `EventFlag` temperature means are strongly ordered (`Watch` 30.05 C, `Warning` 41.86 C, `Alarm` 52.94 C, `Runaway` 66.74 C). This is evidence that the risk fields are associated with or potentially derived from temperature, supporting their continued exclusion as leakage-prone fields. It does not prove the exact field-generation rule.

## 6. Categorical Structure and Session Assessment

The file mixes multiple chargers, cells, charging stages, and state labels at near-row-level frequency. The 276 distinct charger-cell combinations, 98.7% cell-ID transition rate, frequent stage changes, low lag-one correlations, and large thermal jumps show that `500 consecutive timestamps` does not establish `500 consecutive measurements from one physical battery process`.

The timestamps themselves are complete and uniform. The physical entity/session relationship is not. The best evidence-based interpretation is an ordered collection of disparate records or synthetic/generated scenarios, not a documented continuous battery thermal run. This is an inference from observed structure, not a claim about undisclosed data-generation methods.

## 7. Provenance Finding

No local documentation establishes source provenance. In particular, the supplied files do not state the data source, whether observations are real or simulated, sensor identities, pack configuration, unit conventions beyond column names, data-collection protocol, session boundaries, or target timestamp alignment. The `Notes` column provides row-level phrases but no dataset-level provenance.

## 8. Scientific Suitability Assessment

**Classification: C. Insufficiently documented / questionable for the intended EV thermal forecasting use case.**

The file is numerically complete and contains internally consistent derived fields, so it can still be useful for demonstrating CSV inspection, leakage detection, and an academic prototype workflow. It is not scientifically adequate evidence for genuine thermal forecasting: its sequence lacks continuity, entities switch almost every minute, future/past temperature relationships are near zero, and no provenance documents explain the structure.

## 9. Evidence Supporting the Classification

- 394 cell-ID changes and 359 charger-ID changes in 399 adjacent row transitions.
- 65.91% of one-minute max-temperature changes exceed 10 C; the maximum absolute jump is 53.63 C.
- `MaxTemp_C` lag-one correlation is only 0.014 and five-minute autocorrelation is -0.102.
- All approved physical candidate correlations with temperature are weak, while leakage-risk fields are strongly associated with it.
- `AvgTemp_C` is exactly constructed from min/max and `ChargePower_kW` is nearly exactly constructed from voltage/current.
- No provenance or collection documentation is available locally.

## 10. Recommended Next Step

Do not tune models, build the dashboard, or add safety/recommendation layers based on this file. Request or locate documentation from the dataset provider that identifies whether data are measured or simulated, entity/session keys, sampling protocol, pack topology, sensor definitions, and causal timestamp alignment. If the project later adopts another dataset, choose it deliberately after that review; do not merge, download, or replace data automatically. Preserve the current CSV and final 100-row holdout unchanged as an auditable record of this diagnosis.
