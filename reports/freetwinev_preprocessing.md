# Phase 3: FreeTwinEV Preprocessing & Data Pipeline Report

**Course:** Advanced AI Techniques (AAI)  
**Project:** AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)  
**Domain:** Climate, Energy, and Sustainability  
**Author:** AI Technical Assistant (Pair Programmer)  
**Date:** September 2026  
**Status:** Completed Phase 3 — Preprocessed Datasets & Audits Verified — Awaiting Approval Before Phase 4  

---

## 1. Objective

Phase 3 implements and verifies the complete **Preprocessing & Data Pipeline** for the **FreeTwinEV P45B 1s4p Battery Module Measurement** dataset.

The core objective is to transform raw physical laboratory time-series measurements into modeling-ready, leakage-controlled, partitioned datasets. The pipeline operationalizes the mathematical targets and feature schemas established in Phase 2 across two prediction horizons ($h = 60\text{ s}$ and $h = 300\text{ s}$) and three feature experiments (Baseline, Extended, Ablation).

> [!IMPORTANT]
> **Strict Phase 3 Constraints:**
> - **NO machine learning models** were trained in this phase.
> - **NO hyperparameters** were tuned.
> - **NO raw data files** or ZIP archives were modified.
> - Preprocessing scalers were fitted **strictly on the Training partition only**.
> - All Phase 0–2 artifacts and the rejected dataset audit trail remain completely untouched.

---

## 2. Exact Run-Level Split Protocol

Randomly shuffling time-series rows is scientifically invalid due to high temporal autocorrelation ($\Delta t = 5.0\text{ s}$, lag-1 autocorrelation $r \approx 0.9999$). Therefore, an **experiment-level (run-level) holdout split** is locked across all 14 physical runs.

### Split Designations
- **Training Set (10 runs, 71.4% of experiments):**  
  `ID01`, `ID02`, `ID11`, `ID12`, `ID13`, `ID14`, `ID15`, `ID17`, `ID20`, `ID21`  
  Covers diverse discharge levels (-20A to -120A), coolant setpoints (7°C to 25°C), and flow rates (0.1 to 1.5 L/min). Includes `ID20` as an empirical repeatability check of `ID11`.
- **Validation Set (2 runs, 14.3% of experiments):**  
  `ID16` (-120A extreme load, 15°C, 0.5 L/min medium flow) and `ID22` (-80A heavy load, 8°C chilled coolant, 2.4 L/min high flow).  
  Reserved strictly for model selection and hyperparameter validation in future phases.
- **Final Test Set (2 runs, 14.3% of experiments) — UNTOUCHED:**  
  `ID18` (-80A heavy load, 15°C, 0.1 L/min starved cooling) and `ID19` (-120A extreme load, 15°C, 0.1 L/min starved cooling with early thermal cutoff at 69.5°C).  
  These runs represent severe out-of-distribution thermal-stress conditions and remain strictly held out until final model evaluation.

### Split Manifest Table

| Exp ID | Directory | Split | Current / C-rate | Coolant Temp | Flow Rate | Raw Rows | Valid 60s Rows | Valid 300s Rows | Operational Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ID01** | `ID01_20251126_150908248` | **train** | -20A (~1.1C) | 25°C | 1.5 L/min | 2,413 | 2,401 | 2,353 | Light discharge, warm coolant |
| **ID02** | `ID02_20251127_141017875` | **train** | -40A (~2.2C) | 25°C | 1.5 L/min | 728 | 716 | 668 | Moderate discharge, warm coolant |
| **ID11** | `ID11_20251128_102057991` | **train** | -40A (~2.2C) | 15°C | 1.5 L/min | 1,071 | 1,059 | 1,011 | Standard baseline run |
| **ID12** | `ID12_20251128_134031152` | **train** | -80A (~4.4C) | 15°C | 1.5 L/min | 1,153 | 1,141 | 1,093 | Heavy discharge, high flow |
| **ID13** | `ID13_20251201_113728861` | **train** | -120A (~6.7C) | 15°C | 1.5 L/min | 3,012 | 3,000 | 2,952 | Extreme discharge, full cycles |
| **ID14** | `ID14_20251201_163543484` | **train** | -40A (~2.2C) | 15°C | 0.5 L/min | 904 | 892 | 844 | Moderate discharge, medium flow |
| **ID15** | `ID15_20251202_133030514` | **train** | -80A (~4.4C) | 15°C | 0.5 L/min | 1,117 | 1,105 | 1,057 | Heavy discharge, medium flow |
| **ID16** | `ID16_20251203_103357473` | **val** | -120A (~6.7C) | 15°C | 0.5 L/min | 2,915 | 2,903 | 2,855 | Validation: extreme load, medium flow |
| **ID17** | `ID17_20251204_111640041` | **train** | -40A (~2.2C) | 15°C | 0.1 L/min | 1,182 | 1,170 | 1,122 | Moderate discharge, starved cooling |
| **ID18** | `ID18_20251204_144331500` | **test** | -80A (~4.4C) | 15°C | 0.1 L/min | 1,254 | 1,242 | 1,194 | Test: heavy load, starved cooling |
| **ID19** | `ID19_20251205_120936377` | **test** | -120A (~6.7C) | 15°C | 0.1 L/min | 2,219 | 2,207 | 2,159 | Test: extreme load, starved cooling |
| **ID20** | `ID20_20251205_160840427` | **train** | -40A (~2.2C) | 15°C | 1.5 L/min | 1,224 | 1,212 | 1,164 | Repeatability check of ID11 |
| **ID21** | `ID21_20251209_144434492` | **train** | -40A (~2.2C) | 7°C | 1.5 L/min | 1,433 | 1,421 | 1,373 | Chilled coolant, moderate load |
| **ID22** | `ID22_20251210_144332381` | **val** | -80A (~4.4C) | 8°C | 2.4 L/min | 1,622 | 1,610 | 1,562 | Validation: chilled coolant, heavy load |
| **Total** | **14 Runs** | **All Unique** | **[-120A, +15A]** | **7°C–25°C** | **0.1–2.4 L/min** | **22,247** | **22,079** | **21,407** | **Zero split overlap confirmed** |

---

## 3. Modeling Targets

For every timestep $t$, the current maximum cell surface temperature across the 12 calibrated sensors is:
$$T_{\text{cell,max}}(t) = \max_{j \in \{1..12\}} \left( T_{\text{cell}, j}(t) - 273.15 \right)$$

The modeling datasets format both dynamic temperature rise and benchmark absolute targets:

### 1. Primary Operational Target (Horizon $h = 60\text{ s}$ / 12 steps)
- **Primary Delta Target:** `target_DeltaT_60s` $= T_{\text{cell,max}}(t + 12) - T_{\text{cell,max}}(t)$
- **Benchmark Absolute Target:** `target_T_cell_max_60s` $= T_{\text{cell,max}}(t + 12)$

### 2. Secondary Strategic Target (Horizon $h = 300\text{ s}$ / 60 steps)
- **Primary Delta Target:** `target_DeltaT_300s` $= T_{\text{cell,max}}(t + 60) - T_{\text{cell,max}}(t)$
- **Benchmark Absolute Target:** `target_T_cell_max_300s` $= T_{\text{cell,max}}(t + 60)$

### Target Scaling Decision:
> [!NOTE]
> **Targets are kept UNTOUCHED (unscaled in physical °C).**  
> Retaining targets in degrees Celsius allows direct physical interpretation of evaluation metrics (MAE and RMSE in °C), direct benchmarking against naive persistence ($\Delta T = 0$), and direct boundary checks against automotive safety thresholds (e.g., $45\text{ }^\circ\text{C}$, $60\text{ }^\circ\text{C}$, $70\text{ }^\circ\text{C}$).

---

## 4. The Three Feature Experiments

We define three distinct, non-overlapping feature spaces:

### Experiment A — Baseline (9 Core Features)
Compact physical telemetry standardly available in automotive thermal management:
1. `current` (A)
2. `voltage` (V)
3. `SOC` (%)
4. `liquid_flow` (L/min)
5. `temp_inlet_C` (°C)
6. `temp_ambient_C` (°C)
7. `T_cell_max` (°C)
8. `T_cell_spread` (°C)
9. `T_plate_mean` (°C)

### Experiment B — Extended (21 Features)
The 9 Core features plus 12 physically motivated derived features:
- **Joule Heating Proxies:** `current_squared` ($I^2$), `current_abs` ($|I|$), `power_gross` ($V \cdot I$)
- **Rolling Load History (Past Heating Memory):**
  - `current_mean_60s`, `current_mean_300s` (rolling means of current)
  - `current_sq_mean_60s`, `current_sq_mean_300s` (rolling means of $I^2$, proxy for cumulative heat energy)
- **Thermal Velocity & Conduction Gradients:**
  - `T_cell_max_diff_30s`, `T_cell_max_diff_60s` (backward-only temperature slopes $dT/dt$)
  - `T_cell_to_plate_diff` ($T_{\text{cell,max}} - T_{\text{plate,mean}}$, conduction driving force across TIM)
  - `T_cell_to_inlet_diff` ($T_{\text{cell,max}} - T_{\text{inlet}}$, total thermal gradient to fluid sink)
- **Cooling Interaction:** `flow_x_inlet_temp` ($F \cdot T_{\text{inlet}}$)

### Experiment C — Ablation (23 Features)
The 21 Extended features plus 2 downstream calorimetric features:
- `temp_outlet_C` (coolant exit temperature)
- `DeltaT_coolant_C` ($T_{\text{outlet}} - T_{\text{inlet}}$, sensible fluid temperature rise)

> [!CAUTION]
> **Ablation Boundary:** Experiment C is strictly designated as an ablation study to quantify whether downstream fluid calorimetric signals provide additional predictive value beyond boundary conditions. It is not part of the primary baseline feature set.

---

## 5. Preprocessing & Scaler Justification: StandardScaler vs. RobustScaler

We conducted an empirical distribution audit to select the primary transformation between `StandardScaler` and `RobustScaler`:

### The Near-Zero IQR Problem in RobustScaler
In laboratory battery experiments, substantial experimental time is spent in rest or open-circuit initialization (Steps 2, 3, 4, 6 in protocol). Consequently:
- For `current`: Median = -0.0038 A, 25th percentile = -0.0049 A, 75th percentile = -0.0026 A.
- The resulting Interquartile Range ($\text{IQR} = Q_{75} - Q_{25}$) is **$0.0023\text{ A}$** (micro-ampere DAQ noise floor).
- Under `RobustScaler` ($z = \frac{x - \text{median}}{\text{IQR}}$), an active discharge current of -120 A is divided by $0.0023\text{ A}$, producing an unphysically massive scaled value:
  $$z_{\text{current}} \approx \mathbf{-47,764}$$
  Similar catastrophic numerical inflation occurs for `power_gross` ($z \approx -38,880$), `current_mean_60s` ($z \approx -67,108$), and `current_mean_300s` ($z \approx -64,436$).

### Resolution: StandardScaler as Primary Pipeline Scaler
`StandardScaler` normalizes using the global standard deviation ($\sigma = 20.51\text{ A}$ for current), which reflects the true physical dynamic operating range:
- Under `StandardScaler`:
  - `current` scales smoothly to $z \in [-5.55, +1.06]$.
  - `voltage` scales to $z \in [-2.22, +1.04]$.
  - `T_cell_max` scales to $z \in [-1.55, +7.29]$.
  - `current_squared` scales to $z \in [-0.31, +9.36]$.
- On the Training partition: **$\text{Mean} = 0.0000$** and **$\text{Std} = 1.0000$** for every single feature.

> [!TIP]
> **Verdict:** `StandardScaler` is selected as the primary scaler for Phase 3 because it avoids the near-zero IQR collapse of battery resting periods and produces stable, well-conditioned inputs across all physical regimes.

---

## 6. Missing Values & Horizon NaN Handling

The dataset exhibits exceptional raw integrity with zero missing values in raw features:
1. **Raw Feature NaNs:** Exactly **0 missing values (0.00%)** across all 22,247 observations.
2. **Terminal Target NaNs:** Target shifting forward into the future naturally produces NaNs at the end of each physical run where future observations do not exist:
   - For $h = 60\text{ s}$ (12 steps): Exactly the final 12 rows of each run produce target NaNs ($14 \times 12 = 168$ rows dropped).
   - For $h = 300\text{ s}$ (60 steps): Exactly the final 60 rows of each run produce target NaNs ($14 \times 60 = 840$ rows dropped).
3. **Run-Boundary Safety:** All rows with terminal target NaNs were dropped strictly per-run. Zero observations were imputed, and zero NaNs exist in any saved modeling dataset.

---

## 7. Automated Leakage Audit (All 6 Experiments PASSED)

An automated verification suite ([`results/freetwinev/preprocessed/leakage_audit_results.json`](file:///c:/AAI%20MP/results/freetwinev/preprocessed/leakage_audit_results.json)) evaluated all 6 dataset configurations across 6 strict criteria:

| Experiment Key | Target in Features? | Future Columns in X? | Feature NaNs | Target NaNs | Run ID Overlap | Scaler Fitted on Train Only? | Overall Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_60s` | **False** (Pass) | **False** (Pass) | **0** (Pass) | **0** (Pass) | **0 runs** (Pass) | **9 of 9** (Pass) | **PASS** |
| `extended_60s` | **False** (Pass) | **False** (Pass) | **0** (Pass) | **0** (Pass) | **0 runs** (Pass) | **21 of 21** (Pass) | **PASS** |
| `ablation_60s` | **False** (Pass) | **False** (Pass) | **0** (Pass) | **0** (Pass) | **0 runs** (Pass) | **23 of 23** (Pass) | **PASS** |
| `baseline_300s` | **False** (Pass) | **False** (Pass) | **0** (Pass) | **0** (Pass) | **0 runs** (Pass) | **9 of 9** (Pass) | **PASS** |
| `extended_300s` | **False** (Pass) | **False** (Pass) | **0** (Pass) | **0** (Pass) | **0 runs** (Pass) | **21 of 21** (Pass) | **PASS** |
| `ablation_300s` | **False** (Pass) | **False** (Pass) | **0** (Pass) | **0** (Pass) | **0 runs** (Pass) | **23 of 23** (Pass) | **PASS** |

---

## 8. Distribution Shift Audit on Stress Tests (ID18, ID19)

Runs `ID18` (-80A, 0.1 L/min) and `ID19` (-120A, 0.1 L/min) in the final test set are intentionally designed as **starved-cooling thermal-stress tests**. We audited the distribution shift between Train, Validation, and Test:

### Key Shift Findings:
1. **Thermal Non-Uniformity (`T_cell_spread`):**
   - Training Set Range: $[0.31\text{ }^\circ\text{C}, 26.52\text{ }^\circ\text{C}]$
   - Validation Set Range: $[0.39\text{ }^\circ\text{C}, 26.10\text{ }^\circ\text{C}]$
   - **Test Set Range:** $[0.51\text{ }^\circ\text{C}, \mathbf{29.27\text{ }^\circ\text{C}}]$
   - *Audit finding:* The maximum inter-cell thermal spread in the test set exceeds the training envelope by $+2.75\text{ }^\circ\text{C}$, caused by severe heat stagnation under 0.1 L/min starved cooling.
2. **Coolant Flow Rate (`liquid_flow`):**
   - Training Set Mean: $1.12\text{ L/min}$ (Range: $[0.00, 86.27\text{ L/min}]$)
   - Validation Set Mean: $1.17\text{ L/min}$ (Range: $[0.00, 90.94\text{ L/min}]$)
   - **Test Set Mean:** $\mathbf{0.09\text{ L/min}}$ (Range: $[0.00, 0.17\text{ L/min}]$)
   - *Audit finding:* The test set operates continuously in the starved-flow regime ($0.1\text{ L/min}$), testing model generalization when active cooling is severely diminished.

These shifts are preserved intentionally as realistic automotive stress scenarios.

---

## 9. Final Modeling Dataset Dimensions

18 modeling datasets (unscaled physical units in `preprocessed/`, scaled variants in `preprocessed/scaled/`) were generated and saved:

| Horizon | Feature Experiment | Features | Train Rows | Val Rows | Test Rows | Total Labeled Rows |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **60s** | **Baseline** | 9 | 14,117 | 4,513 | 3,449 | **22,079** |
| **60s** | **Extended** | 21 | 14,117 | 4,513 | 3,449 | **22,079** |
| **60s** | **Ablation** | 23 | 14,117 | 4,513 | 3,449 | **22,079** |
| **300s** | **Baseline** | 9 | 13,637 | 4,417 | 3,353 | **21,407** |
| **300s** | **Extended** | 21 | 13,637 | 4,417 | 3,353 | **21,407** |
| **300s** | **Ablation** | 23 | 13,637 | 4,417 | 3,353 | **21,407** |

---

## 10. Persisted Preprocessing Artifacts

All artifacts are persisted under [`results/freetwinev/preprocessed/`](file:///c:/AAI%20MP/results/freetwinev/preprocessed/):
1. **Manifests & Metadata:**
   - [`split_manifest.csv`](file:///c:/AAI%20MP/results/freetwinev/preprocessed/split_manifest.csv)
   - [`feature_manifest.csv`](file:///c:/AAI%20MP/results/freetwinev/preprocessed/feature_manifest.csv) and [`feature_manifest.json`](file:///c:/AAI%20MP/results/freetwinev/preprocessed/feature_manifest.json)
   - [`preprocessing_config.json`](file:///c:/AAI%20MP/results/freetwinev/preprocessed/preprocessing_config.json)
2. **Audit Results:**
   - [`leakage_audit_results.json`](file:///c:/AAI%20MP/results/freetwinev/preprocessed/leakage_audit_results.json) (automated PASS/FAIL verification)
   - [`scaling_audit.csv`](file:///c:/AAI%20MP/results/freetwinev/preprocessed/scaling_audit.csv)
   - [`distribution_shift_audit.csv`](file:///c:/AAI%20MP/results/freetwinev/preprocessed/distribution_shift_audit.csv)
3. **Serialized Scalers (in `scalers/`):**
   - `scaler_baseline_60s.joblib`, `scaler_extended_60s.joblib`, `scaler_ablation_60s.joblib`
   - `scaler_baseline_300s.joblib`, `scaler_extended_300s.joblib`, `scaler_ablation_300s.joblib`
4. **Code & Notebook Artifacts:**
   - Reusable Library: [`src/freetwinev_preprocessing.py`](file:///c:/AAI%20MP/src/freetwinev_preprocessing.py)
   - Standalone Pipeline Runner: [`src/run_freetwinev_preprocessing.py`](file:///c:/AAI%20MP/src/run_freetwinev_preprocessing.py)
   - Interactive Notebook: [`notebooks/09_preprocessing_pipeline.ipynb`](file:///c:/AAI%20MP/notebooks/09_preprocessing_pipeline.ipynb)

---

## 11. Known Limitations

1. **Purge Transients in Flow Rate:** Flow rate in `ID21` and `ID22` exhibits brief initialization spikes (~86–91 L/min). Because `StandardScaler` was fitted on the 10 training runs (where max flow was 86.27 L/min), flow features are bounded in $z \in [-0.71, +56.6]$. These spikes are genuine DAQ initialization transients and are preserved without synthetic clipping.
2. **Quasi-Static Operating Points:** In individual physical runs, flow setpoints were held approximately constant. Dynamic rapid flow pulsing was not performed in the laboratory protocol.
3. **Module Surrogate Scope:** The test article is a 1S4P cylindrical cell module with an external bottom liquid cold plate, serving as a functional experimental surrogate for EV thermal management.

---

## 12. Verification & Validation Sign-Off

- [x] **Data Integrity:** Raw CSV files and ZIP archive untouched; 22,247 total observations accounted for.
- [x] **Split Integrity:** Exactly 10 train runs, 2 validation runs, and 2 test runs with zero overlap.
- [x] **Target Integrity:** Clean terminal NaN dropping per horizon; targets preserved in unscaled °C.
- [x] **Feature Integrity:** Baseline (9), Extended (21), and Ablation (23) features correctly mapped.
- [x] **Scaling Integrity:** Scalers fitted strictly on Training data only; val/test transformed using train statistics.
- [x] **Leakage Integrity:** All 6 automated leakage audits returned **PASS**.
- [x] **Reproducibility:** Pipeline is fully deterministic and rerun verified.

**Phase 3 Preprocessing & Data Pipeline is COMPLETE.** Awaiting approval to proceed to Phase 4 (Baseline Models & Benchmark Evaluation).
