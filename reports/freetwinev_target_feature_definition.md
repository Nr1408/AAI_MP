# Phase 2: FreeTwinEV Target & Feature Definition Report

**Course:** Advanced AI Techniques (AAI)  
**Project:** AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)  
**Domain:** Climate, Energy, and Sustainability  
**Author:** AI Technical Assistant (Pair Programmer)  
**Date:** September 2026  
**Status:** Completed Phase 2 — Awaiting Approval Before Phase 3 (Preprocessing & Data Pipeline)  

---

## 1. Objective

The objective of Phase 2 is to establish the formal mathematical definitions, causality constraints, feature taxonomy, and run-boundary enforcement mechanisms for the **FreeTwinEV P45B 1s4p Battery Module Measurement** dataset.

The core scientific problem is formulated as:
$$\text{Predicting future battery thermal states using only information available at decision time } t$$
to support a downstream, transparent thermal-risk monitoring and cooling-recommendation layer.

> [!IMPORTANT]
> **Phase 2 Boundary Rules:**
> - **NO machine learning models** were trained.
> - **NO hyperparameters** were tuned.
> - **NO raw data files** or ZIP archives were modified.
> - The audit trails for Phase 0, Phase 1, and the rejected dataset remain intact.

---

## 2. Target Definitions

In battery thermal management, the critical operational and safety hazard is the **localized peak cell surface temperature** (hotspot) rather than an arbitrary pack average.

### Base Thermal Envelope at Time $t$
Given the 12 calibrated physical cell surface sensors (converted from Kelvin to Celsius: $T_{\text{C}} = T_{\text{K}} - 273.15$):
- Cell 1: `temp_cell-1_TH20`, `TH21`, `TH22`
- Cell 2: `temp_cell-2_TH17`, `TH18`, `TH19`
- Cell 3: `temp_cell-3_TH14`, `TH15`, `TH16`
- Cell 4: `temp_cell-4_TH11`, `TH12`, `TH13`

The module peak temperature and thermal non-uniformity at time $t$ are:
$$T_{\text{cell,max}}(t) = \max_{j \in \{1..12\}} T_{\text{cell}, j}(t)$$
$$T_{\text{cell,spread}}(t) = \max_{j \in \{1..12\}} T_{\text{cell}, j}(t) - \min_{j \in \{1..12\}} T_{\text{cell}, j}(t)$$

### Target Formulation Candidates

We formally define two candidate prediction formulations for a given prediction horizon $h$:

#### Formulation A: Absolute Future Peak Temperature
$$y_{\text{abs}}(t) = T_{\text{cell,max}}(t + h)$$
- **Physical Meaning:** Direct physical surface temperature (°C) of the hottest cell location $h$ seconds into the future.
- **Characteristics:** Directly maps to automotive safety thresholds (e.g., $50\text{ }^\circ\text{C}$ degradation limit, $70\text{ }^\circ\text{C}$ cutoff limit). However, due to strong thermal inertia ($r \approx 0.98$ at 60s), this target is dominated by the current temperature level $T_{\text{cell,max}}(t)$.

#### Formulation B: Future Temperature Rise (Delta-T)
$$y_{\Delta}(t) = \Delta T_h(t) = T_{\text{cell,max}}(t + h) - T_{\text{cell,max}}(t)$$
- **Physical Meaning:** The dynamic thermal change (°C) driven by electrical Joule heating ($I^2 R$) minus convective heat extraction ($\dot{Q}_{\text{cool}}$) over horizon $h$.
- **Characteristics:** Removes static baseline persistence. Isolates the dynamic heat generation and cooling physics. The persistence baseline prediction is identically zero ($\Delta T = 0$). Predicted absolute temperature is reconstructed via:
  $$\hat{T}_{\text{cell,max}}(t + h) = T_{\text{cell,max}}(t) + \hat{\Delta T}_h(t)$$

### Specific Target Variables by Horizon
At $\Delta t = 5.0\text{ seconds}$ per step:
1. **Primary Operational Horizon ($h = 60\text{ seconds} = 12\text{ steps}$):**
   - Absolute Target: `target_T_cell_max_60s` $= T_{\text{cell,max}}(t + 12)$
   - Delta Target: `target_DeltaT_60s` $= T_{\text{cell,max}}(t + 12) - T_{\text{cell,max}}(t)$
2. **Secondary Strategic Horizon ($h = 300\text{ seconds} = 60\text{ steps}$):**
   - Absolute Target: `target_T_cell_max_300s` $= T_{\text{cell,max}}(t + 60)$
   - Delta Target: `target_DeltaT_300s` $= T_{\text{cell,max}}(t + 60) - T_{\text{cell,max}}(t)$

---

## 3. Feature Taxonomy and Specification

Every feature is classified into a structured taxonomy documenting its source, physical units, availability at decision time $t$, mathematical transformation, backward window length, leakage status, and physical rationale.

| Feature Name | Category | Source Column(s) | Physical Unit | Time Availability | Transformation Formula | Window Length | Leakage Status | Physical Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| `current` | Core Electrical | `current` | A | Time $t$ | Identity ($I(t)$) | $t$ | Approved | Primary driver of ohmic Joule heating ($I^2 R$) and electrochemical reaction. |
| `voltage` | Core Electrical | `voltage` | V | Time $t$ | Identity ($V(t)$) | $t$ | Approved | Terminal potential; reflects internal polarization and cutoff boundary. |
| `SOC` | Core Electrical | `SOC` | % | Time $t$ | Identity ($SOC(t)$) | $t$ | Approved | Thermodynamic state; internal resistance and entropic heat depend on SOC. |
| `liquid_flow` | Core Cooling | `liquid_flow` | L/min | Time $t$ | Identity ($F(t)$) | $t$ | Approved | Primary cooling actuation; governs convection heat transfer coefficient $h(F)$. |
| `temp_inlet_C` | Core Cooling | `temp_inlet` | °C | Time $t$ | $T_{\text{inlet}} - 273.15$ | $t$ | Approved | Heat sink boundary temperature of coolant entering the cold plate. |
| `temp_ambient_C`| Core Environment| `temp_ambient` | °C | Time $t$ | $T_{\text{amb}} - 273.15$ | $t$ | Approved | Ambient chamber boundary for external natural convection and radiation. |
| `T_cell_max` | Core Thermal | 12 Cell Sensors | °C | Time $t$ | $\max(T_{\text{cell}, 1..12} - 273.15)$ | $t$ | Approved | Current peak hotspot temperature; base state for dynamic temperature change. |
| `T_cell_spread`| Core Thermal | 12 Cell Sensors | °C | Time $t$ | $\max(T_j) - \min(T_j)$ | $t$ | Approved | Current spatial temperature non-uniformity across the module. |
| `T_plate_mean` | Core Thermal | `temp_TH1`–`TH10`| °C | Time $t$ | $\text{mean}(T_{\text{TH}, 1..10} - 273.15)$| $t$ | Approved | Compact representation of the 10 cold plate surface sensors. |
| `current_squared`| Derived Load | `current` | $\text{A}^2$ | Time $t$ | $I(t)^2$ | $t$ | Approved | Direct non-linear proxy for instantaneous Joule heat generation rate. |
| `current_abs` | Derived Load | `current` | A | Time $t$ | $\|I(t)\|$ | $t$ | Approved | Current load magnitude independent of sign convention. |
| `power_gross` | Derived Load | `voltage`, `current`| W | Time $t$ | $V(t) \cdot I(t)$ | $t$ | Approved | Electrical power throughput ($P < 0$ during discharge, $P > 0$ during charge). |
| `current_mean_60s`| Derived History| `current` | A | Time $\le t$ | $\frac{1}{12}\sum_{k=0}^{11} I(t-k)$ | Past 60s | Approved | Rolling mean current over preceding 1 minute; filters high-frequency noise. |
| `current_mean_300s`| Derived History| `current`| A | Time $\le t$ | $\frac{1}{60}\sum_{k=0}^{59} I(t-k)$ | Past 300s | Approved | Sustained current load over preceding 5 minutes; matches thermal mass delay. |
| `current_sq_mean_60s`| Derived History| `current`| $\text{A}^2$ | Time $\le t$ | $\frac{1}{12}\sum_{k=0}^{11} I(t-k)^2$ | Past 60s | Approved | Integrated Joule heating accumulated in cell core over the past 1 minute. |
| `current_sq_mean_300s`| Derived History| `current`| $\text{A}^2$ | Time $\le t$ | $\frac{1}{60}\sum_{k=0}^{59} I(t-k)^2$ | Past 300s | Approved | Integrated Joule heating accumulated over the past 5 minutes. |
| `T_cell_max_diff_30s`| Derived History| `T_cell_max` | °C | Time $\le t$ | $T(t) - T(t-30\text{s})$ | Past 30s | Approved | Recent 30-second thermal velocity ($dT/dt$); indicates acceleration of heating. |
| `T_cell_max_diff_60s`| Derived History| `T_cell_max` | °C | Time $\le t$ | $T(t) - T(t-60\text{s})$ | Past 60s | Approved | Recent 60-second thermal velocity; indicates sustained thermal trend. |
| `T_cell_to_plate_diff`| Derived Gradient| `T_cell_max`, `T_plate_mean`| °C| Time $t$ | $T_{\text{cell,max}} - T_{\text{plate,mean}}$ | $t$ | Approved | Thermal conduction driving force across thermal paste and interface ($q \propto \Delta T$). |
| `T_cell_to_inlet_diff`| Derived Gradient| `T_cell_max`, `temp_inlet`| °C | Time $t$ | $T_{\text{cell,max}} - T_{\text{inlet}}$ | $t$ | Approved | Overall temperature potential difference between module core and cooling fluid. |
| `flow_x_inlet_temp`| Derived Interaction| `liquid_flow`, `temp_inlet`| $\text{L/min}\cdot^\circ\text{C}$| Time $t$| $F(t) \cdot T_{\text{inlet}}(t)$ | $t$| Approved | Non-linear interaction proxy for advective cooling capacity. |
| `temp_outlet_C` | Ablation Cooling| `temp_outlet` | °C | Time $t$ | $T_{\text{outlet}} - 273.15$ | $t$ | Class E (Ablation) | Downstream fluid exit temperature; measures sensible heat already extracted. |
| `DeltaT_coolant_C`| Ablation Cooling| `temp_outlet`, `temp_inlet`| °C | Time $t$ | $T_{\text{outlet}} - T_{\text{inlet}}$ | $t$ | Class E (Ablation) | Coolant temperature rise; calorimetric measure of past cooling heat removal. |

---

## 4. Core Baseline Feature Set

The **Core Baseline Feature Set** contains **9 features**:
1. `current` (A)
2. `voltage` (V)
3. `SOC` (%)
4. `liquid_flow` (L/min)
5. `temp_inlet_C` (°C)
6. `temp_ambient_C` (°C)
7. `T_cell_max` (°C)
8. `T_cell_spread` (°C)
9. `T_plate_mean` (°C)

### Design Rationale:
- **Low Dimensionality & High Interpretability:** Directly maps to standard automotive sensor telemetry (BMS voltage/current/SOC, thermal loop flow and chiller inlet temperature, module thermistors).
- **Physical Sufficiency:** Captures the electrical power state, heat sink boundary temperature, active cooling actuation, and current module thermal envelope.
- **Zero Collinearity Explosion:** Replaces 10 individual plate sensors with `T_plate_mean` (which have pairwise correlation $r > 0.98$) and summarizes 12 cell sensors into peak and spread.

---

## 5. Derived Feature Set

The **Optional Derived Feature Set** contains **12 physically motivated features**:
- **Electrical Heating Proxies (3):** `current_squared` ($I^2$), `current_abs` ($|I|$), `power_gross` ($V \cdot I$).
- **Rolling Load History (4):** `current_mean_60s`, `current_mean_300s`, `current_sq_mean_60s`, `current_sq_mean_300s`.
- **Thermal Gradients & Velocity (4):** `T_cell_max_diff_30s`, `T_cell_max_diff_60s`, `T_cell_to_plate_diff`, `T_cell_to_inlet_diff`.
- **Cooling Interaction (1):** `flow_x_inlet_temp` ($F \cdot T_{\text{inlet}}$).

Combined with the Core Baseline, this yields an **Extended Feature Set of 21 features**.

### Correlation Validation:
Empirical audit on all 22,247 rows reveals why derived features are scientifically essential for predicting $\Delta T_{60\text{s}}$:
- Raw `T_cell_max` correlation with $\Delta T_{60\text{s}}$: **-0.0812** (almost uncorrelated).
- `T_cell_max_diff_30s` correlation with $\Delta T_{60\text{s}}$: **+0.8269** (strong physical heating velocity signal).
- `current_squared` correlation with $\Delta T_{60\text{s}}$: **+0.6236** (direct Joule heat source signal).
- `power_gross` correlation with $\Delta T_{60\text{s}}$: **-0.5567** (direct electrical work transfer signal).

---

## 6. Excluded Features

The following variables are formally excluded from the core baseline feature set:

| Variable / Channel | Leakage Class | Rationale for Exclusion |
| :--- | :---: | :--- |
| `timesamp` *(sic)* | Administrative | Represents arbitrary elapsed experiment time from zero. Not an intrinsic physical state. (Preserved internally for ordering and run segmentation). |
| Any measurement at $t+h$ | Class B (Future) | Strict future leakage. Not available at decision time $t$. |
| Individual cell sensors at $t+h$ | Class C (Target) | Constituent components of the prediction target itself. |
| Raw sensors `temp_TH1`–`TH10` | Redundant | High mutual collinearity ($r > 0.98$). Adequately summarized by `T_plate_mean`. |
| 12 individual cell sensors at $t$ | Redundant | Summarized by `T_cell_max` and `T_cell_spread` to keep baseline compact and prevent overfitting. |
| `temp_outlet_C` & `DeltaT_coolant_C` | Class E (Ablation) | Downstream calorimetric measurement. Rarely instrumented on module-level EV packs. Excluded from core baseline; retained for ablation. |
| Synthetic risk / event flags | Leakage / Synthetic | None exist in FreeTwinEV (unlike the rejected dataset which had derived `TR_Probability`, `BMS_Status`, and midpoint `AvgTemp_C`). |

---

## 7. Leakage Audit

To guarantee mathematical defensibility:
1. **No Forward Information:** Every feature row at index $k$ (corresponding to time $t$) is computed strictly using indices $j \le k$.
2. **Rolling Windows:** All rolling means and differences use backward-looking windows ($[t - W, t]$). No centered or forward-looking rolling filters are permitted.
3. **No Target Contamination:** Future target columns (`target_T_cell_max_60s`, `target_DeltaT_60s`, etc.) are computed via forward shifting (`shift(-12)`, `shift(-60)`) and are never used as inputs to any feature calculation.

---

## 8. Run-Boundary Audit

A critical flaw in naive time-series processing is applying `.shift()` or `.rolling()` across a concatenated DataFrame, which causes values from the end of Experiment A to leak into the start of Experiment B.

In [`src/freetwinev_targets.py`](file:///c:/AAI%20MP/src/freetwinev_targets.py) and [`src/freetwinev_features.py`](file:///c:/AAI%20MP/src/freetwinev_features.py), target and feature computations are executed **strictly per experiment run**.

### Empirical Run-Boundary Verification Results
From [`results/freetwinev/run_boundary_audit.csv`](file:///c:/AAI%20MP/results/freetwinev/run_boundary_audit.csv):

| Exp ID | Total Rows | Target 60s Expected NaNs | Target 60s Observed NaNs | NaNs at Run End Only? | Target 300s Expected NaNs | Target 300s Observed NaNs | NaNs at Run End Only? | Feature NaNs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ID01 | 2,413 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| ID02 | 728 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| ID11 | 1,071 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| ID12 | 1,153 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| ID13 | 3,012 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| ID14 | 904 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| ID15 | 1,117 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| ID16 | 2,915 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| ID17 | 1,182 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| ID18 | 1,254 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| ID19 | 2,219 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| ID20 | 1,224 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| ID21 | 1,433 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| ID22 | 1,622 | 12 | 12 | **True** | 60 | 60 | **True** | **0** |
| **Total** | **22,247** | **168** | **168** | **All True** | **840** | **840** | **All True** | **0** |

- For $h = 60\text{s}$ (12 steps): Exactly $14 \times 12 = \mathbf{168\text{ NaNs}}$ occur, located strictly in the final 12 rows of each run. Valid rows = **22,079**.
- For $h = 300\text{s}$ (60 steps): Exactly $14 \times 60 = \mathbf{840\text{ NaNs}}$ occur, located strictly in the final 60 rows of each run. Valid rows = **21,407**.
- Feature NaNs across all 23 channels: Exactly **0**.

---

## 9. Horizon Analysis

From [`results/freetwinev/target_summary_stats.csv`](file:///c:/AAI%20MP/results/freetwinev/target_summary_stats.csv):

| Metric | Primary Horizon (60s / 12 steps) | Secondary Horizon (300s / 60 steps) | Physical Interpretation |
| :--- | :---: | :---: | :--- |
| **Valid Observations** | 22,079 | 21,407 | Full physical experiment coverage |
| **Mean Absolute $\Delta T$** | **0.5200 °C** | **2.3604 °C** | 4.5× larger thermal dynamic range at 300s |
| **Standard Deviation ($\Delta T$)** | 1.3712 °C | 5.6832 °C | Broad spread reflecting diverse cooling conditions |
| **Max Absolute $\Delta T$** | **+14.47 °C** | **+47.28 °C** | Extreme heating observed in starved-cooling runs |
| **95th Percentile $\|\Delta T\|$** | 2.7054 °C | 12.7928 °C | Substantial operational temperature change |
| **Autocorrelation of $T_{\text{max}}$** | 0.9853 | 0.7479 | At 300s, persistence baseline weakens significantly |

- **Primary Horizon (60s):** Aligns with thermal control reaction loops. Predicts rapid temperature acceleration during transient discharge spikes.
- **Secondary Horizon (300s):** Aligns with strategic thermal envelope management. Predicts cumulative battery heating across sustained discharge phases.

---

## 10. Final Phase 2 Recommendations for Phase 3

We recommend the following exact specifications for **Phase 3 (Preprocessing & Data Pipeline)**:

1. **Target Formulation:**
   - **Primary Model Target:** $\Delta T_{60}(t) = T_{\text{cell,max}}(t+60\text{s}) - T_{\text{cell,max}}(t)$
   - **Secondary Model Target:** $\Delta T_{300}(t) = T_{\text{cell,max}}(t+300\text{s}) - T_{\text{cell,max}}(t)$
   - Compute absolute future temperature as $\hat{T}(t+h) = T(t) + \Delta \hat{T}(t)$ for thermal risk classification.
2. **Feature Sets for Modeling Experiments:**
   - **Baseline Experiment:** 9 Core Features (`current`, `voltage`, `SOC`, `liquid_flow`, `temp_inlet_C`, `temp_ambient_C`, `T_cell_max`, `T_cell_spread`, `T_plate_mean`).
   - **Extended Experiment:** 21 Features (9 Core + 12 Derived non-linear and load history terms).
   - **Ablation Experiment:** Extended Features + `temp_outlet_C` and `DeltaT_coolant_C` (evaluating calorimetric contribution).
3. **Data Splitting Protocol (Run-Level Holdout):**
   - **Train Set (10 runs, 15,532 rows):** ID01, ID02, ID11, ID12, ID13, ID14, ID15, ID17, ID20, ID21.
   - **Validation Set (2 runs, 4,537 rows):** ID16 (-120A, 0.5 L/min), ID22 (-80A, chilled 8°C).
   - **Untouched Final Test Set (2 runs, 2,178 rows):** ID18 (-80A, 0.1 L/min), ID19 (-120A, 0.1 L/min).
4. **Scaler Strategy:** Fit standard scalers (e.g. RobustScaler or StandardScaler) strictly on the Training Set runs and apply transform to Validation and Test runs.

---

## Generated Artifacts

- **Target Utilities Module:** [`src/freetwinev_targets.py`](file:///c:/AAI%20MP/src/freetwinev_targets.py)
- **Feature Utilities Module:** [`src/freetwinev_features.py`](file:///c:/AAI%20MP/src/freetwinev_features.py)
- **Target & Feature Audit Script:** [`src/audit_targets_features.py`](file:///c:/AAI%20MP/src/audit_targets_features.py)
- **Interactive Phase 2 Notebook:** [`notebooks/08_target_feature_definition.ipynb`](file:///c:/AAI%20MP/notebooks/08_target_feature_definition.ipynb)
- **Feature Taxonomy Schema:** [`results/freetwinev/target_feature_schema.csv`](file:///c:/AAI%20MP/results/freetwinev/target_feature_schema.csv)
- **Run-Boundary Audit Table:** [`results/freetwinev/run_boundary_audit.csv`](file:///c:/AAI%20MP/results/freetwinev/run_boundary_audit.csv)
- **Target Descriptive Statistics:** [`results/freetwinev/target_summary_stats.csv`](file:///c:/AAI%20MP/results/freetwinev/target_summary_stats.csv)
- **Feature Descriptive Statistics:** [`results/freetwinev/feature_summary_stats.csv`](file:///c:/AAI%20MP/results/freetwinev/feature_summary_stats.csv)
- **Feature-Target Correlations:** [`results/freetwinev/feature_target_correlations.csv`](file:///c:/AAI%20MP/results/freetwinev/feature_target_correlations.csv)
