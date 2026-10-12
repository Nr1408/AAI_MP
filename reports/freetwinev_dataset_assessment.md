# Phase 0: FreeTwinEV Dataset Assessment Report

**Course:** Advanced AI Techniques (AAI)  
**Project:** AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)  
**Domain:** Climate, Energy, and Sustainability  
**Author:** AI Technical Assistant (Handover Audit)  
**Date:** September 2026  
**Status:** Completed Assessment — Awaiting User Approval for Phase 1  

---

## Executive Summary

The previous candidate dataset (`EV_Battery_Charging_TR_Dataset_with_Notes (1).csv`) was formally rejected (Category C) due to target leakage (midpoint `AvgTemp_C`), non-physical row-to-row jumping across unrelated cells/chargers (mean 1-minute delta of 18.0 °C; lag-1 autocorrelation 0.014), and failure of ML models to beat a naive baseline.

This assessment investigates the replacement candidate: **FreeTwinEV P45B 1s4p Battery Module Measurement dataset** (`1s4p Battery Module Measurement.zip`).

### Verdict: **CATEGORY A — Strong Candidate for Battery Thermal AI/ML Modeling**

| Dimension | Assessment | Evidence |
| :--- | :---: | :--- |
| **1. Provenance** | **Pass** | EU Horizon Europe FreeTwinEV research project; Slovak University of Technology (STU Bratislava); Milan Plzák; Zenodo release under CC BY 4.0. |
| **2. Documentation** | **Pass** | Comprehensive laboratory protocol (`Measurement_Protocol.xlsx`) with full Test Matrix, Sensor Layout, Material properties, and operating limits. |
| **3. Experimental Credibility** | **Pass** | Physical laboratory rig with 4 × Molicel INR21700-P45B cylindrical cells in 1s4p module with liquid cold plate and DAQ instrumentation. |
| **4. Temporal Continuity** | **Pass** | 22,667 rows across 14 runs; strictly uniform $\Delta t = 5.0\text{ s}$ sampling; zero negative time steps; zero duplicate timestamps; zero gaps $> 7.5\text{ s}$. |
| **5. Physical Thermal Diffusion** | **Pass** | Mean step-to-step $|\Delta T| = 0.009 - 0.084$ °C per 5s step; lag-1 autocorrelation $0.9998 - 0.9999$; smooth continuous thermal inertia. |
| **6. Temperature Channels** | **Pass** | 25 distinct physical sensors: 1 ambient, 2 coolant (inlet/outlet), 10 cold plate surface sensors, and 12 axial cell sensors (3 per cell). |
| **7. Electrical Channels** | **Pass** | Continuous physical measurements: Current (-120.5 A to +15.1 A, ~1.1C to ~6.7C), Voltage (2.52 V to 4.20 V), SOC (0% to 100%). |
| **8. Cooling Management** | **Pass** | Systematically varied coolant flow rates (0.1, 0.5, 1.5 L/min) and coolant inlet temperatures (5 °C, 15 °C, 25 °C) creating clear thermal impact. |
| **9. Validation Independence** | **Pass** | 14 distinct physical experiments covering a systematic Design of Experiments (DOE) matrix, enabling clean Run-Level Holdout without leakage. |
| **10. Leakage Profile** | **Pass** | Zero synthetic target leakage; zero post-event risk flags; raw physical sensor signals only. |

---

## Check A — Provenance and Experimental Setup

From `README.txt`, `Measurement_Protocol.xlsx`, document core metadata, and project citations:

- **Research Project:** FreeTwinEV (EU Horizon Europe Twinning initiative to advance research in battery digital twins and thermal management).
- **Institution:** Slovak University of Technology in Bratislava (STU), Faculty of Electrical Engineering and Information Technology (STU FEI).
- **Lead Researcher / Author:** Milan Plzák (STU FEI).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Battery Module Configuration:**
  - Configuration: 1S4P (4 cells connected in parallel, 1 series).
  - Cell Type: Molicel INR21700-P45B high-power cylindrical lithium-ion cells.
  - Nominal Cell Capacity: 4.5 Ah per cell (Total module capacity = 18.0 Ah).
  - Voltage Limits: 2.5 V (discharge cutoff) to 4.2 V (charge cutoff).
  - Thermal Safety Threshold: Maximum cell surface temperature cutoff $T_{\text{surf}} \ge 70\text{ }^\circ\text{C}$.
- **Cooling System:**
  - Single-channel aluminum liquid cold plate mounted to the cell module.
  - Coolant Fluid: Water circuit connected to a laboratory chiller/heater bath.
  - Thermal Interface Material (TIM): Arctic MX-6 thermal paste ($k \approx 0.5-2.0\text{ W/m}\cdot\text{K}$) applied between cells and cooling-channel PCB sensors.
- **DAQ Equipment & Channel Mapping:**
  - Voltage: Chan306
  - Current: Chan307
  - Flow rate: Chan308
  - Thermocouple / thermistor sensors: Channels Chan107 to Chan120.
  - Sampling Frequency: 0.2 Hz ($\Delta t = 5.0\text{ s}$).

---

## Check B — File Structure and Data Integrity

The dataset directory `AllData` contains 14 subdirectories, each corresponding to one complete experimental run. Each directory contains a CSV and an XLSX file of identical content.

### Header & Channel Specification (30 Columns)

| Channel Name in CSV | Raw Unit | Physical Domain | Description |
| :--- | :---: | :--- | :--- |
| `timesamp` *(sic)* | `s` | Time | Elapsed experiment time in seconds |
| `current` | `A` | Electrical | Module current (Negative = discharge, Positive = charge) |
| `voltage` | `V` | Electrical | Module terminal voltage |
| `SOC` | `%` | Electrical | State of Charge (0 to 100%) |
| `liquid_flow` | `L/min` | Cooling | Coolant volumetric flow rate |
| `temp_ambient` | `K` | Thermal | Ambient chamber temperature |
| `temp_inlet` | `K` | Cooling | Coolant temperature at cold plate inlet |
| `temp_outlet` | `K` | Cooling | Coolant temperature at cold plate outlet |
| `temp_TH1` – `temp_TH10` | `K` | Thermal | 10 independent sensors across the cold plate surface |
| `temp_cell-1_TH20`–`TH22`| `K` | Thermal | 3 axial sensors placed along Cell 1 height |
| `temp_cell-2_TH17`–`TH19`| `K` | Thermal | 3 axial sensors placed along Cell 2 height |
| `temp_cell-3_TH14`–`TH16`| `K` | Thermal | 3 axial sensors placed along Cell 3 height |
| `temp_cell-4_TH11`–`TH13`| `K` | Thermal | 3 axial sensors placed along Cell 4 height |

### Key Structural Findings:
1. **Row 0 contains units:** The first row in every CSV file contains unit abbreviations (`s`, `A`, `V`, `%`, `L/min`, `K`). Real numeric data begins at row index 1.
2. **Column naming typo:** Timestamp is spelled `timesamp` (missing the 't'). Preprocessing code must handle this explicitly.
3. **Temperature units:** All temperature channels are recorded in **Kelvin (K)**. Conversion to Celsius ($T_{\text{C}} = T_{\text{K}} - 273.15$) is required for analysis and reporting.
4. **Missing values:** 0 null values across all 22,667 rows and all 30 columns.
5. **Exact duplicate rows:** 0.

---

## Check C — Temporal Continuity and Experimental Runs Overview

Every run represents an unbroken, continuous physical laboratory test.

### Overview of All 14 Measurement Runs

| Exp ID | Directory Name | Rows | Duration (min) | Sampling $\Delta t_{\text{med}}$ | Gaps $> 7.5\text{s}$ | Min Current (A) | Max Current (A) | Mean Flow (L/min) | Mean $T_{\text{inlet}}$ (°C) | Peak Cell $T_{\max}$ (°C) | Max Cell Spread (°C) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ID01** | `ID01_20251126_150908248` | 2,413 | 201.0 | 5.00 s | 0 | -20.1 | +15.1 | 1.50 | 25.1 | 31.5 | 2.3 |
| **ID02** | `ID02_20251127_141017875` | 728 | 60.6 | 5.00 s | 0 | -40.1 | +15.1 | 1.51 | 25.1 | 38.4 | 5.2 |
| **ID11** | `ID11_20251128_102057991` | 1,071 | 89.2 | 5.00 s | 0 | -40.1 | 0.0 | 1.48 | 15.0 | 34.5 | 8.6 |
| **ID12** | `ID12_20251128_134031152` | 1,153 | 96.0 | 5.00 s | 0 | -80.3 | 0.0 | 1.43 | 15.4 | 55.1 | 18.5 |
| **ID13** | `ID13_20251201_113728861` | 3,012 | 250.9 | 5.00 s | 0 | -120.5 | +15.1 | 0.98 | 17.1 | 73.0 | 26.5 |
| **ID14** | `ID14_20251201_163543484` | 904 | 75.3 | 5.00 s | 0 | -40.1 | 0.0 | 0.49 | 15.4 | 36.2 | 10.1 |
| **ID15** | `ID15_20251202_133030514` | 1,117 | 93.0 | 5.00 s | 0 | -80.3 | 0.0 | 0.48 | 15.6 | 61.1 | 23.8 |
| **ID16** | `ID16_20251203_103357473` | 2,915 | 242.8 | 5.00 s | 0 | -120.4 | +15.1 | 0.49 | 15.2 | 70.7 | 26.1 |
| **ID17** | `ID17_20251204_111640041` | 1,182 | 98.4 | 5.00 s | 0 | -40.1 | 0.0 | 0.09 | 16.2 | 40.4 | 13.5 |
| **ID18** | `ID18_20251204_144331500` | 1,254 | 104.4 | 5.00 s | 0 | -80.3 | 0.0 | 0.09 | 16.0 | 68.1 | 29.3 |
| **ID19** | `ID19_20251205_120936377` | 2,219 | 184.8 | 5.00 s | 0 | -120.5 | +15.1 | 0.09 | 15.8 | 69.5 | 22.5 |
| **ID20** | `ID20_20251205_160840427` | 1,224 | 101.9 | 5.00 s | 0 | -40.1 | 0.0 | 1.36 | 15.8 | 40.9 | 15.1 |
| **ID21** | `ID21_20251209_144434492` | 1,433 | 119.3 | 5.00 s | 0 | -40.1 | 0.0 | 1.59 | 7.0 | 28.2 | 9.0 |
| **ID22** | `ID22_20251210_144332381` | 1,622 | 135.1 | 5.00 s | 0 | -80.3 | 0.0 | 2.39 | 7.9 | 48.0 | 17.2 |
| **Total / Summary** | **14 Runs** | **22,667** | **1,718.7 min (28.6 h)** | **5.00 s** | **0** | **-120.5** | **+15.1** | **0.09–2.39** | **7.0–25.1** | **28.2–73.0** | **2.3–29.3** |

### Continuity Comparison with Rejected Dataset

| Metric | Rejected Dataset | FreeTwinEV Dataset | Physical Assessment |
| :--- | :---: | :---: | :--- |
| **Sampling Interval** | 60.0 s (nominal) | 5.00 s (measured) | 12× finer temporal resolution |
| **Lag-1 Autocorrelation** | **0.014** (white noise) | **0.9998 to 0.9999** | Reflects genuine physical thermal mass |
| **Lag-12 (60s) Autocorr** | N/A | **0.973 to 0.992** | Smooth thermal inertia over 1 minute |
| **Lag-60 (300s) Autocorr** | -0.102 | **0.589 to 0.854** | Meaningful 5-minute forecasting potential |
| **Mean Absolute Step $\Delta T$** | **17.99 °C** | **0.009 to 0.084 °C** | 200× to 2000× smoother step transitions |
| **Maximum Step Jump** | **53.63 °C** | **1.86 °C** (under 120A pulse) | Realistic rate of temperature rise ($<0.37\text{ }^\circ\text{C/s}$) |

---

## Check D — Temperature Channels and Target Definition

FreeTwinEV includes **25 distinct physical temperature channels**:
1. **Ambient Temperature (1 sensor):** `temp_ambient`
2. **Coolant Circuit Temperatures (2 sensors):** `temp_inlet`, `temp_outlet`
3. **Cold Plate Surface Temperatures (10 sensors):** `temp_TH1` through `temp_TH10`
4. **Battery Cell Surface Temperatures (12 sensors):**
   - **Cell 1:** `temp_cell-1_TH20`, `temp_cell-1_TH21`, `temp_cell-1_TH22`
   - **Cell 2:** `temp_cell-2_TH17`, `temp_cell-2_TH18`, `temp_cell-2_TH19`
   - **Cell 3:** `temp_cell-3_TH14`, `temp_cell-3_TH15`, `temp_cell-3_TH16`
   - **Cell 4:** `temp_cell-4_TH11`, `temp_cell-4_TH12`, `temp_cell-4_TH13`

### Constructing the Prediction Target

In battery thermal management systems (BTMS), the primary control objective is preventing the maximum cell surface temperature from exceeding safety limits (e.g. 50 °C for degradation, 70 °C for thermal runaway onset).

We can construct:
$$T_{\text{cell,max}}(t) = \max_{j \in \{1..12\}} \left( T_{\text{cell}, j}(t) - 273.15 \right)$$

Because this maximum is computed over **12 genuine physical sensors** located on 4 distinct cells, it is not an arbitrary synthetic average. Furthermore:
- **Temperature non-uniformity ($\Delta T_{\text{module}}$):**
  $$\Delta T_{\text{spread}}(t) = \max_j T_{\text{cell}, j}(t) - \min_j T_{\text{cell}, j}(t)$$
  reaches up to **29.3 °C** in low-flow high-discharge tests (ID18: -80A at 0.1 L/min), providing an outstanding physical indicator for thermal stress and cooling failure.

---

## Check E & F — Electrical and Cooling Variables

### Electrical Measurements
- **Current (`current`):**
  - High-stress discharge pulses: -120 A (~6.67C), -80 A (~4.44C), -40 A (~2.22C), -20 A (~1.11C).
  - Rest phases: 0.0 A.
  - CC-CV charge: +15.1 A (~0.84C) up to 4.2 V.
- **Voltage (`voltage`):** Dynamic voltage response from 2.52 V to 4.20 V, matching physical cell polarization and IR drop.
- **State of Charge (`SOC`):** Continuously tracks from 100.0% down to cut-off (0.0% to 27.8%).

### Cooling Measurements
- **Volumetric Flow Rate (`liquid_flow`):**
  - High flow: ~1.5 L/min
  - Medium flow: ~0.5 L/min
  - Low flow (starved cooling): ~0.1 L/min
- **Inlet Temperature (`temp_inlet`):**
  - Chilled: ~7.0 °C (ID21, ID22)
  - Moderate: ~15.0–16.0 °C (ID11–ID20)
  - Ambient warm: ~25.1 °C (ID01, ID02)
- **Coolant Temperature Rise ($\Delta T_{\text{coolant}} = T_{\text{outlet}} - T_{\text{inlet}}$):**
  - Directly measures sensible heat absorbed by the cooling plate from the cells.

---

## Check G — Experiment Independence and Validation Strategy

### Systematic Design of Experiments (DOE)

At the standard 15 °C coolant setpoint, the dataset contains a complete **$3 \times 3$ Factorial Matrix**:

| Flow Rate | -40 A (~2.2C) | -80 A (~4.4C) | -120 A (~6.7C) |
| :---: | :---: | :---: | :---: |
| **1.5 L/min** | **ID11** (Peak 34.5°C) | **ID12** (Peak 55.1°C) | **ID13** (Peak 73.0°C) |
| **0.5 L/min** | **ID14** (Peak 36.2°C) | **ID15** (Peak 61.1°C) | **ID16** (Peak 70.7°C) |
| **0.1 L/min** | **ID17** (Peak 40.4°C) | **ID18** (Peak 68.1°C) | **ID19** (Peak 69.5°C) |

Additional runs provide:
- **25 °C Coolant:** ID01 (-20 A, 1.5 L/min), ID02 (-40 A, 1.5 L/min)
- **5 °C Chilled Coolant:** ID21 (-40 A, 1.5 L/min), ID22 (-80 A, 2.4 L/min)
- **Repeat Run:** ID20 (-40 A at 15 °C, 1.5 L/min) directly tests repeatability against ID11.

### Validation Strategy: Run-Level Holdout (Never Row Shuffle)

> [!CRITICAL]
> **Randomly shuffling rows across time would invalidate all scientific integrity.** Because $\Delta t = 5.0\text{ s}$ and lag-1 autocorrelation is 0.9999, row-level shuffling leaks adjacent points into the test set.

**Recommended Splitting Protocol:**
- **Training Set (9–10 runs, ~15,000 rows):** Cover varied C-rates and cooling conditions (e.g., ID01, ID02, ID11, ID12, ID13, ID14, ID15, ID17, ID20, ID21).
- **Validation Set (2 runs, ~3,500 rows):** For model selection and hyperparameter tuning (e.g., ID16: -120A at 0.5 L/min, ID22: -80A at chilled 5°C).
- **Final Holdout Test Set (2 runs, ~3,500 rows):** Strictly held out until final evaluation:
  - **ID18:** High thermal stress with starved cooling (-80A discharge at 0.1 L/min; peak 68.1 °C, 29.3 °C spread).
  - **ID19:** Extreme stress (-120A discharge at 0.1 L/min; thermal cut-off triggered at 69.5 °C).
- **Alternative:** **Leave-One-Run-Out (LORO) Cross-Validation** across the 14 experiments.

---

## Check H — Future Prediction Feasibility

With $\Delta t = 5\text{ s}$, battery thermal response exhibits clear physical time constants:
- **30 seconds ($h=6$ steps):** Fast thermal polarization.
- **60 seconds ($h=12$ steps, 1 min):** Tactical cooling response time.
- **120 seconds ($h=24$ steps, 2 min):** Operational cooling adjustment window.
- **300 seconds ($h=60$ steps, 5 min):** Strategic thermal management horizon.

Because autocorrelation remains high and gradually decays ($r = 0.58 - 0.85$ at 5 minutes), predicting:
$$X(t) \longrightarrow T_{\text{cell,max}}(t + h)$$
is physically feasible and scientifically well-posed. Models can learn how current load $I(t)$ generates heat while coolant flow $F(t)$ and inlet temperature $T_{\text{inlet}}(t)$ remove heat over time.

---

## Check I — Target Leakage Identification and Feature Exclusion Rules

To ensure complete scientific defensibility:

| Feature / Variable | Status at Time $t$ | Decision | Rationale |
| :--- | :---: | :---: | :--- |
| `timesamp` | Raw elapsed seconds | **Exclude** | Arbitrary experiment start offset; not a physical state. |
| Any sensor at $t+h$ | Future measurement | **STRICTLY EXCLUDED** | Target leakage. |
| $T_{\text{cell}, j}(t+h)$ | Future cell temperatures | **STRICTLY EXCLUDED** | The target itself or direct components of it. |
| $T_{\text{cell,max}}(t)$ | Current maximum cell temp | **Approved Input** | Baseline state for autoregressive / delta prediction $\Delta T(t+h)$. |
| $T_{\text{cell}, j}(t)$ | 12 cell surface sensors at $t$ | **Approved Input** | Module spatial temperature profile at decision time. |
| $T_{\text{plate}, k}(t)$ | 10 cold plate sensors at $t$ | **Approved Input** | Boundary condition on cold plate at decision time. |
| `current(t)` | Current load | **Approved Input** | Primary heat generation source ($I^2 R$). |
| `voltage(t)`, `SOC(t)` | Terminal voltage, SOC | **Approved Input** | Electrical operating state. |
| `liquid_flow(t)` | Coolant flow rate | **Approved Input** | Active cooling actuation state. |
| `temp_inlet(t)` | Coolant inlet temperature | **Approved Input** | Thermal sink boundary temperature. |
| `temp_outlet(t)` | Coolant outlet temperature | **Approved with Caution** | Can be included as current state or omitted to test purely predictive heat transfer. |
| `temp_ambient(t)` | Ambient chamber temperature | **Approved Input** | Environmental boundary condition. |

---

## Check J — Final Suitability Classification and Comparison

| Evaluation Criterion | Rejected Dataset | FreeTwinEV Dataset | Verdict |
| :--- | :--- | :--- | :---: |
| **1. Provenance** | Unknown / unverified synthetic script | EU Horizon FreeTwinEV, STU Bratislava | **Pass** |
| **2. Documentation** | None (only row risk phrases) | Detailed protocol workbook & test matrix | **Pass** |
| **3. Experimental Credibility** | Synthetic / generated | Physical 1s4p module test rig | **Pass** |
| **4. Temporal Continuity** | Broken (row transitions jumped cells) | 14 continuous physical time series | **Pass** |
| **5. Sampling Regularity** | 1-min nominal, non-continuous | Strict 5.0 s uniform DAQ sampling | **Pass** |
| **6. Target Integrity** | Midpoint leakage (`AvgTemp_C`) | 12 independent physical cell sensors | **Pass** |
| **7. Electrical Scope** | Incoherent current/voltage values | Controlled -120A to +15A cycles | **Pass** |
| **8. Cooling Interaction** | No cooling variables present | Flow rate & inlet temp systematically varied | **Pass** |
| **9. Multi-Experiment Diversity** | 0 (all 500 rows mixed) | 14 distinct runs with 3×3 DOE | **Pass** |
| **10. Suitability Classification** | **CATEGORY C (Rejected)** | **CATEGORY A (Approved Candidate)** | **SUITABLE** |

---

## Artifacts Generated in Phase 0

1. **Detailed Protocol Dump:**
   - [`results/freetwinev/protocol_readable.txt`](file:///c:/AAI%20MP/results/freetwinev/protocol_readable.txt)
   - [`results/freetwinev/protocol_parsed.json`](file:///c:/AAI%20MP/results/freetwinev/protocol_parsed.json)
2. **Runs Overview Table:**
   - [`results/freetwinev/runs_overview.csv`](file:///c:/AAI%20MP/results/freetwinev/runs_overview.csv)
   - [`results/freetwinev/runs_inspection_summary.json`](file:///c:/AAI%20MP/results/freetwinev/runs_inspection_summary.json)
3. **Diagnostic Figures:**
   - [`results/freetwinev/figures/01_runs_matrix_overview.png`](file:///c:/AAI%20MP/results/freetwinev/figures/01_runs_matrix_overview.png)
   - [`results/freetwinev/figures/02_representative_run_id13.png`](file:///c:/AAI%20MP/results/freetwinev/figures/02_representative_run_id13.png)
   - [`results/freetwinev/figures/03_physical_continuity_comparison.png`](file:///c:/AAI%20MP/results/freetwinev/figures/03_physical_continuity_comparison.png)
   - [`results/freetwinev/figures/04_autocorrelation_vs_horizon.png`](file:///c:/AAI%20MP/results/freetwinev/figures/04_autocorrelation_vs_horizon.png)
4. **Interactive Inspection Notebook:**
   - [`notebooks/06_freetwinev_inspection.ipynb`](file:///c:/AAI%20MP/notebooks/06_freetwinev_inspection.ipynb)
5. **Inspection Script:**
   - [`src/freetwinev_inspection.py`](file:///c:/AAI%20MP/src/freetwinev_inspection.py)

---

## Recommended Next Steps

With Phase 0 complete and FreeTwinEV verified as a genuine, defensible Category A experimental dataset, we are ready to proceed to:

1. **PHASE 1 — Exploratory Data Analysis (EDA):**
   - Detailed analysis of heat generation ($Q = I^2 R$) vs heat removal ($Q = \dot{m} C_p \Delta T$).
   - Thermal gradient mapping along cell axes and across module positions.
   - Non-linear cooling response curves across flow rates (0.1 vs 0.5 vs 1.5 L/min).
2. **PHASE 2 — Target & Feature Formulation:**
   - Formulate target $T_{\text{cell,max}}(t+h)$ and temperature rise $\Delta T(t+h) = T_{\text{cell,max}}(t+h) - T_{\text{cell,max}}(t)$ for candidate horizons (e.g., $h = 60\text{ s}$ and $h = 300\text{ s}$).
   - Construct rolling thermal inertia, moving averages, and heat generation features.
3. **PHASE 3 — Preprocessing & Train/Validation Design:**
   - Implement run-level split with strict out-of-experiment holdouts (ID18 and ID19).
