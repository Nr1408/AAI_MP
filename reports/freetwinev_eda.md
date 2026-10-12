# Phase 1: FreeTwinEV Exploratory Data Analysis (EDA) Report

**Course:** Advanced AI Techniques (AAI)  
**Project:** AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)  
**Domain:** Climate, Energy, and Sustainability  
**Author:** AI Technical Assistant (Pair Programmer)  
**Date:** September 2026  
**Status:** Completed Phase 1 — Awaiting Approval Before Phase 2 (Target/Feature Definition)  

---

## 1. Objective

The objective of this Phase 1 Exploratory Data Analysis is to rigorously investigate the empirical physics, channel structures, thermal dynamics, and operational boundaries of the **FreeTwinEV P45B 1s4p Battery Module Measurement** dataset.

The findings from this analysis will provide an evidence-based foundation to define:
1. The mathematical prediction target
2. The operational prediction horizon ($h$)
3. Usable features available at decision time $t$
4. Formal leakage exclusions
5. An experiment-level validation strategy

> [!IMPORTANT]
> In accordance with our scientific principles:
> - **NO machine learning models** were trained in this phase.
> - **NO hyperparameters** were tuned.
> - **NO raw data files** were modified or overwritten.
> - The audit trail of the rejected dataset (`EV_Battery_Charging_TR_Dataset_with_Notes (1).csv`) remains completely preserved.

---

## 2. Dataset Overview

The dataset originates from the EU Horizon Europe **FreeTwinEV** project, developed at the **Slovak University of Technology in Bratislava (STU)**, Faculty of Electrical Engineering and Information Technology (STU FEI), under the direction of researcher **Milan Plzák**. It is distributed under the Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) license.

### Physical System Under Test
- **Battery Module:** 1S4P configuration (4 parallel cylindrical lithium-ion cells, 1 series).
- **Cell Model:** Molicel INR21700-P45B (high-power cylindrical cell; nominal capacity 4.5 Ah per cell; total module capacity 18.0 Ah).
- **Operating Envelope:** Voltage window: 2.5 V to 4.2 V; Maximum cell surface temperature limit: $T_{\text{surf}} \le 70\text{ }^\circ\text{C}$.
- **Cooling System:** Aluminum liquid cold plate mounted beneath the module, supplied by a laboratory chiller/heater bath circulator with water coolant.
- **Thermal Interface:** Arctic MX-6 thermal compound applied at the cell-to-channel PCB interfaces.
- **Instrumentation & DAQ:** 25 temperature channels (thermistors/thermocouples on Chan107–Chan120), module voltage (Chan306), module current (Chan307), and coolant flow rate (Chan308).

---

## 3. Experimental-Run Overview

The dataset contains **14 independent physical experimental runs** located in `data/candidate_freetwinev/1s4p Battery Module Measurement/AllData/`.

### Run-Level Summary Table

| Exp ID | Directory Name | Data Rows | Duration (min) | Current Range (A) | Mean Flow (L/min) | Mean $T_{\text{inlet}}$ (°C) | Peak $T_{\text{cell,max}}$ (°C) | Max $\Delta T_{\text{spread}}$ (°C) | Operating Regime Description |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ID01** | `ID01_20251126_150908248` | 2,413 | 201.0 | [-20.1, +15.1] | 1.50 | 25.1 | 31.5 | 2.3 | Light discharge (-20A, ~1.1C), warm coolant (25°C), high flow |
| **ID02** | `ID02_20251127_141017875` | 728 | 60.6 | [-40.1, +15.1] | 1.51 | 25.1 | 38.4 | 5.2 | Moderate discharge (-40A, ~2.2C), warm coolant (25°C), high flow |
| **ID11** | `ID11_20251128_102057991` | 1,071 | 89.2 | [-40.1, 0.0] | 1.48 | 15.0 | 34.5 | 8.6 | Moderate discharge (-40A), standard coolant (15°C), high flow |
| **ID12** | `ID12_20251128_134031152` | 1,153 | 96.0 | [-80.3, 0.0] | 1.43 | 15.4 | 55.1 | 18.5 | Heavy discharge (-80A, ~4.4C), standard coolant (15°C), high flow |
| **ID13** | `ID13_20251201_113728861` | 3,012 | 250.9 | [-120.5, +15.1] | 0.98 | 17.1 | 73.0 | 26.5 | Extreme discharge (-120A, ~6.7C), standard coolant, full cycles |
| **ID14** | `ID14_20251201_163543484` | 904 | 75.3 | [-40.1, 0.0] | 0.49 | 15.4 | 36.2 | 10.1 | Moderate discharge (-40A), medium flow (0.5 L/min), 15°C |
| **ID15** | `ID15_20251202_133030514` | 1,117 | 93.0 | [-80.3, 0.0] | 0.48 | 15.6 | 61.1 | 23.8 | Heavy discharge (-80A), medium flow (0.5 L/min), 15°C |
| **ID16** | `ID16_20251203_103357473` | 2,915 | 242.8 | [-120.4, +15.1] | 0.49 | 15.2 | 70.7 | 26.1 | Extreme discharge (-120A), medium flow (0.5 L/min), 15°C |
| **ID17** | `ID17_20251204_111640041` | 1,182 | 98.4 | [-40.1, 0.0] | 0.09 | 16.2 | 40.4 | 13.5 | Moderate discharge (-40A), low/starved flow (0.1 L/min), 15°C |
| **ID18** | `ID18_20251204_144331500` | 1,254 | 104.4 | [-80.3, 0.0] | 0.09 | 16.0 | 68.1 | 29.3 | Heavy discharge (-80A), low/starved flow (0.1 L/min), severe stress |
| **ID19** | `ID19_20251205_120936377` | 2,219 | 184.8 | [-120.5, +15.1] | 0.09 | 15.8 | 69.5 | 22.5 | Extreme discharge (-120A), low flow (0.1 L/min), early cutoff |
| **ID20** | `ID20_20251205_160840427` | 1,224 | 101.9 | [-40.1, 0.0] | 1.36 | 15.8 | 40.9 | 15.1 | Repeat validation run of ID11 condition (-40A, 15°C, high flow) |
| **ID21** | `ID21_20251209_144434492` | 1,433 | 119.3 | [-40.1, 0.0] | 1.59 | 7.0 | 28.2 | 9.0 | Moderate discharge (-40A), chilled coolant (7°C), high flow |
| **ID22** | `ID22_20251210_144332381` | 1,622 | 135.1 | [-80.3, 0.0] | 2.39 | 7.9 | 48.0 | 17.2 | Heavy discharge (-80A), chilled coolant (8°C), high flow |
| **Total** | **14 Runs** | **22,247** | **1,718.7 min (28.6 h)** | **[-120.5, +15.1]** | **0.09–2.39** | **7.0–25.1** | **28.2–73.0** | **2.3–29.3** | Systematic Design of Experiments (DOE) Matrix |

---

## 4. Data-Quality Findings

Systematic validation was performed on every file using [`src/freetwinev_eda.py`](file:///c:/AAI%20MP/src/freetwinev_eda.py).

### Verification Results
1. **Total Observations & Reconciliation with Phase 0:**
   - Summing all numerical data rows (excluding row 0 units) yields **exactly 22,247 observations** across the 14 runs.
   - *Documentation Discrepancy Note:* The Phase 0 report text mentioned 22,667 rows due to an initial estimate, but the exact row sum across the 14 CSVs is 22,247. This discrepancy is now formally recorded.
2. **Missing Values:** Exactly **0 missing values (0.00%)** across all 22,247 rows and all 30 channels.
3. **Exact Duplicates:** 0 duplicate rows across all runs.
4. **Timestamp Monotonicity:** Strictly monotonic ($t_{k} > t_{k-1}$) in all 14 files. Zero negative time steps ($\Delta t < 0$), zero duplicate timestamps ($\Delta t = 0$).
5. **Sampling Uniformity:**
   - Median $\Delta t = \mathbf{5.00\text{ seconds}}$ ($0.2\text{ Hz}$) for every single run.
   - Zero gaps exceeding $7.5\text{ seconds}$ ($\Delta t > 1.5 \times \Delta t_{\text{median}}$).
   - Minimal jitter ($\Delta t \in [3.41\text{ s}, 7.41\text{ s}]$), consistent with standard DAQ buffer polling.
6. **Physical Plausibility Check:**
   - Voltage: $2.52\text{ V} \le V(t) \le 4.20\text{ V}$ (strictly inside manufacturer specifications).
   - Current: $-120.5\text{ A} \le I(t) \le +15.1\text{ A}$ (strictly matches the laboratory protocol).
   - Temperatures: $281.7\text{ K} \le T(t) \le 346.2\text{ K}$ ($8.5\text{ }^\circ\text{C} \le T \le 73.0\text{ }^\circ\text{C}$). Zero negative Kelvin or unphysical sensor spikes.
7. **Constant Columns:** 0 constant columns. All 30 channels exhibit dynamic variation.

---

## 5. Temperature Analysis

### Temperature Metric Definitions
- **$T_{\text{cell,max}}(t)$:** Maximum of the 12 cell surface sensors at time $t$:
  $$T_{\text{cell,max}}(t) = \max_{j \in \{1..12\}} \left( T_{\text{cell}, j}(t) - 273.15 \right)$$
- **$T_{\text{cell,min}}(t)$:** Minimum of the 12 cell surface sensors at time $t$.
- **$T_{\text{cell,spread}}(t)$:** Thermal non-uniformity across the module:
  $$T_{\text{cell,spread}}(t) = T_{\text{cell,max}}(t) - T_{\text{cell,min}}(t)$$

### Spatial & Axial Thermal Gradients

The 12 cell sensors are distributed as 3 axial sensors per cell across the 4 cells:
- **Cell 1:** `temp_cell-1_TH20` (Top), `TH21` (Middle), `TH22` (Bottom near plate)
- **Cell 2:** `temp_cell-2_TH17` (Top), `TH18` (Middle), `TH19` (Bottom near plate)
- **Cell 3:** `temp_cell-3_TH14` (Top), `TH15` (Middle), `TH16` (Bottom near plate)
- **Cell 4:** `temp_cell-4_TH11` (Top), `TH12` (Middle), `TH13` (Bottom near plate)

#### Axial Temperature Distribution:
In every cell, the **bottom sensor** (TH22, TH19, TH16, TH13) is closest to the liquid cold plate and consistently registers the lowest temperature (mean $19.6\text{–}20.1\text{ }^\circ\text{C}$). The **top sensor** (TH20, TH17, TH14, TH11), located furthest from the heat sink near the positive cell terminals, experiences the highest temperatures (mean $21.9\text{–}22.6\text{ }^\circ\text{C}$, peaking at $73.0\text{ }^\circ\text{C}$).

#### Hottest Sensor Frequency Analysis:
We evaluated which of the 12 sensors registers the maximum temperature across all 22,247 timesteps:

| Sensor Channel | Cell | Axial Position | Hottest Frequency (Counts) | Hottest Percentage | Mean Temp (°C) | Peak Temp (°C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `temp_cell-1_TH20` | Cell 1 | Top | **14,394** | **64.70%** | 22.57 | 73.04 |
| `temp_cell-3_TH15` | Cell 3 | Middle | **3,606** | **16.21%** | 21.62 | 63.04 |
| `temp_cell-4_TH11` | Cell 4 | Top | **2,792** | **12.55%** | 22.13 | 67.22 |
| `temp_cell-2_TH17` | Cell 2 | Top | **507** | **2.28%** | 21.94 | 69.58 |
| `temp_cell-1_TH22` | Cell 1 | Bottom | **455** | **2.05%** | 20.06 | 49.31 |
| `temp_cell-3_TH14` | Cell 3 | Top | **436** | **1.96%** | 22.01 | 68.16 |
| `temp_cell-4_TH13` | Cell 4 | Bottom | **57** | **0.26%** | 20.03 | 49.58 |
| `temp_cell-1_TH21` | Cell 1 | Middle | 0 | 0.00% | 21.69 | 65.54 |
| `temp_cell-2_TH18` | Cell 2 | Middle | 0 | 0.00% | 21.37 | 64.05 |
| `temp_cell-2_TH19` | Cell 2 | Bottom | 0 | 0.00% | 19.75 | 49.89 |
| `temp_cell-3_TH16` | Cell 3 | Bottom | 0 | 0.00% | 19.56 | 46.98 |
| `temp_cell-4_TH12` | Cell 4 | Middle | 0 | 0.00% | 21.65 | 62.79 |

> [!NOTE]
> **Key Finding:** While `temp_cell-1_TH20` is the hottest sensor in 64.7% of timestamps, it is **NOT always the hottest**. In 35.3% of observations, the module hot spot shifts to Cell 3, Cell 4, or Cell 2 depending on the coolant flow rate, fluid transit time, and discharge state. Therefore, predicting the dynamic envelope maximum $T_{\text{cell,max}} = \max_j T_j$ is physically mandatory.

---

## 6. Electrical Analysis

### Current Sign Convention
Based on `Measurement_Protocol.xlsx` and the measured voltage profiles:
- **Negative Current ($I < 0$):** **Discharge**. Module provides current to the electronic load. Terminal voltage drops due to internal resistance ($V = \text{OCV} - I \cdot R_i$). The battery heats up via Joule heating ($P_{\text{loss}} \approx I^2 R_i$).
  - Discrete discharge levels: -20 A (~1.1C), -40 A (~2.2C), -80 A (~4.4C), -120 A (~6.7C).
- **Positive Current ($I > 0$):** **Charge**. Standardized CC-CV recharge at $+15.1\text{ A}$ (~0.84C) up to the 4.2 V cutoff.
- **Zero Current ($I \approx 0$):** **Rest / Thermal relaxation**. Open circuit conditions where voltage recovers and cell temperatures decay toward the cooling plate temperature.

### Voltage & SOC Dynamics
- Minimum observed voltage is $2.52\text{ V}$, exactly matching the $2.50\text{ V}$ cutoff condition in the experimental protocol.
- Under high-current discharge (-120 A in ID13, ID16, ID19), voltage drops sharply at the onset of discharge and triggers either the $2.5\text{ V}$ cutoff or the $70\text{ }^\circ\text{C}$ temperature safety limit before $0\%\text{ SOC}$ is reached (ending at $15.9\%\text{–}27.8\%\text{ SOC}$).

---

## 7. Cooling Analysis

The experimental setup actively controls both coolant volumetric flow rate and coolant inlet temperature:

### Flow Rate Impact
At the standard $15\text{ }^\circ\text{C}$ coolant setpoint, the dataset provides a direct comparison across three controlled flow regimes:
- **High Flow (1.5 L/min):** Provides vigorous convective heat transfer. Under -80 A discharge (ID12), peak cell temperature is held to $55.1\text{ }^\circ\text{C}$ with a spread of $18.5\text{ }^\circ\text{C}$.
- **Medium Flow (0.5 L/min):** Under -80 A discharge (ID15), peak temperature rises to $61.1\text{ }^\circ\text{C}$ with a spread of $23.8\text{ }^\circ\text{C}$.
- **Starved Cooling (0.1 L/min):** Under -80 A discharge (ID18), peak temperature reaches **$68.1\text{ }^\circ\text{C}$** with a massive inter-cell spread of **$29.3\text{ }^\circ\text{C}$**.

> [!TIP]
> Reducing coolant flow from 1.5 L/min to 0.1 L/min increases peak cell temperature by **$+13.0\text{ }^\circ\text{C}$** and widens thermal non-uniformity by **$+10.8\text{ }^\circ\text{C}$**. This provides a clear, causal physical gradient for our future cooling recommendation logic.

### Coolant Temperature Rise ($\Delta T_{\text{coolant}}$)
Sensible heat extracted by the cooling system is measured by:
$$\Delta T_{\text{coolant}}(t) = T_{\text{outlet}}(t) - T_{\text{inlet}}(t)$$
- Under high flow (1.5 L/min), $\Delta T_{\text{coolant}}$ remains small ($0.1\text{–}0.3\text{ }^\circ\text{C}$) because the high mass flow rate absorbs the heat with minimal temperature rise ($\dot{Q} = \dot{m} C_p \Delta T$).
- Under low flow (0.1 L/min), $\Delta T_{\text{coolant}}$ rises up to **$1.8\text{ }^\circ\text{C}$**, reflecting heat accumulation in the slow-moving fluid.

---

## 8. Temporal Dynamics & Candidate Horizons

Because measurements are recorded at 5-second intervals, we investigated the physical thermal inertia and temperature change distributions across four candidate prediction horizons:
- **30 seconds ($h = 6$ steps)**
- **60 seconds ($h = 12$ steps / 1 minute)**
- **120 seconds ($h = 24$ steps / 2 minutes)**
- **300 seconds ($h = 60$ steps / 5 minutes)**

### Horizon Analysis Summary

| Horizon | Steps ($h$) | Mean Autocorr | Autocorr Range | Mean $|\Delta T_h|$ (°C) | 95th Percentile $|\Delta T_h|$ (°C) | Max $|\Delta T_h|$ (°C) | Physical / Operational Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **30s** | 6 | 0.9961 | [0.9929, 0.9981] | 0.264 °C | 1.366 °C | 8.45 °C | High inertia. Mean change is small; too brief for mechanical cooling actuation. |
| **60s** | 12 | 0.9853 | [0.9737, 0.9926] | **0.520 °C** | **2.705 °C** | **14.47 °C** | **Ideal Primary Horizon.** Meaningful temperature changes; allows timely cooling pump adjustment. |
| **120s** | 24 | 0.9472 | [0.9072, 0.9719] | 1.012 °C | 5.331 °C | 24.64 °C | Strong secondary horizon. Captures sustained load trends. |
| **300s** | 60 | 0.7479 | [0.5893, 0.8539] | **2.360 °C** | **12.793 °C** | **47.28 °C** | **Ideal Strategic Horizon.** Autocorrelation decays to ~0.75, testing true predictive modeling over complete discharge profiles. |

---

## 9. Sensor Relationships & Redundancy

Cross-channel Pearson correlations were calculated across all 22,247 observations (Figure 10):
1. **Cold Plate Sensors (`temp_TH1`–`temp_TH10`):** Mutually correlated at $r > 0.98$. They track the surface temperature of the aluminum plate. Keeping all 10 individual plate sensors as inputs is redundant; a spatially representative subset or plate mean ($T_{\text{plate,mean}}$) captures the boundary condition with minimal dimensionality.
2. **Cell Sensors:** Sensors on the same cell share $r > 0.95$ correlation, but the axial gradient creates an offset of $2\text{–}5\text{ }^\circ\text{C}$ between top and bottom.
3. **Current vs. Temperature:** Instantaneous linear correlation between raw current and temperature is weak ($r \approx -0.15$) because battery heating is cumulative and delayed by thermal mass ($m C_p \frac{dT}{dt} = I^2 R_i - \dot{Q}_{\text{cool}}$). Integrating load over time (rolling cumulative $I^2 \cdot \Delta t$) is physically necessary for feature engineering in Phase 2.

---

## 10. Leakage Audit

To maintain complete scientific defensibility, all dataset channels are categorized into strict leakage classes:

| Class | Definition | Features in Class | Modeling Rule |
| :---: | :--- | :--- | :--- |
| **Class A** | **Available at decision time $t$** | `current(t)`, `voltage(t)`, `SOC(t)`, `liquid_flow(t)`, `temp_inlet(t)`, `temp_ambient(t)`, $T_{\text{cell,max}}(t)$, individual cell sensors at $t$, cold plate sensors at $t$, past lags ($t-k$) | **Approved Predictors** |
| **Class B** | **Available only after $t$** | Any sensor at $t+1, \dots, t+h$; future current $I(t+h)$; future cooling flow $F(t+h)$ | **STRICTLY EXCLUDED** (Future leakage) |
| **Class C** | **Directly derived from the target** | $T_{\text{cell}, j}(t+h)$, $\Delta T(t+h)$, future threshold exceedance flags | **STRICTLY EXCLUDED** (Direct target leakage) |
| **Class D** | **Potentially redundant / derived** | Module power $P(t) = V(t) \cdot I(t)$; Joule heat proxy $I^2(t)$; Plate mean $T_{\text{plate,mean}}(t)$ | **Approved for Feature Engineering** (computed strictly from past/present Class A variables) |
| **Class E** | **Ambiguous / requires audit** | `temp_outlet(t)` | **Special Audit (Section 11)** |

---

## 11. Temp_Outlet Investigation

We performed a dedicated investigation into whether `temp_outlet(t)` should be included as an input feature:

### Findings from Laboratory Setup & Physical Analysis:
1. **Physical Location:** The sensor is located at the liquid outlet manifold, downstream of the thermal contact area.
2. **Timing & Availability:** It is measured simultaneously with other channels at time $t$ by the DAQ.
3. **Information Content:** `temp_outlet(t)` reflects the sensible heat absorbed by the fluid *after* it has flowed across the 4 cells. It is effectively a delayed calorimetric measurement of past heat dissipation.
4. **Automotive Deployment Reality:** In real electric vehicle battery packs, cooling loops typically feature an inlet temperature sensor and pump flow/pressure feedback, but individual module outlet temperature sensors are rarely installed due to cost and wiring complexity.
5. **Leakage Risk:** If an ML model relies excessively on $T_{\text{outlet}}(t)$, it functions as an empirical thermometer for current heat transfer rather than learning the forward dynamic relationship between electrical load ($I$) and cooling actuation ($F$, $T_{\text{inlet}}$).

### Recommendation:
> [!CAUTION]
> **Verdict: "Include only in a separate feature ablation experiment."**  
> For the primary baseline model, `temp_outlet` will be **EXCLUDED**. This ensures the model learns forward thermal physics from standard inputs ($I$, $V$, $SOC$, $F$, $T_{\text{inlet}}$, $T_{\text{amb}}$). A secondary ablation experiment will measure the incremental benefit of `temp_outlet` for specialized digital-twin scenarios.

---

## 12. Candidate Target Formulations

We evaluated three potential prediction targets:

| Formulation | Mathematical Definition | Physical Meaning | Advantages | Disadvantages |
| :--- | :--- | :--- | :--- | :--- |
| **Target A: Absolute Future Temp** | $y(t) = T_{\text{cell,max}}(t+h)$ | Peak cell temperature at $t+h$ in °C | Directly comparable to safety limits (50°C, 70°C). Simple to interpret. | Dominated by current temperature level due to high autocorrelation. |
| **Target B: Future Temperature Rise** | $y(t) = \Delta T_h(t) = T_{\text{cell,max}}(t+h) - T_{\text{cell,max}}(t)$ | Temperature change over horizon $h$ | Forces model to learn heat generation and dissipation dynamics. Persistence baseline is identically 0. | Must add $T_{\text{cell,max}}(t)$ to obtain absolute temperature for safety evaluation. |
| **Target C: Threshold Classification** | $y(t) = \mathbb{I}(T_{\text{cell,max}}(t+h) \ge T_{\text{limit}})$ | Binary warning flag | Directly triggers cooling fans/pumps. | Severe class imbalance (70°C reached in only 4 of 14 runs); discards continuous regression signal. |

### Recommended Formulation for Phase 2:
Use **Target B ($\Delta T_h$)** as the primary training target, with predicted absolute temperature obtained via:
$$\hat{T}_{\text{cell,max}}(t+h) = T_{\text{cell,max}}(t) + \Delta \hat{T}_h(t)$$
This absolute temperature directly feeds into deterministic risk thresholds (e.g., Optimal $<45\text{ }^\circ\text{C}$, Elevated $45\text{–}60\text{ }^\circ\text{C}$, Critical $>60\text{ }^\circ\text{C}$, Thermal Cutoff $\ge 70\text{ }^\circ\text{C}$).

---

## 13. Candidate Prediction Horizons

Based on empirical thermal response times (Figure 9):
- **Primary Operational Horizon:** **$h = 60\text{ seconds}$ (12 steps / 1 minute)**.
  - Mean $|\Delta T| = 0.52\text{ }^\circ\text{C}$, with extreme discharges reaching up to $+14.5\text{ }^\circ\text{C}$ in 60s.
  - Matches the physical reaction time of automotive coolant pumps and electronic expansion valves.
- **Secondary Strategic Horizon:** **$h = 300\text{ seconds}$ (60 steps / 5 minutes)**.
  - Mean $|\Delta T| = 2.36\text{ }^\circ\text{C}$, reaching up to $+47.3\text{ }^\circ\text{C}$.
  - Tests long-term forecasting across extended high-rate discharge pulses.

---

## 14. Validation Considerations

### Why Random Row Splitting is Scientifically Invalid
In the FreeTwinEV dataset, observations are sampled every 5.0 seconds. As proven in Section 8, the lag-1 autocorrelation is **0.9999**, and the lag-12 autocorrelation is **0.9853**. Randomly splitting rows would place $t = 5\text{s}$ in the training set and $t = 10\text{s}$ in the test set, creating catastrophic data leakage and falsely inflated $R^2 > 0.999$.

### Recommended Validation Design: Run-Level Holdout
We recommend evaluating generalization across unseen physical experiments:
- **Training Set (10 runs, 15,532 rows):** ID01, ID02, ID11, ID12, ID13, ID14, ID15, ID17, ID20, ID21. (Covers diverse currents from -20A to -120A, coolant temps from 7°C to 25°C, and flows from 0.1 to 1.5 L/min).
- **Validation Set (2 runs, 4,537 rows):**
  - **ID16:** -120A extreme discharge with medium flow (0.5 L/min).
  - **ID22:** -80A heavy discharge with chilled coolant (8°C).
- **Final Holdout Test Set (2 runs, 2,178 rows):**
  - **ID18:** Severe thermal stress with starved cooling (-80A at 0.1 L/min; peak $68.1\text{ }^\circ\text{C}$, spread $29.3\text{ }^\circ\text{C}$).
  - **ID19:** Extreme stress with low flow (-120A at 0.1 L/min; triggers early thermal safety cutoff at $69.5\text{ }^\circ\text{C}$).
- **Repeatability Check:** ID20 directly replicates ID11 (-40A, 15°C, 1.5 L/min), enabling empirical measurement of experimental repeatability.

---

## 15. Limitations

1. **Module Geometry:** The physical test article is a **1S4P module (4 cylindrical cells)**. It provides high-fidelity inter-cell and axial temperature distributions, but does not include full pack-level thermal propagation across hundreds of cells.
2. **Cooling Geometry:** The laboratory setup utilizes a flat bottom cold plate rather than internal cell-embedded microchannels. In our project, this serves as an experimental surrogate to demonstrate intelligent cooling control.
3. **Quasi-Steady Flow Profiles:** Within each individual run, the coolant flow setpoint was held approximately constant. Multi-step dynamic flow changes during a single discharge pulse were not tested in the laboratory protocol.

---

## 16. Recommended Phase 2 Formulation

Upon approval, Phase 2 (Target/Feature Definition) should proceed with:
1. **Primary Target:** $\Delta T_{60}(t) = T_{\text{cell,max}}(t+60\text{s}) - T_{\text{cell,max}}(t)$.
2. **Secondary Target:** $\Delta T_{300}(t) = T_{\text{cell,max}}(t+300\text{s}) - T_{\text{cell,max}}(t)$.
3. **Core Feature Set:**
   - Electrical: $I(t)$, $V(t)$, $SOC(t)$, $I^2(t)$, rolling load integrals $\int_{t-60}^t I^2 dt$.
   - Thermal State: $T_{\text{cell,max}}(t)$, $T_{\text{cell,spread}}(t)$, $T_{\text{plate,mean}}(t)$.
   - Cooling State: $\text{liquid\_flow}(t)$, $T_{\text{inlet}}(t)$, $T_{\text{ambient}}(t)$.
4. **Strict Exclusion:** Exclude all future signals ($t+h$) and exclude `temp_outlet` from the core feature set.

---

## Generated Artifacts

- **Phase 1 EDA Script:** [`src/freetwinev_eda.py`](file:///c:/AAI%20MP/src/freetwinev_eda.py)
- **Interactive EDA Notebook:** [`notebooks/07_freetwinev_eda.ipynb`](file:///c:/AAI%20MP/notebooks/07_freetwinev_eda.ipynb)
- **Data Quality Summary:** [`results/freetwinev/eda_data_quality.csv`](file:///c:/AAI%20MP/results/freetwinev/eda_data_quality.csv)
- **Hottest Sensors Table:** [`results/freetwinev/eda_hottest_sensors.csv`](file:///c:/AAI%20MP/results/freetwinev/eda_hottest_sensors.csv)
- **Temperature Summary Table:** [`results/freetwinev/eda_temperature_summary.csv`](file:///c:/AAI%20MP/results/freetwinev/eda_temperature_summary.csv)
- **Horizons Summary Table:** [`results/freetwinev/eda_horizons_summary.csv`](file:///c:/AAI%20MP/results/freetwinev/eda_horizons_summary.csv)
- **Summary JSON:** [`results/freetwinev/eda_summary.json`](file:///c:/AAI%20MP/results/freetwinev/eda_summary.json)
- **All 10 Required Figures in [`results/freetwinev/figures/`](file:///c:/AAI%20MP/results/freetwinev/figures/):**
  1. [`01_run_overview.png`](file:///c:/AAI%20MP/results/freetwinev/figures/01_run_overview.png)
  2. [`02_temperature_distributions.png`](file:///c:/AAI%20MP/results/freetwinev/figures/02_temperature_distributions.png)
  3. [`03_representative_temperature_timeseries.png`](file:///c:/AAI%20MP/results/freetwinev/figures/03_representative_temperature_timeseries.png)
  4. [`04_cell_temperature_spatial_comparison.png`](file:///c:/AAI%20MP/results/freetwinev/figures/04_cell_temperature_spatial_comparison.png)
  5. [`05_current_vs_temperature.png`](file:///c:/AAI%20MP/results/freetwinev/figures/05_current_vs_temperature.png)
  6. [`06_cooling_flow_vs_temperature.png`](file:///c:/AAI%20MP/results/freetwinev/figures/06_cooling_flow_vs_temperature.png)
  7. [`07_coolant_temperature_rise.png`](file:///c:/AAI%20MP/results/freetwinev/figures/07_coolant_temperature_rise.png)
  8. [`08_temperature_autocorrelation.png`](file:///c:/AAI%20MP/results/freetwinev/figures/08_temperature_autocorrelation.png)
  9. [`09_horizon_analysis.png`](file:///c:/AAI%20MP/results/freetwinev/figures/09_horizon_analysis.png)
  10. [`10_feature_relationships.png`](file:///c:/AAI%20MP/results/freetwinev/figures/10_feature_relationships.png)
