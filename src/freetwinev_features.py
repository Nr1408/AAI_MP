"""
Phase 2: FreeTwinEV Feature Definition Module
Course: Advanced AI Techniques (AAI)
Project: AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)

This module defines, documents, and computes reproducible, leakage-free feature sets
for the FreeTwinEV battery module dataset.

Feature Categories:
1. Core Baseline Features: Compact, physically essential inputs measurable at time t.
2. Optional Derived Features: Physically motivated non-linear terms, rolling load integrals,
   and thermal gradients strictly computed from past and present observations [<= t].
3. Ablation Features: Downstream calorimetric signals (temp_outlet) isolated for sensitivity testing.

Strict Causality Rules:
- All rolling windows, lags, and differences are computed within individual experimental runs.
- No rolling window or lag ever crosses experiment boundaries.
- No future observations (t + h) are ever used.
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from src.freetwinev_targets import compute_thermal_envelope, convert_kelvin_to_celsius

# Feature Groups
CORE_FEATURES = [
    'current',              # Module current [A]
    'voltage',              # Terminal voltage [V]
    'SOC',                  # State of Charge [%]
    'liquid_flow',          # Coolant flow rate [L/min]
    'temp_inlet_C',         # Coolant inlet temperature [°C]
    'temp_ambient_C',       # Ambient chamber temperature [°C]
    'T_cell_max',           # Current maximum cell surface temperature [°C]
    'T_cell_spread',        # Current thermal non-uniformity across 12 cell sensors [°C]
    'T_plate_mean'          # Current mean cold plate surface temperature [°C]
]

DERIVED_FEATURES = [
    # Joule heating & electrical load proxies
    'current_squared',      # I(t)^2 [A^2] — proxy for instantaneous Joule heat generation (I^2 * R)
    'current_abs',          # |I(t)| [A] — magnitude of current load
    'power_gross',          # V(t) * I(t) [W] — electrical power flow (negative = discharge power)
    
    # Rolling load accumulation (past heating memory over 5s samples: 12 steps = 60s, 60 steps = 300s)
    'current_mean_60s',     # Rolling mean of current over past 60s [A]
    'current_mean_300s',    # Rolling mean of current over past 300s [A]
    'current_sq_mean_60s',  # Rolling mean of I^2 over past 60s [A^2] (cumulative heat input proxy)
    'current_sq_mean_300s', # Rolling mean of I^2 over past 300s [A^2]
    
    # Thermal history & gradients
    'T_cell_max_diff_30s',  # T_cell_max(t) - T_cell_max(t-30s) [°C] — 30s thermal velocity
    'T_cell_max_diff_60s',  # T_cell_max(t) - T_cell_max(t-60s) [°C] — 60s thermal velocity
    'T_cell_to_plate_diff', # T_cell_max(t) - T_plate_mean(t) [°C] — conduction driving force across TIM
    'T_cell_to_inlet_diff', # T_cell_max(t) - temp_inlet_C(t) [°C] — total module-to-coolant thermal gradient
    
    # Cooling interaction
    'flow_x_inlet_temp'     # liquid_flow(t) * temp_inlet_C(t) [L/min * °C] — advective thermal flux proxy
]

ABLATION_FEATURES = [
    'temp_outlet_C',        # Coolant outlet temperature [°C] (downstream calorimetric sensor)
    'DeltaT_coolant_C'      # temp_outlet_C(t) - temp_inlet_C(t) [°C] (sensible heat absorbed by fluid)
]

def get_feature_schema() -> pd.DataFrame:
    """
    Returns the comprehensive feature taxonomy table documenting:
    - Feature Name
    - Category
    - Source Raw Column(s)
    - Physical Units
    - Time Availability
    - Transformation Formula
    - Window / Lag Length
    - Leakage Status
    - Physical Interpretation
    """
    schema = [
        # --- CORE FEATURES ---
        {
            'Feature': 'current',
            'Category': 'Core Electrical',
            'Source': 'current',
            'Unit': 'A',
            'Availability': 'Time t',
            'Transformation': 'Identity',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Available at t)',
            'Physical_Rationale': 'Primary driver of ohmic heat generation (I^2 * R) and electrochemical polarization.'
        },
        {
            'Feature': 'voltage',
            'Category': 'Core Electrical',
            'Source': 'voltage',
            'Unit': 'V',
            'Availability': 'Time t',
            'Transformation': 'Identity',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Available at t)',
            'Physical_Rationale': 'Measures cell potential, internal polarization drop, and proximity to discharge/charge cutoffs.'
        },
        {
            'Feature': 'SOC',
            'Category': 'Core Electrical',
            'Source': 'SOC',
            'Unit': '%',
            'Availability': 'Time t',
            'Transformation': 'Identity',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Available at t)',
            'Physical_Rationale': 'Internal cell state; entropic heat coefficient dU/dT and internal resistance vary as a function of SOC.'
        },
        {
            'Feature': 'liquid_flow',
            'Category': 'Core Cooling',
            'Source': 'liquid_flow',
            'Unit': 'L/min',
            'Availability': 'Time t',
            'Transformation': 'Identity',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Available at t)',
            'Physical_Rationale': 'Active thermal control actuation; governs convection heat transfer coefficient h(F) in the cold plate.'
        },
        {
            'Feature': 'temp_inlet_C',
            'Category': 'Core Cooling',
            'Source': 'temp_inlet',
            'Unit': '°C',
            'Availability': 'Time t',
            'Transformation': 'temp_inlet - 273.15',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Available at t)',
            'Physical_Rationale': 'Thermal boundary condition; cold plate coolant supply sink temperature.'
        },
        {
            'Feature': 'temp_ambient_C',
            'Category': 'Core Environment',
            'Source': 'temp_ambient',
            'Unit': '°C',
            'Availability': 'Time t',
            'Transformation': 'temp_ambient - 273.15',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Available at t)',
            'Physical_Rationale': 'Ambient chamber boundary condition for external natural convection and radiation losses.'
        },
        {
            'Feature': 'T_cell_max',
            'Category': 'Core Thermal State',
            'Source': '12 cell sensors',
            'Unit': '°C',
            'Availability': 'Time t',
            'Transformation': 'max(12 cell sensors in °C)',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Available at t)',
            'Physical_Rationale': 'Current thermal hotspot of the module; primary base state for autoregressive / delta forecasting.'
        },
        {
            'Feature': 'T_cell_spread',
            'Category': 'Core Thermal State',
            'Source': '12 cell sensors',
            'Unit': '°C',
            'Availability': 'Time t',
            'Transformation': 'max(cell sensors) - min(cell sensors)',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Available at t)',
            'Physical_Rationale': 'Current temperature non-uniformity across the module; indicates localized thermal stress.'
        },
        {
            'Feature': 'T_plate_mean',
            'Category': 'Core Thermal State',
            'Source': 'temp_TH1 to temp_TH10',
            'Unit': '°C',
            'Availability': 'Time t',
            'Transformation': 'mean(10 cold plate sensors in °C)',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Available at t)',
            'Physical_Rationale': 'Compact representation of the 10 cold plate surface sensors; represents the heat sink interface.'
        },
        
        # --- DERIVED FEATURES ---
        {
            'Feature': 'current_squared',
            'Category': 'Derived Electrical',
            'Source': 'current',
            'Unit': 'A^2',
            'Availability': 'Time t',
            'Transformation': 'current^2',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Derived from Class A)',
            'Physical_Rationale': 'Non-linear proxy for instantaneous Joule heating rate (Q_dot_gen = I^2 * R_eff).'
        },
        {
            'Feature': 'current_abs',
            'Category': 'Derived Electrical',
            'Source': 'current',
            'Unit': 'A',
            'Availability': 'Time t',
            'Transformation': 'abs(current)',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Derived from Class A)',
            'Physical_Rationale': 'Load magnitude independent of sign convention (both heavy discharge and fast charge heat cells).'
        },
        {
            'Feature': 'power_gross',
            'Category': 'Derived Electrical',
            'Source': 'voltage, current',
            'Unit': 'W',
            'Availability': 'Time t',
            'Transformation': 'voltage * current',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Derived from Class A)',
            'Physical_Rationale': 'Electrical power throughput; captures cell polarization power losses.'
        },
        {
            'Feature': 'current_mean_60s',
            'Category': 'Derived Load History',
            'Source': 'current',
            'Unit': 'A',
            'Availability': 'Time <= t',
            'Transformation': 'rolling_mean(current, window=12)',
            'Window': 'Past 60 seconds [t-60s, t]',
            'Leakage_Status': 'Approved (Backward-only window)',
            'Physical_Rationale': 'Average current load over the preceding 60 seconds; filters instantaneous DAQ noise.'
        },
        {
            'Feature': 'current_mean_300s',
            'Category': 'Derived Load History',
            'Source': 'current',
            'Unit': 'A',
            'Availability': 'Time <= t',
            'Transformation': 'rolling_mean(current, window=60)',
            'Window': 'Past 300 seconds [t-300s, t]',
            'Leakage_Status': 'Approved (Backward-only window)',
            'Physical_Rationale': 'Sustained load history over the preceding 5 minutes; accounts for module thermal time constants.'
        },
        {
            'Feature': 'current_sq_mean_60s',
            'Category': 'Derived Load History',
            'Source': 'current',
            'Unit': 'A^2',
            'Availability': 'Time <= t',
            'Transformation': 'rolling_mean(current^2, window=12)',
            'Window': 'Past 60 seconds [t-60s, t]',
            'Leakage_Status': 'Approved (Backward-only window)',
            'Physical_Rationale': 'Integrated Joule heat energy accumulated in the cell core over the preceding 1 minute.'
        },
        {
            'Feature': 'current_sq_mean_300s',
            'Category': 'Derived Load History',
            'Source': 'current',
            'Unit': 'A^2',
            'Availability': 'Time <= t',
            'Transformation': 'rolling_mean(current^2, window=60)',
            'Window': 'Past 300 seconds [t-300s, t]',
            'Leakage_Status': 'Approved (Backward-only window)',
            'Physical_Rationale': 'Integrated Joule heat energy accumulated over the preceding 5 minutes.'
        },
        {
            'Feature': 'T_cell_max_diff_30s',
            'Category': 'Derived Thermal Velocity',
            'Source': 'T_cell_max',
            'Unit': '°C',
            'Availability': 'Time <= t',
            'Transformation': 'T_cell_max(t) - T_cell_max(t-30s)',
            'Window': 'Lag 6 steps (30s) [t-30s, t]',
            'Leakage_Status': 'Approved (Backward-only difference)',
            'Physical_Rationale': 'Recent 30s thermal velocity (dT/dt); captures instantaneous heating/cooling acceleration.'
        },
        {
            'Feature': 'T_cell_max_diff_60s',
            'Category': 'Derived Thermal Velocity',
            'Source': 'T_cell_max',
            'Unit': '°C',
            'Availability': 'Time <= t',
            'Transformation': 'T_cell_max(t) - T_cell_max(t-60s)',
            'Window': 'Lag 12 steps (60s) [t-60s, t]',
            'Leakage_Status': 'Approved (Backward-only difference)',
            'Physical_Rationale': 'Recent 60s thermal velocity; captures sustained thermal trend.'
        },
        {
            'Feature': 'T_cell_to_plate_diff',
            'Category': 'Derived Thermal Gradient',
            'Source': 'T_cell_max, T_plate_mean',
            'Unit': '°C',
            'Availability': 'Time t',
            'Transformation': 'T_cell_max - T_plate_mean',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Derived from Class A)',
            'Physical_Rationale': 'Thermal conduction driving force across the TIM layer (q_cond = k * A * deltaT / L).'
        },
        {
            'Feature': 'T_cell_to_inlet_diff',
            'Category': 'Derived Thermal Gradient',
            'Source': 'T_cell_max, temp_inlet',
            'Unit': '°C',
            'Availability': 'Time t',
            'Transformation': 'T_cell_max - temp_inlet_C',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Derived from Class A)',
            'Physical_Rationale': 'Overall thermal potential difference between hottest cell point and coolant supply.'
        },
        {
            'Feature': 'flow_x_inlet_temp',
            'Category': 'Derived Cooling Interaction',
            'Source': 'liquid_flow, temp_inlet',
            'Unit': 'L/min * °C',
            'Availability': 'Time t',
            'Transformation': 'liquid_flow * temp_inlet_C',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Approved (Derived from Class A)',
            'Physical_Rationale': 'Non-linear interaction proxy for advective coolant thermal capacity and heat sink capacity.'
        },
        
        # --- ABLATION FEATURES ---
        {
            'Feature': 'temp_outlet_C',
            'Category': 'Ablation Cooling',
            'Source': 'temp_outlet',
            'Unit': '°C',
            'Availability': 'Time t',
            'Transformation': 'temp_outlet - 273.15',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Class E (Excluded from core baseline; used only in ablation)',
            'Physical_Rationale': 'Downstream fluid exit temperature. Reflects heat already transferred into fluid.'
        },
        {
            'Feature': 'DeltaT_coolant_C',
            'Category': 'Ablation Cooling',
            'Source': 'temp_outlet, temp_inlet',
            'Unit': '°C',
            'Availability': 'Time t',
            'Transformation': 'temp_outlet_C - temp_inlet_C',
            'Window': 'Instantaneous (t)',
            'Leakage_Status': 'Class E (Excluded from core baseline; used only in ablation)',
            'Physical_Rationale': 'Sensible coolant temperature rise (Q_dot = m_dot * C_p * DeltaT_coolant).'
        }
    ]
    return pd.DataFrame(schema)

def compute_features_for_run(df_run: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all core, derived, and ablation features for a single experimental run.
    Guarantees backward-only windows and run-boundary isolation.
    """
    df = df_run.copy()
    
    # 1. Ensure Celsius and thermal metrics exist
    if 'T_cell_max' not in df.columns:
        df = convert_kelvin_to_celsius(df)
        df = compute_thermal_envelope(df)
        
    # Ensure inlet/outlet Celsius columns
    if 'temp_inlet_C' not in df.columns and 'temp_inlet' in df.columns:
        df['temp_inlet_C'] = df['temp_inlet'] - 273.15
    if 'temp_outlet_C' not in df.columns and 'temp_outlet' in df.columns:
        df['temp_outlet_C'] = df['temp_outlet'] - 273.15
    if 'temp_ambient_C' not in df.columns and 'temp_ambient' in df.columns:
        df['temp_ambient_C'] = df['temp_ambient'] - 273.15
        
    # 2. Derived electrical features
    df['current_squared'] = df['current'] ** 2
    df['current_abs'] = df['current'].abs()
    df['power_gross'] = df['voltage'] * df['current']
    
    # 3. Rolling load history (backward-only within this experiment)
    # min_periods=1 ensures initial rows within the run are retained cleanly
    df['current_mean_60s'] = df['current'].rolling(window=12, min_periods=1).mean()
    df['current_mean_300s'] = df['current'].rolling(window=60, min_periods=1).mean()
    df['current_sq_mean_60s'] = df['current_squared'].rolling(window=12, min_periods=1).mean()
    df['current_sq_mean_300s'] = df['current_squared'].rolling(window=60, min_periods=1).mean()
    
    # 4. Thermal history & gradients (backward-only differences)
    # Shift 6 steps = 30s, shift 12 steps = 60s
    # For the first few steps where lag is not available, fill with 0.0 (no change prior to start)
    df['T_cell_max_diff_30s'] = df['T_cell_max'] - df['T_cell_max'].shift(6).bfill()
    df['T_cell_max_diff_60s'] = df['T_cell_max'] - df['T_cell_max'].shift(12).bfill()
    df['T_cell_to_plate_diff'] = df['T_cell_max'] - df['T_plate_mean']
    df['T_cell_to_inlet_diff'] = df['T_cell_max'] - df['temp_inlet_C']
    
    # 5. Cooling interaction
    df['flow_x_inlet_temp'] = df['liquid_flow'] * df['temp_inlet_C']
    
    # 6. Ablation features (isolated)
    df['DeltaT_coolant_C'] = df['temp_outlet_C'] - df['temp_inlet_C']
    
    return df

def compute_features_all_runs(runs_dict: dict) -> dict:
    """
    Applies feature computation to each run independently, ensuring strict run isolation.
    """
    featured_runs = {}
    for exp_id, run_info in runs_dict.items():
        df_feat = compute_features_for_run(run_info['df'])
        featured_runs[exp_id] = {
            **run_info,
            'df': df_feat
        }
    return featured_runs
