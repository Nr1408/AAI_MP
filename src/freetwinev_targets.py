"""
Phase 2: FreeTwinEV Target Definition Module
Course: Advanced AI Techniques (AAI)
Project: AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)

This module implements reproducible, leak-free target construction for the FreeTwinEV dataset.
Key responsibilities:
- Conversion of temperature channels from Kelvin to Celsius.
- Computation of physical thermal state metrics across the 12 cell sensors and 10 cold plate sensors.
- Strict run-boundary enforcement when creating future target variables:
  Never shifts targets across the end of one experimental run into another.
- Construction of candidate targets:
  * Absolute future peak temperature: T_cell_max(t+h)
  * Future temperature rise: DeltaT_h(t) = T_cell_max(t+h) - T_cell_max(t)
"""

import pandas as pd
import numpy as np

CELL_SENSORS = [
    'temp_cell-1_TH20', 'temp_cell-1_TH21', 'temp_cell-1_TH22',
    'temp_cell-2_TH17', 'temp_cell-2_TH18', 'temp_cell-2_TH19',
    'temp_cell-3_TH14', 'temp_cell-3_TH15', 'temp_cell-3_TH16',
    'temp_cell-4_TH11', 'temp_cell-4_TH12', 'temp_cell-4_TH13'
]

PLATE_SENSORS = [f'temp_TH{i}' for i in range(1, 11)]

CELL_MAP = {
    'Cell 1': ['temp_cell-1_TH20', 'temp_cell-1_TH21', 'temp_cell-1_TH22'],
    'Cell 2': ['temp_cell-2_TH17', 'temp_cell-2_TH18', 'temp_cell-2_TH19'],
    'Cell 3': ['temp_cell-3_TH14', 'temp_cell-3_TH15', 'temp_cell-3_TH16'],
    'Cell 4': ['temp_cell-4_TH11', 'temp_cell-4_TH12', 'temp_cell-4_TH13']
}

def convert_kelvin_to_celsius(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Celsius columns for all temperature channels in df without modifying raw columns.
    T_C = T_K - 273.15
    """
    df_out = df.copy()
    temp_cols = [c for c in df_out.columns if c.startswith('temp_') and not c.endswith('_C')]
    for col in temp_cols:
        df_out[col + '_C'] = df_out[col] - 273.15
    return df_out

def compute_thermal_envelope(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cell-level and plate-level physical thermal state metrics at time t.
    All outputs are in degrees Celsius.
    """
    df_out = df.copy()
    
    # Ensure Celsius columns exist
    cell_cols_C = [s + '_C' if (s + '_C') in df_out.columns else s for s in CELL_SENSORS]
    plate_cols_C = [s + '_C' if (s + '_C') in df_out.columns else s for s in PLATE_SENSORS]
    
    # Cell module envelope
    df_out['T_cell_max'] = df_out[cell_cols_C].max(axis=1)
    df_out['T_cell_min'] = df_out[cell_cols_C].min(axis=1)
    df_out['T_cell_mean'] = df_out[cell_cols_C].mean(axis=1)
    df_out['T_cell_spread'] = df_out['T_cell_max'] - df_out['T_cell_min']
    
    # Cold plate envelope
    df_out['T_plate_mean'] = df_out[plate_cols_C].mean(axis=1)
    df_out['T_plate_max'] = df_out[plate_cols_C].max(axis=1)
    df_out['T_plate_min'] = df_out[plate_cols_C].min(axis=1)
    df_out['T_plate_spread'] = df_out['T_plate_max'] - df_out['T_plate_min']
    
    # Individual cell maximums
    for cell_name, sensors in CELL_MAP.items():
        cols_c = [s + '_C' if (s + '_C') in df_out.columns else s for s in sensors]
        df_out[f'{cell_name}_max'] = df_out[cols_c].max(axis=1)
        
    return df_out

def add_future_targets(df_run: pd.DataFrame, horizons: dict = None) -> pd.DataFrame:
    """
    Constructs future prediction targets for a SINGLE experimental run.
    
    Parameters:
    -----------
    df_run: DataFrame containing observations of ONE experiment, ordered chronologically.
    horizons: dict mapping horizon name to step count.
              Default: {'60s': 12, '300s': 60} (at 5s per step).
              
    Returns:
    --------
    DataFrame with target columns added:
    - target_T_cell_max_{h}: T_cell_max(t + h)
    - target_DeltaT_{h}: T_cell_max(t + h) - T_cell_max(t)
    
    NOTE ON RUN BOUNDARIES:
    The last h rows of df_run will have NaN for target columns because no future
    measurements exist within this physical experiment.
    These must NEVER be filled from a different experiment.
    """
    if horizons is None:
        horizons = {'60s': 12, '300s': 60}
        
    df_out = df_run.copy()
    
    if 'T_cell_max' not in df_out.columns:
        df_out = compute_thermal_envelope(df_out)
        
    for h_name, steps in horizons.items():
        target_abs_col = f'target_T_cell_max_{h_name}'
        target_delta_col = f'target_DeltaT_{h_name}'
        
        # Future peak cell temperature at t + steps
        df_out[target_abs_col] = df_out['T_cell_max'].shift(-steps)
        
        # Future temperature rise over horizon h
        df_out[target_delta_col] = df_out[target_abs_col] - df_out['T_cell_max']
        
    return df_out

def process_all_runs_targets(runs_dict: dict, horizons: dict = None) -> dict:
    """
    Applies Celsius conversion, thermal envelope calculation, and target construction
    to each run independently, guaranteeing zero cross-run leakage.
    """
    if horizons is None:
        horizons = {'60s': 12, '300s': 60}
        
    processed_runs = {}
    for exp_id, run_info in runs_dict.items():
        df_c = convert_kelvin_to_celsius(run_info['df'])
        df_env = compute_thermal_envelope(df_c)
        df_targets = add_future_targets(df_env, horizons=horizons)
        processed_runs[exp_id] = {
            **run_info,
            'df': df_targets
        }
    return processed_runs
