"""
Phase 1: FreeTwinEV Exploratory Data Analysis (EDA)
Course: Advanced AI Techniques (AAI)
Use Case: AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)

This script performs rigorous exploratory data analysis of the FreeTwinEV dataset across all 14 runs.
It calculates:
- Data quality metrics per run (confirming Phase 0 findings, checking NaNs, constants, monotonicity, intervals)
- Temperature distributions, axial/spatial gradients, hottest sensor frequency analysis
- Electrical dynamics (current sign convention, voltage, SOC)
- Cooling dynamics (flow rate impact, inlet/outlet behavior, coolant temperature rise)
- Temporal dynamics across candidate horizons (30s, 60s, 120s, 300s)
- Feature relationships, correlation structures, and redundancy
- 10 publication-quality diagnostic figures saved to results/freetwinev/figures/
"""

import os
import sys
import glob
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Constants & Channel Layout
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

AXIAL_POSITION = {
    'temp_cell-1_TH20': 'Top', 'temp_cell-1_TH21': 'Middle', 'temp_cell-1_TH22': 'Bottom (near plate)',
    'temp_cell-2_TH17': 'Top', 'temp_cell-2_TH18': 'Middle', 'temp_cell-2_TH19': 'Bottom (near plate)',
    'temp_cell-3_TH14': 'Top', 'temp_cell-3_TH15': 'Middle', 'temp_cell-3_TH16': 'Bottom (near plate)',
    'temp_cell-4_TH11': 'Top', 'temp_cell-4_TH12': 'Middle', 'temp_cell-4_TH13': 'Bottom (near plate)'
}

def load_run(csv_path):
    """Load a single run CSV, convert to numeric, and convert temperature to Celsius."""
    df_raw = pd.read_csv(csv_path)
    # Row 0 contains unit strings ('s', 'A', 'V', '%', 'L/min', 'K')
    units = df_raw.iloc[0].to_dict()
    df = df_raw.iloc[1:].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Standardize time column name
    time_col = 'timesamp' if 'timesamp' in df.columns else df.columns[0]
    df['time_s'] = df[time_col]
    df['time_min'] = df['time_s'] / 60.0
    
    # Convert temperatures from Kelvin to Celsius
    temp_cols = [c for c in df.columns if c.startswith('temp_')]
    for c in temp_cols:
        df[c + '_C'] = df[c] - 273.15
        
    # Derived thermal metrics (in-memory analysis only)
    cell_cols_C = [c + '_C' for c in CELL_SENSORS]
    df['T_cell_max_C'] = df[cell_cols_C].max(axis=1)
    df['T_cell_min_C'] = df[cell_cols_C].min(axis=1)
    df['T_cell_mean_C'] = df[cell_cols_C].mean(axis=1)
    df['T_cell_spread_C'] = df['T_cell_max_C'] - df['T_cell_min_C']
    
    # Cold plate mean
    plate_cols_C = [c + '_C' for c in PLATE_SENSORS]
    df['T_plate_mean_C'] = df[plate_cols_C].mean(axis=1)
    
    # Coolant temperature rise
    df['DeltaT_coolant_C'] = df['temp_outlet_C'] - df['temp_inlet_C']
    
    # Per-cell maximums
    for cell_name, sensors in CELL_MAP.items():
        cols_c = [s + '_C' for s in sensors]
        df[f'{cell_name}_max_C'] = df[cols_c].max(axis=1)
        df[f'{cell_name}_mean_C'] = df[cols_c].mean(axis=1)
        
    return df, units

def load_all_runs(all_data_dir):
    """Load all 14 runs into a dictionary keyed by experiment ID."""
    run_dirs = sorted(glob.glob(os.path.join(all_data_dir, "ID*")))
    runs = {}
    for rdir in run_dirs:
        dir_name = os.path.basename(rdir)
        exp_id = dir_name.split('_')[0]
        csv_path = glob.glob(os.path.join(rdir, "*.csv"))[0]
        df, units = load_run(csv_path)
        runs[exp_id] = {
            'exp_id': exp_id,
            'dir_name': dir_name,
            'csv_path': csv_path,
            'df': df,
            'units': units
        }
    return runs

def compute_data_quality(runs):
    """Inspect every run for data quality metrics."""
    quality_records = []
    total_data_rows = 0
    
    for exp_id, run_info in runs.items():
        df = run_info['df']
        total_data_rows += len(df)
        t = df['time_s'].values
        dt = np.diff(t)
        
        # Check missing & constant columns
        raw_cols = [c for c in df.columns if not c.endswith('_C') and c not in ['time_s', 'time_min']]
        null_counts = df[raw_cols].isnull().sum().sum()
        constant_cols = [c for c in raw_cols if df[c].nunique() <= 1]
        
        # Check monotonicity
        is_monotonic = bool(np.all(dt > 0))
        dup_timestamps = int(np.sum(dt == 0))
        neg_steps = int(np.sum(dt < 0))
        
        # Sampling interval stats
        dt_min = float(np.min(dt)) if len(dt) > 0 else 0
        dt_max = float(np.max(dt)) if len(dt) > 0 else 0
        dt_med = float(np.median(dt)) if len(dt) > 0 else 0
        dt_mean = float(np.mean(dt)) if len(dt) > 0 else 0
        gaps_gt_7_5s = int(np.sum(dt > 7.5))
        
        # Physical plausibility
        min_v = df['voltage'].min()
        max_v = df['voltage'].max()
        min_i = df['current'].min()
        max_i = df['current'].max()
        min_temp_c = df['T_cell_min_C'].min()
        max_temp_c = df['T_cell_max_C'].max()
        
        quality_records.append({
            'Exp_ID': exp_id,
            'Directory': run_info['dir_name'],
            'Data_Rows': len(df),
            'Duration_min': round(df['time_min'].iloc[-1], 2),
            'Null_Count': int(null_counts),
            'Constant_Cols_Count': len(constant_cols),
            'Is_Monotonic': is_monotonic,
            'Duplicate_Timestamps': dup_timestamps,
            'Negative_Steps': neg_steps,
            'dt_median_s': round(dt_med, 2),
            'dt_min_s': round(dt_min, 2),
            'dt_max_s': round(dt_max, 2),
            'Gaps_gt_7.5s': gaps_gt_7_5s,
            'Voltage_Range_V': f"[{min_v:.2f}, {max_v:.2f}]",
            'Current_Range_A': f"[{min_i:.1f}, {max_i:.1f}]",
            'Cell_Temp_Range_C': f"[{min_temp_c:.1f}, {max_temp_c:.1f}]"
        })
        
    df_quality = pd.DataFrame(quality_records)
    return df_quality, total_data_rows

def compute_temperature_analysis(runs):
    """Analyze temperature distributions, spatial gradients, and hottest sensor frequency."""
    hottest_counts = {s: 0 for s in CELL_SENSORS}
    total_samples = 0
    sensor_stats = {s: {'min': [], 'max': [], 'mean': []} for s in CELL_SENSORS}
    
    run_temp_summaries = []
    
    for exp_id, run_info in runs.items():
        df = run_info['df']
        total_samples += len(df)
        cell_vals = df[[s + '_C' for s in CELL_SENSORS]].values
        
        # Hottest sensor at each timestamp
        hottest_idx = np.argmax(cell_vals, axis=1)
        for idx in hottest_idx:
            hottest_counts[CELL_SENSORS[idx]] += 1
            
        for s in CELL_SENSORS:
            sensor_stats[s]['min'].append(df[s + '_C'].min())
            sensor_stats[s]['max'].append(df[s + '_C'].max())
            sensor_stats[s]['mean'].append(df[s + '_C'].mean())
            
        run_temp_summaries.append({
            'Exp_ID': exp_id,
            'T_cell_max_peak_C': round(df['T_cell_max_C'].max(), 2),
            'T_cell_max_mean_C': round(df['T_cell_max_C'].mean(), 2),
            'T_cell_min_lowest_C': round(df['T_cell_min_C'].min(), 2),
            'T_cell_spread_max_C': round(df['T_cell_spread_C'].max(), 2),
            'T_cell_spread_mean_C': round(df['T_cell_spread_C'].mean(), 2),
            'T_plate_mean_C': round(df['T_plate_mean_C'].mean(), 2),
            'T_inlet_mean_C': round(df['temp_inlet_C'].mean(), 2),
            'T_outlet_mean_C': round(df['temp_outlet_C'].mean(), 2),
            'T_ambient_mean_C': round(df['temp_ambient_C'].mean(), 2),
            'Cell1_peak_C': round(df['Cell 1_max_C'].max(), 2),
            'Cell2_peak_C': round(df['Cell 2_max_C'].max(), 2),
            'Cell3_peak_C': round(df['Cell 3_max_C'].max(), 2),
            'Cell4_peak_C': round(df['Cell 4_max_C'].max(), 2)
        })
        
    df_hottest = pd.DataFrame([
        {
            'Sensor': s,
            'Cell': s.split('_')[1].split('-')[1],
            'Channel_TH': s.split('_')[-1],
            'Axial_Position': AXIAL_POSITION[s],
            'Hottest_Count': hottest_counts[s],
            'Hottest_Percent': round(hottest_counts[s] / total_samples * 100, 2),
            'Overall_Mean_C': round(np.mean(sensor_stats[s]['mean']), 2),
            'Overall_Max_C': round(np.max(sensor_stats[s]['max']), 2)
        }
        for s in CELL_SENSORS
    ]).sort_values(by='Hottest_Percent', ascending=False)
    
    df_temp_summary = pd.DataFrame(run_temp_summaries)
    return df_hottest, df_temp_summary

def compute_temporal_dynamics(runs):
    """Analyze autocorrelations and delta-T distributions across candidate horizons."""
    horizons = {'30s': 6, '60s': 12, '120s': 24, '300s': 60}
    horizon_results = {h: {'deltas': [], 'autocorrs': []} for h in horizons}
    
    for exp_id, run_info in runs.items():
        df = run_info['df']
        t_max = df['T_cell_max_C']
        
        for h_name, steps in horizons.items():
            if len(df) > steps + 10:
                ac = t_max.autocorr(steps)
                horizon_results[h_name]['autocorrs'].append(ac)
                delta_h = t_max.iloc[steps:].values - t_max.iloc[:-steps].values
                horizon_results[h_name]['deltas'].extend(delta_h.tolist())
                
    summary_horizon = []
    for h_name, steps in horizons.items():
        deltas = np.array(horizon_results[h_name]['deltas'])
        abs_deltas = np.abs(deltas)
        acs = horizon_results[h_name]['autocorrs']
        summary_horizon.append({
            'Horizon': h_name,
            'Steps': steps,
            'Mean_Autocorr': round(float(np.mean(acs)), 4),
            'Min_Autocorr': round(float(np.min(acs)), 4),
            'Max_Autocorr': round(float(np.max(acs)), 4),
            'Mean_Delta_C': round(float(np.mean(deltas)), 4),
            'Mean_Abs_Delta_C': round(float(np.mean(abs_deltas)), 4),
            'Max_Abs_Delta_C': round(float(np.max(abs_deltas)), 4),
            'P95_Abs_Delta_C': round(float(np.percentile(abs_deltas, 95)), 4),
            'P99_Abs_Delta_C': round(float(np.percentile(abs_deltas, 99)), 4)
        })
        
    df_horizons = pd.DataFrame(summary_horizon)
    return df_horizons, horizon_results

def generate_all_10_figures(runs, df_horizons, out_dir='results/freetwinev/figures'):
    """Generate all 10 required presentation-quality figures."""
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.0)
    
    # -------------------------------------------------------------
    # Figure 1: 01_run_overview.png
    # -------------------------------------------------------------
    exp_ids = list(runs.keys())
    durations = [runs[e]['df']['time_min'].iloc[-1] for e in exp_ids]
    i_mins = [runs[e]['df']['current'].min() for e in exp_ids]
    flow_means = [runs[e]['df']['liquid_flow'].mean() for e in exp_ids]
    tinlet_means = [runs[e]['df']['temp_inlet_C'].mean() for e in exp_ids]
    peak_temps = [runs[e]['df']['T_cell_max_C'].max() for e in exp_ids]
    peak_spreads = [runs[e]['df']['T_cell_spread_C'].max() for e in exp_ids]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Figure 1: FreeTwinEV Experimental Runs Overview (14 Runs, 22,247 Observations)', fontsize=14, fontweight='bold')
    
    axes[0, 0].bar(exp_ids, durations, color='royalblue')
    axes[0, 0].set_title('Duration per Experiment [min]', fontweight='bold')
    axes[0, 0].set_ylabel('Minutes')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(exp_ids, i_mins, color='crimson')
    axes[0, 1].set_title('Minimum Current (Discharge Level) [A]', fontweight='bold')
    axes[0, 1].set_ylabel('Current [A]')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    axes[0, 2].bar(exp_ids, flow_means, color='darkorange')
    axes[0, 2].set_title('Mean Coolant Flow Rate [L/min]', fontweight='bold')
    axes[0, 2].set_ylabel('L/min')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    axes[1, 0].bar(exp_ids, tinlet_means, color='teal')
    axes[1, 0].set_title('Mean Coolant Inlet Temperature [°C]', fontweight='bold')
    axes[1, 0].set_ylabel('Temperature [°C]')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].bar(exp_ids, peak_temps, color='purple')
    axes[1, 1].axhline(70, color='red', linestyle='--', label='70°C Safety Cutoff')
    axes[1, 1].set_title('Peak Cell Temperature Reached [°C]', fontweight='bold')
    axes[1, 1].set_ylabel('Peak T_cell_max [°C]')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    axes[1, 2].bar(exp_ids, peak_spreads, color='firebrick')
    axes[1, 2].set_title('Maximum Temperature Spread Across Cells [°C]', fontweight='bold')
    axes[1, 2].set_ylabel('Max (T_max - T_min) [°C]')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '01_run_overview.png'), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 2: 02_temperature_distributions.png
    # -------------------------------------------------------------
    all_dfs = [runs[e]['df'] for e in exp_ids]
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Figure 2: Temperature Distributions across System Channels', fontsize=14, fontweight='bold')
    
    box_data = {
        'Ambient': df_all['temp_ambient_C'],
        'Coolant In': df_all['temp_inlet_C'],
        'Coolant Out': df_all['temp_outlet_C'],
        'Cold Plate': df_all['T_plate_mean_C'],
        'Cell Mean': df_all['T_cell_mean_C'],
        'Cell Min': df_all['T_cell_min_C'],
        'Cell Max': df_all['T_cell_max_C'],
    }
    df_box = pd.DataFrame(box_data)
    sns.boxplot(data=df_box, ax=axes[0], palette='Set2')
    axes[0].set_title('Overall Temperature Distributions (All 22,247 Samples)', fontweight='bold')
    axes[0].set_ylabel('Temperature [°C]')
    axes[0].axhline(70, color='red', linestyle=':', label='70°C Cutoff')
    axes[0].legend()
    
    # Histogram of T_cell_max and T_cell_spread
    sns.histplot(df_all['T_cell_max_C'], ax=axes[1], color='purple', kde=True, label='T_cell_max [°C]', bins=40)
    sns.histplot(df_all['T_cell_spread_C'], ax=axes[1], color='crimson', kde=True, label='T_cell_spread [°C]', bins=40)
    axes[1].set_title('Density of Peak Cell Temperature & Inter-Cell Spread', fontweight='bold')
    axes[1].set_xlabel('Temperature [°C]')
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '02_temperature_distributions.png'), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 3: 03_representative_temperature_timeseries.png
    # -------------------------------------------------------------
    # Compare ID13 (extreme -120A discharge) vs ID18 (starved cooling -80A at 0.1 L/min)
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col')
    fig.suptitle('Figure 3: Representative Time Series: High-Stress ID13 (Left) vs Starved Cooling ID18 (Right)', fontsize=14, fontweight='bold')
    
    for col_idx, (eid, title) in enumerate([('ID13', 'ID13 (-120A, 1.5 L/min, 15°C)'), ('ID18', 'ID18 (-80A, 0.1 L/min, 15°C)')]):
        df_e = runs[eid]['df']
        t_m = df_e['time_min']
        
        # Panel 1: Current & Voltage
        axes[0, col_idx].plot(t_m, df_e['current'], color='crimson', label='Current [A]')
        axes[0, col_idx].set_ylabel('Current [A]', color='crimson')
        axes[0, col_idx].set_title(f'{title} — Electrical Profile', fontweight='bold')
        axes[0, col_idx].grid(True, linestyle='--', alpha=0.5)
        ax0_r = axes[0, col_idx].twinx()
        ax0_r.plot(t_m, df_e['voltage'], color='navy', linestyle='--', label='Voltage [V]')
        ax0_r.set_ylabel('Voltage [V]', color='navy')
        
        # Panel 2: Cooling Conditions
        axes[1, col_idx].plot(t_m, df_e['temp_inlet_C'], color='blue', label='Coolant In')
        axes[1, col_idx].plot(t_m, df_e['temp_outlet_C'], color='cyan', label='Coolant Out')
        axes[1, col_idx].plot(t_m, df_e['T_plate_mean_C'], color='darkgreen', linestyle=':', label='Plate Mean')
        axes[1, col_idx].set_ylabel('Temp [°C]')
        axes[1, col_idx].set_title('Cooling Temperatures & Flow', fontweight='bold')
        axes[1, col_idx].legend(loc='upper right', fontsize=9)
        axes[1, col_idx].grid(True, linestyle='--', alpha=0.5)
        ax1_r = axes[1, col_idx].twinx()
        ax1_r.plot(t_m, df_e['liquid_flow'], color='darkorange', label='Flow')
        ax1_r.set_ylabel('Flow [L/min]', color='darkorange')
        
        # Panel 3: Cell Temperatures
        for cs in ['Cell 1_max_C', 'Cell 2_max_C', 'Cell 3_max_C', 'Cell 4_max_C']:
            axes[2, col_idx].plot(t_m, df_e[cs], alpha=0.5, label=cs.replace('_max_C', ''))
        axes[2, col_idx].plot(t_m, df_e['T_cell_max_C'], color='black', linewidth=1.5, linestyle='--', label='T_cell_max')
        axes[2, col_idx].axhline(70, color='red', linestyle=':', label='70°C Cutoff')
        axes[2, col_idx].set_ylabel('Cell Temp [°C]')
        axes[2, col_idx].set_xlabel('Elapsed Time [min]')
        axes[2, col_idx].set_title('Cell Temperatures & Thermal Envelope', fontweight='bold')
        axes[2, col_idx].legend(loc='upper right', fontsize=9)
        axes[2, col_idx].grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '03_representative_temperature_timeseries.png'), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 4: 04_cell_temperature_spatial_comparison.png
    # -------------------------------------------------------------
    # Axial & Inter-cell spatial gradients
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Figure 4: Spatial Thermal Gradients: Across Cells (Left) and Axial Height (Right)', fontsize=14, fontweight='bold')
    
    cell_box = {
        'Cell 1': df_all['Cell 1_max_C'],
        'Cell 2': df_all['Cell 2_max_C'],
        'Cell 3': df_all['Cell 3_max_C'],
        'Cell 4': df_all['Cell 4_max_C'],
    }
    sns.boxplot(data=pd.DataFrame(cell_box), ax=axes[0], palette='Blues_r')
    axes[0].set_title('Cell-to-Cell Peak Temperature Distribution', fontweight='bold')
    axes[0].set_ylabel('Peak Temperature [°C]')
    
    # Axial height: Top, Middle, Bottom sensors across all cells
    top_cols = ['temp_cell-1_TH20_C', 'temp_cell-2_TH17_C', 'temp_cell-3_TH14_C', 'temp_cell-4_TH11_C']
    mid_cols = ['temp_cell-1_TH21_C', 'temp_cell-2_TH18_C', 'temp_cell-3_TH15_C', 'temp_cell-4_TH12_C']
    bot_cols = ['temp_cell-1_TH22_C', 'temp_cell-2_TH19_C', 'temp_cell-3_TH16_C', 'temp_cell-4_TH13_C']
    
    axial_data = {
        'Top (Furthest from plate)': df_all[top_cols].mean(axis=1),
        'Middle': df_all[mid_cols].mean(axis=1),
        'Bottom (Closest to cold plate)': df_all[bot_cols].mean(axis=1)
    }
    sns.boxplot(data=pd.DataFrame(axial_data), ax=axes[1], palette='Oranges_r')
    axes[1].set_title('Axial Height Temperature Distribution (Cold Plate Contact Effect)', fontweight='bold')
    axes[1].set_ylabel('Temperature [°C]')
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '04_cell_temperature_spatial_comparison.png'), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 5: 05_current_vs_temperature.png
    # -------------------------------------------------------------
    # Joule heating relationship: Current level vs rate of temperature rise
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Figure 5: Electrical Load & Joule Heating Dynamics', fontsize=14, fontweight='bold')
    
    # Current vs Peak Cell Temp by Run
    run_i_peak = [(runs[e]['df']['current'].min(), runs[e]['df']['T_cell_max_C'].max(), e) for e in exp_ids]
    i_vals, peak_vals, labels = zip(*run_i_peak)
    axes[0].scatter(np.abs(i_vals), peak_vals, color='crimson', s=100, edgecolor='black', zorder=4)
    for i, p, lbl in zip(i_vals, peak_vals, labels):
        axes[0].annotate(f"{lbl} ({abs(i):.0f}A)", (abs(i)+1.5, p-0.5), fontsize=9)
    axes[0].axhline(70, color='red', linestyle='--', label='70°C Safety Threshold')
    axes[0].set_title('Discharge Current Magnitude vs Peak Cell Temperature Reached', fontweight='bold')
    axes[0].set_xlabel('Discharge Current |I| [A]')
    axes[0].set_ylabel('Peak Cell Temperature [°C]')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Instantaneous Current vs Voltage scatter colored by T_cell_max
    sample_sub = df_all.sample(n=3000, random_state=42)
    sc = axes[1].scatter(sample_sub['current'], sample_sub['voltage'], c=sample_sub['T_cell_max_C'], cmap='viridis', alpha=0.6, s=15)
    cbar = plt.colorbar(sc, ax=axes[1])
    cbar.set_label('T_cell_max [°C]')
    axes[1].set_title('Current vs Voltage (Colored by Cell Temperature)', fontweight='bold')
    axes[1].set_xlabel('Current [A] (Negative = Discharge, Positive = Charge)')
    axes[1].set_ylabel('Voltage [V]')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '05_current_vs_temperature.png'), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 6: 06_cooling_flow_vs_temperature.png
    # -------------------------------------------------------------
    # Compare 3 flow rates (1.5, 0.5, 0.1 L/min) under identical discharge currents at 15°C
    # -40A: ID11 (1.5), ID14 (0.5), ID17 (0.1)
    # -80A: ID12 (1.5), ID15 (0.5), ID18 (0.1)
    # -120A: ID13 (1.5), ID16 (0.5), ID19 (0.1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Figure 6: Cooling Flow Rate Impact on Thermal Performance at 15°C Coolant', fontsize=14, fontweight='bold')
    
    matrix_data = [
        {'Current': '-40A', 'Flow': 1.5, 'Run': 'ID11', 'Peak_T': runs['ID11']['df']['T_cell_max_C'].max(), 'Spread': runs['ID11']['df']['T_cell_spread_C'].max()},
        {'Current': '-40A', 'Flow': 0.5, 'Run': 'ID14', 'Peak_T': runs['ID14']['df']['T_cell_max_C'].max(), 'Spread': runs['ID14']['df']['T_cell_spread_C'].max()},
        {'Current': '-40A', 'Flow': 0.1, 'Run': 'ID17', 'Peak_T': runs['ID17']['df']['T_cell_max_C'].max(), 'Spread': runs['ID17']['df']['T_cell_spread_C'].max()},
        {'Current': '-80A', 'Flow': 1.5, 'Run': 'ID12', 'Peak_T': runs['ID12']['df']['T_cell_max_C'].max(), 'Spread': runs['ID12']['df']['T_cell_spread_C'].max()},
        {'Current': '-80A', 'Flow': 0.5, 'Run': 'ID15', 'Peak_T': runs['ID15']['df']['T_cell_max_C'].max(), 'Spread': runs['ID15']['df']['T_cell_spread_C'].max()},
        {'Current': '-80A', 'Flow': 0.1, 'Run': 'ID18', 'Peak_T': runs['ID18']['df']['T_cell_max_C'].max(), 'Spread': runs['ID18']['df']['T_cell_spread_C'].max()},
        {'Current': '-120A', 'Flow': 1.5, 'Run': 'ID13', 'Peak_T': runs['ID13']['df']['T_cell_max_C'].max(), 'Spread': runs['ID13']['df']['T_cell_spread_C'].max()},
        {'Current': '-120A', 'Flow': 0.5, 'Run': 'ID16', 'Peak_T': runs['ID16']['df']['T_cell_max_C'].max(), 'Spread': runs['ID16']['df']['T_cell_spread_C'].max()},
        {'Current': '-120A', 'Flow': 0.1, 'Run': 'ID19', 'Peak_T': runs['ID19']['df']['T_cell_max_C'].max(), 'Spread': runs['ID19']['df']['T_cell_spread_C'].max()},
    ]
    df_mat = pd.DataFrame(matrix_data)
    
    sns.barplot(data=df_mat, x='Current', y='Peak_T', hue='Flow', palette='coolwarm_r', ax=axes[0])
    axes[0].set_title('Peak Cell Temperature vs Coolant Flow Rate', fontweight='bold')
    axes[0].set_ylabel('Peak T_cell_max [°C]')
    axes[0].axhline(70, color='red', linestyle='--', label='70°C Safety Threshold')
    axes[0].legend(title='Flow [L/min]')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    sns.barplot(data=df_mat, x='Current', y='Spread', hue='Flow', palette='coolwarm_r', ax=axes[1])
    axes[1].set_title('Maximum Temperature Non-Uniformity (Spread) vs Flow Rate', fontweight='bold')
    axes[1].set_ylabel('Max T_cell_spread [°C]')
    axes[1].legend(title='Flow [L/min]')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '06_cooling_flow_vs_temperature.png'), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 7: 07_coolant_temperature_rise.png
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Figure 7: Coolant Temperature Rise (ΔT_coolant = T_outlet - T_inlet)', fontsize=14, fontweight='bold')
    
    # DeltaT_coolant vs flow rate
    axes[0].scatter(df_all['liquid_flow'], df_all['DeltaT_coolant_C'], alpha=0.2, color='teal', s=10)
    axes[0].set_title('Coolant Flow Rate vs Coolant Temperature Rise', fontweight='bold')
    axes[0].set_xlabel('Coolant Flow Rate [L/min]')
    axes[0].set_ylabel('ΔT_coolant = T_outlet - T_inlet [°C]')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # DeltaT_coolant vs Current
    axes[1].scatter(df_all['current'], df_all['DeltaT_coolant_C'], alpha=0.2, color='darkorange', s=10)
    axes[1].set_title('Current Load vs Coolant Temperature Rise', fontweight='bold')
    axes[1].set_xlabel('Current [A]')
    axes[1].set_ylabel('ΔT_coolant = T_outlet - T_inlet [°C]')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '07_coolant_temperature_rise.png'), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 8: 08_temperature_autocorrelation.png
    # -------------------------------------------------------------
    lags_sec = [5, 10, 15, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600]
    lags_steps = [int(s / 5) for s in lags_sec]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Figure 8: Temperature Autocorrelation Decay across Lags (Physical Thermal Inertia)', fontsize=14, fontweight='bold')
    
    autocorr_matrix = []
    for eid in exp_ids:
        s_t = runs[eid]['df']['T_cell_max_C']
        acs = [s_t.autocorr(ls) for ls in lags_steps]
        autocorr_matrix.append(acs)
        ax.plot(lags_sec, acs, color='gray', alpha=0.35, linewidth=1)
        
    mean_acs = np.nanmean(autocorr_matrix, axis=0)
    ax.plot(lags_sec, mean_acs, color='royalblue', linewidth=2.5, marker='o', label='Mean Autocorrelation Across Runs')
    
    # Mark candidate prediction horizons
    for hz, col in [(30, 'green'), (60, 'teal'), (120, 'orange'), (300, 'red')]:
        ax.axvline(hz, color=col, linestyle='--', alpha=0.7, label=f'Horizon {hz}s')
        
    ax.set_xlabel('Time Lag [Seconds]')
    ax.set_ylabel('Autocorrelation of T_cell_max')
    ax.set_ylim(-0.1, 1.05)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '08_temperature_autocorrelation.png'), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 9: 09_horizon_analysis.png
    # -------------------------------------------------------------
    # Distribution of Delta_T for each horizon: 30s, 60s, 120s, 300s
    horizons = {'30s': 6, '60s': 12, '120s': 24, '300s': 60}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Figure 9: Future Temperature Change Distributions ΔT(t+h) across Candidate Horizons', fontsize=14, fontweight='bold')
    
    for ax, (h_name, steps) in zip(axes.flatten(), horizons.items()):
        deltas = []
        for eid in exp_ids:
            df_e = runs[eid]['df']
            if len(df_e) > steps:
                d = df_e['T_cell_max_C'].iloc[steps:].values - df_e['T_cell_max_C'].iloc[:-steps].values
                deltas.extend(d.tolist())
        deltas = np.array(deltas)
        abs_d = np.abs(deltas)
        
        sns.histplot(deltas, ax=ax, bins=50, kde=True, color='darkslateblue')
        ax.set_title(f'Horizon {h_name} ({steps} steps) — Mean |ΔT| = {np.mean(abs_d):.3f}°C, Max |ΔT| = {np.max(abs_d):.2f}°C', fontweight='bold')
        ax.set_xlabel('ΔT = T_cell_max(t+h) - T_cell_max(t) [°C]')
        ax.set_ylabel('Count')
        ax.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '09_horizon_analysis.png'), dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # Figure 10: 10_feature_relationships.png
    # -------------------------------------------------------------
    # Correlation heatmap across representative features
    selected_cols = [
        'current', 'voltage', 'SOC', 'liquid_flow',
        'temp_ambient_C', 'temp_inlet_C', 'temp_outlet_C', 'DeltaT_coolant_C',
        'T_plate_mean_C', 'Cell 1_max_C', 'Cell 2_max_C', 'Cell 3_max_C', 'Cell 4_max_C',
        'T_cell_max_C', 'T_cell_spread_C'
    ]
    corr_matrix = df_all[selected_cols].corr()
    
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.suptitle('Figure 10: Cross-Feature Correlation Matrix (All 22,247 Observations)', fontsize=14, fontweight='bold')
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '10_feature_relationships.png'), dpi=150)
    plt.close()
    
    print(f"Generated all 10 figures successfully in {out_dir}")

def generate_eda_notebook(runs, df_quality, df_hottest, df_horizons):
    """Generate notebooks/07_freetwinev_eda.ipynb."""
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 1: FreeTwinEV Exploratory Data Analysis (EDA)\n",
                    "\n",
                    "**Course:** Advanced AI Techniques (AAI)  \n",
                    "**Project:** AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)  \n",
                    "**Dataset:** FreeTwinEV P45B 1s4p Battery Module Measurement  \n",
                    "\n",
                    "This notebook provides the detailed, empirical Phase 1 Exploratory Data Analysis to establish:\n",
                    "1. Rigorous Data Quality across all 14 physical runs\n",
                    "2. Temperature Distributions, Spatial Gradients, and Hottest Sensor Frequency Analysis\n",
                    "3. Electrical Dynamics (Current Sign Convention, Voltage Cutoffs, SOC)\n",
                    "4. Cooling System Dynamics (Flow Rate Impact, Inlet Temperature, Sensible Heat Rise)\n",
                    "5. Temporal Dynamics and Autocorrelation across Candidate Horizons (30s, 60s, 120s, 300s)\n",
                    "6. Cross-Channel Relationships and Dimensionality / Redundancy Analysis\n",
                    "7. Leakage Audit and Target Formulation Evaluation\n",
                    "\n",
                    "**Scientific Principle:** No models are trained in this phase. The goal is to establish the physical and mathematical foundation for Phase 2."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                    "sys.path.insert(0, str(project_root))\n",
                    "\n",
                    "from src.freetwinev_eda import load_all_runs, CELL_SENSORS, PLATE_SENSORS, CELL_MAP\n",
                    "\n",
                    "all_data_dir = project_root / 'data' / 'candidate_freetwinev' / '1s4p Battery Module Measurement' / 'AllData'\n",
                    "runs = load_all_runs(str(all_data_dir))\n",
                    "print(f'Successfully loaded {len(runs)} experimental runs.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Data Quality Across All 14 Experimental Runs"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.freetwinev_eda import compute_data_quality\n",
                    "df_quality, total_rows = compute_data_quality(runs)\n",
                    "print(f'Total observations across all 14 runs: {total_rows:,} rows')\n",
                    "df_quality[['Exp_ID', 'Data_Rows', 'Duration_min', 'Null_Count', 'Is_Monotonic', 'dt_median_s', 'Gaps_gt_7.5s', 'Current_Range_A', 'Cell_Temp_Range_C']]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Temperature Analysis & Hottest Sensor Frequency\n",
                    "\n",
                    "We inspect the spatial distribution across Cell 1 to Cell 4, along cell axial height, and check which sensor is the hottest."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.freetwinev_eda import compute_temperature_analysis\n",
                    "df_hottest, df_temp_summary = compute_temperature_analysis(runs)\n",
                    "print('Hottest Sensor Frequency Table:')\n",
                    "df_hottest[['Sensor', 'Cell', 'Axial_Position', 'Hottest_Count', 'Hottest_Percent', 'Overall_Mean_C', 'Overall_Max_C']]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Representative Trajectories: High-Stress ID13 vs Starved Cooling ID18"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig_path = project_root / 'results' / 'freetwinev' / 'figures' / '03_representative_temperature_timeseries.png'\n",
                    "from IPython.display import Image\n",
                    "Image(filename=str(fig_path))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Cooling Flow Rate Impact at Fixed Current (-40A, -80A, -120A)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig_path = project_root / 'results' / 'freetwinev' / 'figures' / '06_cooling_flow_vs_temperature.png'\n",
                    "Image(filename=str(fig_path))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Temporal Dynamics & Candidate Horizon Analysis (30s, 60s, 120s, 300s)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.freetwinev_eda import compute_temporal_dynamics\n",
                    "df_horizons, _ = compute_temporal_dynamics(runs)\n",
                    "df_horizons"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Autocorrelation Decay & Cross-Channel Relationships"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig_path = project_root / 'results' / 'freetwinev' / 'figures' / '10_feature_relationships.png'\n",
                    "Image(filename=str(fig_path))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    nb_path = 'notebooks/07_freetwinev_eda.ipynb'
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {nb_path}")

def run_full_eda(all_data_dir='data/candidate_freetwinev/1s4p Battery Module Measurement/AllData'):
    """Execute complete EDA pipeline and save outputs."""
    print("Loading all 14 FreeTwinEV runs...")
    runs = load_all_runs(all_data_dir)
    
    print("Computing data quality...")
    df_quality, total_rows = compute_data_quality(runs)
    df_quality.to_csv('results/freetwinev/eda_data_quality.csv', index=False)
    
    print("Computing temperature analysis...")
    df_hottest, df_temp_summary = compute_temperature_analysis(runs)
    df_hottest.to_csv('results/freetwinev/eda_hottest_sensors.csv', index=False)
    df_temp_summary.to_csv('results/freetwinev/eda_temperature_summary.csv', index=False)
    
    print("Computing temporal dynamics across horizons...")
    df_horizons, horizon_results = compute_temporal_dynamics(runs)
    df_horizons.to_csv('results/freetwinev/eda_horizons_summary.csv', index=False)
    
    print("Generating all 10 figures...")
    generate_all_10_figures(runs, df_horizons)
    
    print("Generating 07_freetwinev_eda.ipynb...")
    generate_eda_notebook(runs, df_quality, df_hottest, df_horizons)
    
    summary_data = {
        'total_runs': len(runs),
        'total_observations': total_rows,
        'hottest_sensor_top': df_hottest.iloc[0].to_dict(),
        'horizons': df_horizons.to_dict(orient='records'),
        'runs_overview': df_quality.to_dict(orient='records')
    }
    with open('results/freetwinev/eda_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print("Saved results/freetwinev/eda_summary.json")

if __name__ == '__main__':
    run_full_eda()
