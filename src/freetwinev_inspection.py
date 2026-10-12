import os
import sys
import glob
import json
import openpyxl
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def dump_protocol_excel(excel_path, out_json_path, out_txt_path):
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    protocol_data = {}
    with open(out_txt_path, 'w', encoding='utf-8') as f_txt:
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            sheet_rows = []
            f_txt.write(f"\n==============================\nSHEET: {sheet}\n==============================\n")
            for r in range(1, ws.max_row + 1):
                row_vals = [ws.cell(r, c).value for c in range(1, ws.max_column + 1)]
                while row_vals and row_vals[-1] is None:
                    row_vals.pop()
                if any(v is not None for v in row_vals):
                    cleaned = [str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for v in row_vals]
                    sheet_rows.append({'row': r, 'values': cleaned})
                    f_txt.write(f"Row {r:02d}: {cleaned}\n")
            protocol_data[sheet] = sheet_rows
    
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(protocol_data, f, indent=2, ensure_ascii=False)
    print(f"Dumped protocol sheets: {list(protocol_data.keys())} to {out_json_path} and {out_txt_path}")
    return protocol_data

def inspect_all_runs(all_data_dir):
    run_dirs = sorted(glob.glob(os.path.join(all_data_dir, "ID*")))
    print(f"Found {len(run_dirs)} experiment run directories.")
    
    run_summaries = []
    runs_table_rows = []
    
    for rdir in run_dirs:
        dir_name = os.path.basename(rdir)
        csv_files = glob.glob(os.path.join(rdir, "*.csv"))
        if not csv_files:
            continue
        csv_path = csv_files[0]
        csv_name = os.path.basename(csv_path)
        file_size_bytes = os.path.getsize(csv_path)
        
        # Read header and units
        df_raw = pd.read_csv(csv_path)
        raw_rows, raw_cols = df_raw.shape
        col_names = df_raw.columns.tolist()
        units_row = df_raw.iloc[0].to_dict()
        
        # Data rows (from row 1 onwards)
        df_data = df_raw.iloc[1:].copy()
        
        # Convert all columns to float
        for col in col_names:
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
        
        # Check missing values
        null_counts = df_data.isnull().sum().to_dict()
        total_nulls = sum(null_counts.values())
        
        # Time analysis
        time_col = col_names[0]  # 'timesamp'
        t = df_data[time_col].values
        dt = np.diff(t)
        
        is_monotonic = bool(np.all(dt > 0))
        num_duplicates = int(np.sum(dt == 0))
        num_negative_steps = int(np.sum(dt < 0))
        
        t_start = float(t[0])
        t_end = float(t[-1])
        duration_s = float(t_end - t_start)
        duration_min = duration_s / 60.0
        duration_h = duration_s / 3600.0
        
        dt_min = float(np.min(dt)) if len(dt) > 0 else 0.0
        dt_max = float(np.max(dt)) if len(dt) > 0 else 0.0
        dt_mean = float(np.mean(dt)) if len(dt) > 0 else 0.0
        dt_median = float(np.median(dt)) if len(dt) > 0 else 0.0
        dt_std = float(np.std(dt)) if len(dt) > 0 else 0.0
        
        # Identify gaps > 1.5 * median dt (e.g. > 7.5s)
        gaps_indices = np.where(dt > 1.5 * dt_median)[0]
        num_gaps = int(len(gaps_indices))
        max_gap_s = float(np.max(dt)) if len(dt) > 0 else 0.0
        
        # Electrical: current, voltage, SOC
        elec_stats = {}
        for c in ['current', 'voltage', 'SOC']:
            if c in df_data.columns:
                s = df_data[c]
                elec_stats[c] = {
                    'min': float(s.min()),
                    'max': float(s.max()),
                    'mean': float(s.mean()),
                    'std': float(s.std())
                }
        
        # Cooling: liquid_flow, temp_ambient, temp_inlet, temp_outlet
        cooling_stats = {}
        for c in ['liquid_flow', 'temp_ambient', 'temp_inlet', 'temp_outlet']:
            if c in df_data.columns:
                s = df_data[c]
                cooling_stats[c] = {
                    'min_K': float(s.min()),
                    'max_K': float(s.max()),
                    'mean_K': float(s.mean()),
                    'std_K': float(s.std()),
                    'min_C': float(s.min() - 273.15) if 'temp' in c else None,
                    'max_C': float(s.max() - 273.15) if 'temp' in c else None,
                    'mean_C': float(s.mean() - 273.15) if 'temp' in c else None,
                }
                
        # Cold plate sensors: temp_TH1 to temp_TH10
        plate_cols = [f'temp_TH{i}' for i in range(1, 11) if f'temp_TH{i}' in df_data.columns]
        plate_vals = df_data[plate_cols].values - 273.15  # to Celsius
        plate_stats = {
            'sensor_count': len(plate_cols),
            'min_C': float(np.nanmin(plate_vals)),
            'max_C': float(np.nanmax(plate_vals)),
            'mean_C': float(np.nanmean(plate_vals)),
        }
        
        # Cell sensors:
        # cell 1: temp_cell-1_TH20, temp_cell-1_TH21, temp_cell-1_TH22
        # cell 2: temp_cell-2_TH17, temp_cell-2_TH18, temp_cell-2_TH19
        # cell 3: temp_cell-3_TH14, temp_cell-3_TH15, temp_cell-3_TH16
        # cell 4: temp_cell-4_TH11, temp_cell-4_TH12, temp_cell-4_TH13
        cell_sensor_cols = [c for c in df_data.columns if 'cell' in c]
        cell_vals_C = df_data[cell_sensor_cols].values - 273.15
        
        # Max cell temperature across the 12 sensors at each timestep
        max_cell_temp_C = np.max(cell_vals_C, axis=1)
        min_cell_temp_C = np.min(cell_vals_C, axis=1)
        mean_cell_temp_C = np.mean(cell_vals_C, axis=1)
        
        # Per cell stats
        c1_cols = [c for c in cell_sensor_cols if 'cell-1' in c]
        c2_cols = [c for c in cell_sensor_cols if 'cell-2' in c]
        c3_cols = [c for c in cell_sensor_cols if 'cell-3' in c]
        c4_cols = [c for c in cell_sensor_cols if 'cell-4' in c]
        
        c1_max = np.max(df_data[c1_cols].values - 273.15, axis=1)
        c2_max = np.max(df_data[c2_cols].values - 273.15, axis=1)
        c3_max = np.max(df_data[c3_cols].values - 273.15, axis=1)
        c4_max = np.max(df_data[c4_cols].values - 273.15, axis=1)
        
        # Physical continuity: step-to-step delta in MaxCellTemp
        d_max_cell = np.diff(max_cell_temp_C)
        abs_d_max = np.abs(d_max_cell)
        
        continuity_stats = {
            'max_step_change_C': float(np.max(abs_d_max)),
            'mean_step_change_C': float(np.mean(abs_d_max)),
            'median_step_change_C': float(np.median(abs_d_max)),
            'rate_of_change_C_per_sec_max': float(np.max(abs_d_max / np.maximum(dt, 1e-3))),
            'rate_of_change_C_per_sec_mean': float(np.mean(abs_d_max / np.maximum(dt, 1e-3)))
        }
        
        # Autocorrelation of max cell temp at lags
        s_max_c = pd.Series(max_cell_temp_C)
        autocorrs = {
            'lag_1_step_5s': float(s_max_c.autocorr(1)),
            'lag_2_step_10s': float(s_max_c.autocorr(2)),
            'lag_6_step_30s': float(s_max_c.autocorr(6)),
            'lag_12_step_60s': float(s_max_c.autocorr(12)),
            'lag_24_step_120s': float(s_max_c.autocorr(24)),
            'lag_60_step_300s': float(s_max_c.autocorr(60)) if len(s_max_c) > 65 else None,
        }
        
        exp_id = dir_name.split('_')[0]
        
        summary = {
            'exp_id': exp_id,
            'run_dir': dir_name,
            'csv_name': csv_name,
            'file_size_bytes': file_size_bytes,
            'data_rows': len(df_data),
            'cols_count': raw_cols,
            'time_col_name': time_col,
            'units': units_row,
            'total_nulls': total_nulls,
            'is_monotonic': is_monotonic,
            'num_duplicates': num_duplicates,
            'num_negative_steps': num_negative_steps,
            'duration_s': duration_s,
            'duration_min': duration_min,
            'duration_h': duration_h,
            'dt_min': dt_min,
            'dt_max': dt_max,
            'dt_mean': dt_mean,
            'dt_median': dt_median,
            'dt_std': dt_std,
            'num_gaps_gt_7_5s': num_gaps,
            'max_gap_s': max_gap_s,
            'elec_stats': elec_stats,
            'cooling_stats': cooling_stats,
            'plate_stats': plate_stats,
            'cell_sensor_count': len(cell_sensor_cols),
            'max_cell_temp_C_overall_min': float(np.min(max_cell_temp_C)),
            'max_cell_temp_C_overall_max': float(np.max(max_cell_temp_C)),
            'max_cell_temp_C_overall_mean': float(np.mean(max_cell_temp_C)),
            'cell_temp_spread_C_mean': float(np.mean(max_cell_temp_C - min_cell_temp_C)),
            'cell_temp_spread_C_max': float(np.max(max_cell_temp_C - min_cell_temp_C)),
            'continuity_stats': continuity_stats,
            'autocorrs': autocorrs
        }
        run_summaries.append(summary)
        
        runs_table_rows.append({
            'Exp_ID': exp_id,
            'Directory': dir_name,
            'Rows': len(df_data),
            'Duration_min': round(duration_min, 1),
            'Sampling_Median_s': round(dt_median, 2),
            'Sampling_Max_s': round(dt_max, 2),
            'Gaps_gt_7.5s': num_gaps,
            'I_min_A': round(elec_stats['current']['min'], 1),
            'I_max_A': round(elec_stats['current']['max'], 1),
            'V_min_V': round(elec_stats['voltage']['min'], 2),
            'V_max_V': round(elec_stats['voltage']['max'], 2),
            'SOC_min_%': round(elec_stats['SOC']['min'], 1),
            'SOC_max_%': round(elec_stats['SOC']['max'], 1),
            'Flow_mean_Lmin': round(cooling_stats['liquid_flow']['mean_K'], 2),
            'Tinlet_mean_C': round(cooling_stats['temp_inlet']['mean_C'], 1),
            'Toutlet_mean_C': round(cooling_stats['temp_outlet']['mean_C'], 1),
            'Tplate_mean_C': round(plate_stats['mean_C'], 1),
            'Tcell_max_min_C': round(float(np.min(max_cell_temp_C)), 1),
            'Tcell_max_peak_C': round(float(np.max(max_cell_temp_C)), 1),
            'Tcell_spread_max_C': round(float(np.max(max_cell_temp_C - min_cell_temp_C)), 1),
            'dT_step_max_C': round(continuity_stats['max_step_change_C'], 3),
            'dT_step_mean_C': round(continuity_stats['mean_step_change_C'], 4),
            'Autocorr_Lag1_5s': round(autocorrs['lag_1_step_5s'], 5),
            'Autocorr_Lag12_60s': round(autocorrs['lag_12_step_60s'], 5),
            'Autocorr_Lag60_300s': round(autocorrs['lag_60_step_300s'], 5) if autocorrs['lag_60_step_300s'] is not None else None,
            'Null_Count': total_nulls
        })
        
    df_runs_overview = pd.DataFrame(runs_table_rows)
    df_runs_overview.to_csv('results/freetwinev/runs_overview.csv', index=False)
    print("Saved results/freetwinev/runs_overview.csv")
    
    return run_summaries, df_runs_overview

def generate_diagnostic_figures(all_data_dir, df_runs_overview):
    os.makedirs('results/freetwinev/figures', exist_ok=True)
    
    # Figure 1: Test Matrix Overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FreeTwinEV 1s4p Module Dataset — Experimental Runs Matrix', fontsize=14, fontweight='bold')
    
    # 1A: Discharge current distribution
    axes[0, 0].bar(df_runs_overview['Exp_ID'], df_runs_overview['I_min_A'], color='crimson')
    axes[0, 0].set_title('Minimum Current (Discharge Level in A)', fontweight='bold')
    axes[0, 0].set_ylabel('Current [A]')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 1B: Coolant inlet temperature & Flow
    ax2 = axes[0, 1].twinx()
    axes[0, 1].bar(df_runs_overview['Exp_ID'], df_runs_overview['Tinlet_mean_C'], color='royalblue', alpha=0.7, label='Inlet Temp [°C]')
    ax2.plot(df_runs_overview['Exp_ID'], df_runs_overview['Flow_mean_Lmin'], color='darkorange', marker='o', linewidth=2, label='Flow [L/min]')
    axes[0, 1].set_title('Coolant Conditions (Inlet Temp & Flow Rate)', fontweight='bold')
    axes[0, 1].set_ylabel('Inlet Temp [°C]')
    ax2.set_ylabel('Flow Rate [L/min]')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    # 1C: Duration per run
    axes[1, 0].bar(df_runs_overview['Exp_ID'], df_runs_overview['Duration_min'], color='seagreen')
    axes[1, 0].set_title('Experiment Duration (Minutes)', fontweight='bold')
    axes[1, 0].set_ylabel('Minutes')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 1D: Max Cell Temperature Range
    axes[1, 1].bar(df_runs_overview['Exp_ID'], df_runs_overview['Tcell_max_peak_C'], color='purple', alpha=0.8, label='Peak Max Cell Temp')
    axes[1, 1].axhline(70, color='red', linestyle='--', linewidth=1.5, label='70°C Cut-off Limit')
    axes[1, 1].set_title('Peak Cell Temperature Reached [°C]', fontweight='bold')
    axes[1, 1].set_ylabel('Temperature [°C]')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig('results/freetwinev/figures/01_runs_matrix_overview.png', dpi=150)
    plt.close()
    
    # Figure 2: Trajectory of a High-Stress Run (ID13: -120A, 15°C, 1.5 L/min)
    id13_path = glob.glob(os.path.join(all_data_dir, "ID13*/*.csv"))[0]
    df13 = pd.read_csv(id13_path)
    df13_data = df13.iloc[1:].apply(pd.to_numeric)
    t_min = df13_data['timesamp'] / 60.0
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('FreeTwinEV ID13 Experiment Trajectory (-120 A Discharge, 15°C Coolant, 1.5 L/min)', fontsize=14, fontweight='bold')
    
    # Subplot 1: Electrical (Current & Voltage)
    axes[0].plot(t_min, df13_data['current'], color='crimson', label='Current [A]')
    axes[0].set_ylabel('Current [A]', color='crimson')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    ax0_v = axes[0].twinx()
    ax0_v.plot(t_min, df13_data['voltage'], color='navy', linestyle='--', label='Voltage [V]')
    ax0_v.set_ylabel('Voltage [V]', color='navy')
    axes[0].set_title('Electrical Profile (Discharge, Rest, Charge Cycles)', fontweight='bold')
    
    # Subplot 2: Cooling Variables
    axes[1].plot(t_min, df13_data['temp_inlet'] - 273.15, color='blue', label='Inlet Temp [°C]')
    axes[1].plot(t_min, df13_data['temp_outlet'] - 273.15, color='cyan', label='Outlet Temp [°C]')
    axes[1].plot(t_min, df13_data['temp_ambient'] - 273.15, color='gray', linestyle=':', label='Ambient Temp [°C]')
    axes[1].set_ylabel('Coolant Temp [°C]')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    ax1_f = axes[1].twinx()
    ax1_f.plot(t_min, df13_data['liquid_flow'], color='darkorange', label='Flow [L/min]')
    ax1_f.set_ylabel('Flow [L/min]', color='darkorange')
    axes[1].set_title('Cooling Circuit Conditions', fontweight='bold')
    
    # Subplot 3: Cold Plate Temperatures (10 sensors)
    plate_cols = [f'temp_TH{i}' for i in range(1, 11)]
    for col in plate_cols:
        axes[2].plot(t_min, df13_data[col] - 273.15, alpha=0.6, linewidth=1)
    axes[2].set_ylabel('Plate Temp [°C]')
    axes[2].set_title('Cold Plate Surface Temperatures (Sensors TH1 to TH10)', fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    # Subplot 4: Cell Temperatures (12 sensors across 4 cells)
    cell_cols = [c for c in df13_data.columns if 'cell' in c]
    for col in cell_cols:
        if 'cell-1' in col:
            color = 'red'
        elif 'cell-2' in col:
            color = 'orange'
        elif 'cell-3' in col:
            color = 'green'
        else:
            color = 'blue'
        axes[3].plot(t_min, df13_data[col] - 273.15, color=color, alpha=0.5, linewidth=1)
    
    # Also plot max cell temp
    max_c_13 = np.max(df13_data[cell_cols].values - 273.15, axis=1)
    axes[3].plot(t_min, max_c_13, color='black', linewidth=1.5, linestyle='--', label='Max Cell Temp [°C]')
    axes[3].axhline(70, color='red', linestyle=':', label='70°C Safety Threshold')
    axes[3].set_ylabel('Cell Temp [°C]')
    axes[3].set_xlabel('Elapsed Time [Minutes]')
    axes[3].set_title('Cell Surface Temperatures (12 Axial Sensors across 4 Cells: C1=Red, C2=Orange, C3=Green, C4=Blue)', fontweight='bold')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig('results/freetwinev/figures/02_representative_run_id13.png', dpi=150)
    plt.close()
    
    # Figure 3: Physical Continuity Comparison: FreeTwinEV vs Rejected Dataset
    df_old = pd.read_csv('EV_Battery_Charging_TR_Dataset_with_Notes (1).csv')
    old_deltas = np.abs(np.diff(df_old['MaxTemp_C'].values))
    
    all_ft_deltas = []
    for rdir in sorted(glob.glob(os.path.join(all_data_dir, "ID*"))):
        csv_p = glob.glob(os.path.join(rdir, "*.csv"))[0]
        df_r = pd.read_csv(csv_p).iloc[1:].apply(pd.to_numeric)
        ccols = [c for c in df_r.columns if 'cell' in c]
        max_t = np.max(df_r[ccols].values - 273.15, axis=1)
        all_ft_deltas.extend(np.abs(np.diff(max_t)).tolist())
    all_ft_deltas = np.array(all_ft_deltas)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Physical Continuity Comparison: Rejected Dataset vs FreeTwinEV Dataset', fontsize=14, fontweight='bold')
    
    # Rejected dataset deltas
    axes[0].hist(old_deltas, bins=30, color='crimson', edgecolor='black', alpha=0.8)
    axes[0].set_title(f'REJECTED DATASET (1-min step)\nMean |ΔT| = {np.mean(old_deltas):.2f}°C, Max |ΔT| = {np.max(old_deltas):.2f}°C\n(Unphysical jumps across unrelated cells)', fontweight='bold')
    axes[0].set_xlabel('Absolute Step Temperature Change [°C]')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # FreeTwinEV deltas
    axes[1].hist(all_ft_deltas, bins=50, color='forestgreen', edgecolor='black', alpha=0.8)
    axes[1].set_title(f'FREETWINEV DATASET (5-sec step, N={len(all_ft_deltas)})\nMean |ΔT| = {np.mean(all_ft_deltas):.4f}°C, Max |ΔT| = {np.max(all_ft_deltas):.2f}°C\n(Smooth continuous physical thermal diffusion)', fontweight='bold')
    axes[1].set_xlabel('Absolute Step Temperature Change [°C]')
    axes[1].set_ylabel('Frequency')
    axes[1].set_yscale('log')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig('results/freetwinev/figures/03_physical_continuity_comparison.png', dpi=150)
    plt.close()
    
    # Figure 4: Autocorrelation Comparison across Lags
    lags_sec = [5, 10, 15, 30, 45, 60, 120, 180, 240, 300, 600]
    lags_steps = [int(s / 5) for s in lags_sec]
    
    autocorr_curves = []
    for rdir in sorted(glob.glob(os.path.join(all_data_dir, "ID*"))):
        csv_p = glob.glob(os.path.join(rdir, "*.csv"))[0]
        df_r = pd.read_csv(csv_p).iloc[1:].apply(pd.to_numeric)
        ccols = [c for c in df_r.columns if 'cell' in c]
        max_t = pd.Series(np.max(df_r[ccols].values - 273.15, axis=1))
        ac = [max_t.autocorr(ls) for ls in lags_steps]
        autocorr_curves.append(ac)
        
    mean_ac = np.nanmean(autocorr_curves, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, ac in enumerate(autocorr_curves):
        ax.plot(lags_sec, ac, color='gray', alpha=0.4, linewidth=1)
    ax.plot(lags_sec, mean_ac, color='royalblue', linewidth=2.5, marker='o', label='FreeTwinEV Mean Autocorrelation')
    
    old_s = df_old['MaxTemp_C']
    old_ac_lags_sec = [60, 120, 300]
    old_ac = [old_s.autocorr(1), old_s.autocorr(2), old_s.autocorr(5)]
    ax.plot(old_ac_lags_sec, old_ac, color='crimson', linewidth=2.5, marker='s', linestyle='--', label='Rejected Dataset Autocorrelation (~0.01)')
    
    ax.set_title('Thermal Autocorrelation vs Prediction Horizon (Time Lag)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Lag Horizon [Seconds]')
    ax.set_ylabel('Autocorrelation')
    ax.set_ylim(-0.3, 1.05)
    ax.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig('results/freetwinev/figures/04_autocorrelation_vs_horizon.png', dpi=150)
    plt.close()
    
    print("Generated 4 diagnostic figures in results/freetwinev/figures/")

def generate_inspection_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 0: FreeTwinEV Dataset Inspection and Assessment\n",
                    "\n",
                    "**Course:** Advanced AI Techniques (AAI)\n",
                    "**Project:** AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)\n",
                    "**Candidate Dataset:** FreeTwinEV P45B 1s4p Battery Module Measurement\n",
                    "\n",
                    "This notebook provides the technical inspection and diagnostics for the FreeTwinEV dataset across:\n",
                    "- Provenance and experimental protocol\n",
                    "- Physical structure of the 14 measurement runs\n",
                    "- Temporal continuity and sampling stability\n",
                    "- Temperature sensor mapping (ambient, coolant, cold plate, cell sensors)\n",
                    "- Electrical and cooling operating conditions\n",
                    "- Thermal continuity vs. the rejected dataset\n",
                    "- Future forecasting feasibility and train/test holdout strategy"
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
                    "import glob\n",
                    "import json\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                    "sys.path.insert(0, str(project_root))\n",
                    "\n",
                    "print('Project root:', project_root)\n",
                    "all_data_dir = project_root / 'data' / 'candidate_freetwinev' / '1s4p Battery Module Measurement' / 'AllData'\n",
                    "print('Candidate data exists:', all_data_dir.exists())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Experimental Runs Overview Table\n",
                    "\n",
                    "Overview of all 14 runs extracted from `results/freetwinev/runs_overview.csv`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_runs = pd.read_csv(project_root / 'results' / 'freetwinev' / 'runs_overview.csv')\n",
                    "display_cols = ['Exp_ID', 'Rows', 'Duration_min', 'Sampling_Median_s', 'I_min_A', 'Flow_mean_Lmin', 'Tinlet_mean_C', 'Tcell_max_peak_C', 'Tcell_spread_max_C', 'dT_step_mean_C', 'Autocorr_Lag1_5s', 'Autocorr_Lag12_60s']\n",
                    "df_runs[display_cols]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Experimental Setup and Channel Mapping\n",
                    "\n",
                    "Let us inspect the columns and units of a representative run (ID01)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sample_csv = list(all_data_dir.glob('ID01*/*.csv'))[0]\n",
                    "df_sample = pd.read_csv(sample_csv)\n",
                    "units = df_sample.iloc[0].to_dict()\n",
                    "print('Total columns:', len(df_sample.columns))\n",
                    "print('\\nChannel Units:')\n",
                    "for col, unit in units.items():\n",
                    "    print(f'  - {col}: {unit}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Physical Trajectory of a Representative Run (ID13)\n",
                    "\n",
                    "ID13 represents high-rate discharge (-120 A, ~6.7C), 15°C coolant, 1.5 L/min flow."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "id13_csv = list(all_data_dir.glob('ID13*/*.csv'))[0]\n",
                    "df13 = pd.read_csv(id13_csv).iloc[1:].apply(pd.to_numeric)\n",
                    "t_min = df13['timesamp'] / 60.0\n",
                    "\n",
                    "fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)\n",
                    "axes[0].plot(t_min, df13['current'], color='crimson', label='Current [A]')\n",
                    "axes[0].set_ylabel('Current [A]')\n",
                    "axes[0].legend(loc='upper right')\n",
                    "axes[0].grid(True, linestyle='--', alpha=0.5)\n",
                    "\n",
                    "axes[1].plot(t_min, df13['liquid_flow'], color='darkorange', label='Flow [L/min]')\n",
                    "axes[1].plot(t_min, df13['temp_inlet'] - 273.15, color='blue', label='Coolant Inlet [°C]')\n",
                    "axes[1].plot(t_min, df13['temp_outlet'] - 273.15, color='cyan', label='Coolant Outlet [°C]')\n",
                    "axes[1].set_ylabel('Cooling Conditions')\n",
                    "axes[1].legend(loc='upper right')\n",
                    "axes[1].grid(True, linestyle='--', alpha=0.5)\n",
                    "\n",
                    "cell_cols = [c for c in df13.columns if 'cell' in c]\n",
                    "for c in cell_cols:\n",
                    "    axes[2].plot(t_min, df13[c] - 273.15, alpha=0.4, linewidth=1)\n",
                    "axes[2].plot(t_min, np.max(df13[cell_cols].values - 273.15, axis=1), color='black', linewidth=2, label='Max Cell Temp [°C]')\n",
                    "axes[2].axhline(70, color='red', linestyle='--', label='70°C Cut-off')\n",
                    "axes[2].set_ylabel('Temperature [°C]')\n",
                    "axes[2].set_xlabel('Elapsed Time [min]')\n",
                    "axes[2].legend(loc='upper right')\n",
                    "axes[2].grid(True, linestyle='--', alpha=0.5)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Continuity Comparison: FreeTwinEV vs Rejected Dataset"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_old = pd.read_csv(project_root / 'EV_Battery_Charging_TR_Dataset_with_Notes (1).csv')\n",
                    "old_deltas = np.abs(np.diff(df_old['MaxTemp_C'].values))\n",
                    "print(f'Rejected dataset 1-minute temperature jump: Mean = {np.mean(old_deltas):.2f}°C, Max = {np.max(old_deltas):.2f}°C')\n",
                    "\n",
                    "ft_deltas = np.abs(np.diff(np.max(df13[cell_cols].values - 273.15, axis=1)))\n",
                    "print(f'FreeTwinEV (ID13) 5-second temperature change: Mean = {np.mean(ft_deltas):.4f}°C, Max = {np.max(ft_deltas):.2f}°C')"
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
    nb_path = 'notebooks/06_freetwinev_inspection.ipynb'
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {nb_path}")

if __name__ == '__main__':
    excel_path = 'data/candidate_freetwinev/1s4p Battery Module Measurement/Measurement_Protocol.xlsx'
    out_json = 'results/freetwinev/protocol_parsed.json'
    out_txt = 'results/freetwinev/protocol_readable.txt'
    dump_protocol_excel(excel_path, out_json, out_txt)
    
    all_data_dir = 'data/candidate_freetwinev/1s4p Battery Module Measurement/AllData'
    summaries, df_runs = inspect_all_runs(all_data_dir)
    
    with open('results/freetwinev/runs_inspection_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)
    print("Saved results/freetwinev/runs_inspection_summary.json")
    
    generate_diagnostic_figures(all_data_dir, df_runs)
    generate_inspection_notebook()
