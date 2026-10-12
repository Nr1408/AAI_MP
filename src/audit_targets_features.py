"""
Phase 2: Target & Feature Audit Script
Course: Advanced AI Techniques (AAI)
Project: AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)

This script performs the empirical audit of the defined targets and features across all 14 runs.
It computes:
- Exact descriptive statistics for candidate targets (60s and 300s absolute and delta targets)
- Descriptive statistics for Core, Derived, and Ablation features
- Verification of run boundaries (confirming target shifting does not cross runs)
- Feature-target correlation analysis
- Saves diagnostic CSVs and JSON summary into results/freetwinev/
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import glob
import pandas as pd
import numpy as np

from src.freetwinev_eda import load_all_runs
from src.freetwinev_targets import process_all_runs_targets
from src.freetwinev_features import (
    compute_features_all_runs,
    get_feature_schema,
    CORE_FEATURES,
    DERIVED_FEATURES,
    ABLATION_FEATURES
)

def run_target_feature_audit():
    all_data_dir = 'data/candidate_freetwinev/1s4p Battery Module Measurement/AllData'
    print("1. Loading raw runs...")
    runs = load_all_runs(all_data_dir)
    
    print("2. Processing targets for 60s and 300s horizons with strict run-boundary enforcement...")
    horizons = {'60s': 12, '300s': 60}
    runs_with_targets = process_all_runs_targets(runs, horizons=horizons)
    
    print("3. Computing features (Core, Derived, Ablation)...")
    runs_with_features = compute_features_all_runs(runs_with_targets)
    
    # Export Feature Schema Table
    df_schema = get_feature_schema()
    df_schema.to_csv('results/freetwinev/target_feature_schema.csv', index=False)
    print("Saved results/freetwinev/target_feature_schema.csv")
    
    # 4. Run-Boundary Audit
    print("4. Auditing run boundaries...")
    boundary_records = []
    for exp_id, r_info in runs_with_features.items():
        df_r = r_info['df']
        n_rows = len(df_r)
        
        # Check NaNs in target columns at the end of each run
        nans_tgt_60s = df_r['target_DeltaT_60s'].isna().sum()
        nans_tgt_300s = df_r['target_DeltaT_300s'].isna().sum()
        
        # Verify that NaNs are strictly at the end of the run
        is_end_nan_60s = bool(df_r['target_DeltaT_60s'].iloc[-12:].isna().all()) and (df_r['target_DeltaT_60s'].iloc[:-12].isna().sum() == 0)
        is_end_nan_300s = bool(df_r['target_DeltaT_300s'].iloc[-60:].isna().all()) and (df_r['target_DeltaT_300s'].iloc[:-60].isna().sum() == 0)
        
        # Check NaNs in feature columns
        feat_cols = CORE_FEATURES + DERIVED_FEATURES + ABLATION_FEATURES
        feat_nans = df_r[feat_cols].isna().sum().sum()
        
        boundary_records.append({
            'Exp_ID': exp_id,
            'Total_Rows': n_rows,
            'Target_60s_Expected_NaNs': 12,
            'Target_60s_Observed_NaNs': int(nans_tgt_60s),
            'Target_60s_At_Run_End_Only': is_end_nan_60s,
            'Target_300s_Expected_NaNs': 60,
            'Target_300s_Observed_NaNs': int(nans_tgt_300s),
            'Target_300s_At_Run_End_Only': is_end_nan_300s,
            'Feature_NaN_Count': int(feat_nans)
        })
        
    df_boundary = pd.DataFrame(boundary_records)
    df_boundary.to_csv('results/freetwinev/run_boundary_audit.csv', index=False)
    print("Saved results/freetwinev/run_boundary_audit.csv")
    
    # 5. Concatenate all data for statistical summaries (preserving run boundary awareness)
    dfs_all = [r_info['df'] for r_info in runs_with_features.values()]
    df_full = pd.concat(dfs_all, ignore_index=True)
    
    # 6. Target Statistical Summary
    print("5. Computing target statistical summaries...")
    target_cols = [
        'target_T_cell_max_60s', 'target_DeltaT_60s',
        'target_T_cell_max_300s', 'target_DeltaT_300s'
    ]
    target_stats = []
    for t_col in target_cols:
        s = df_full[t_col].dropna()
        target_stats.append({
            'Target': t_col,
            'Horizon': '60s' if '60s' in t_col else '300s',
            'Type': 'Delta' if 'DeltaT' in t_col else 'Absolute',
            'Valid_Observations': len(s),
            'Mean': round(float(s.mean()), 4),
            'Std': round(float(s.std()), 4),
            'Min': round(float(s.min()), 4),
            'P25': round(float(s.quantile(0.25)), 4),
            'Median': round(float(s.median()), 4),
            'P75': round(float(s.quantile(0.75)), 4),
            'P95': round(float(s.quantile(0.95)), 4),
            'Max': round(float(s.max()), 4),
            'Mean_Abs_Value': round(float(s.abs().mean()), 4)
        })
    df_target_stats = pd.DataFrame(target_stats)
    df_target_stats.to_csv('results/freetwinev/target_summary_stats.csv', index=False)
    print("Saved results/freetwinev/target_summary_stats.csv")
    
    # 7. Feature Statistical Summary
    print("6. Computing feature statistical summaries...")
    all_feat_cols = CORE_FEATURES + DERIVED_FEATURES + ABLATION_FEATURES
    feat_stats = []
    for f_col in all_feat_cols:
        s = df_full[f_col]
        cat = 'Core' if f_col in CORE_FEATURES else ('Derived' if f_col in DERIVED_FEATURES else 'Ablation')
        feat_stats.append({
            'Feature': f_col,
            'Category': cat,
            'Valid_Observations': len(s.dropna()),
            'Null_Count': int(s.isna().sum()),
            'Mean': round(float(s.mean()), 4),
            'Std': round(float(s.std()), 4),
            'Min': round(float(s.min()), 4),
            'P25': round(float(s.quantile(0.25)), 4),
            'Median': round(float(s.median()), 4),
            'P75': round(float(s.quantile(0.75)), 4),
            'Max': round(float(s.max()), 4)
        })
    df_feat_stats = pd.DataFrame(feat_stats)
    df_feat_stats.to_csv('results/freetwinev/feature_summary_stats.csv', index=False)
    print("Saved results/freetwinev/feature_summary_stats.csv")
    
    # 8. Feature-Target Correlations
    print("7. Computing feature-target correlations...")
    corr_records = []
    for f_col in all_feat_cols:
        cat = 'Core' if f_col in CORE_FEATURES else ('Derived' if f_col in DERIVED_FEATURES else 'Ablation')
        r_delta_60 = float(df_full[[f_col, 'target_DeltaT_60s']].dropna().corr().iloc[0, 1])
        r_abs_60 = float(df_full[[f_col, 'target_T_cell_max_60s']].dropna().corr().iloc[0, 1])
        r_delta_300 = float(df_full[[f_col, 'target_DeltaT_300s']].dropna().corr().iloc[0, 1])
        r_abs_300 = float(df_full[[f_col, 'target_T_cell_max_300s']].dropna().corr().iloc[0, 1])
        
        corr_records.append({
            'Feature': f_col,
            'Category': cat,
            'Corr_DeltaT_60s': round(r_delta_60, 4),
            'Corr_T_cell_max_60s': round(r_abs_60, 4),
            'Corr_DeltaT_300s': round(r_delta_300, 4),
            'Corr_T_cell_max_300s': round(r_abs_300, 4)
        })
    df_corr = pd.DataFrame(corr_records)
    df_corr.to_csv('results/freetwinev/feature_target_correlations.csv', index=False)
    print("Saved results/freetwinev/feature_target_correlations.csv")
    
    # 9. Generate Phase 2 Notebook
    print("8. Generating notebooks/08_target_feature_definition.ipynb...")
    generate_phase2_notebook()
    
    print("\nTarget & Feature Audit Complete!")
    return df_boundary, df_target_stats, df_feat_stats, df_corr

def generate_phase2_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 2: Target & Feature Definition\n",
                    "\n",
                    "**Course:** Advanced AI Techniques (AAI)  \n",
                    "**Project:** AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)  \n",
                    "**Dataset:** FreeTwinEV P45B 1s4p Battery Module Measurement  \n",
                    "\n",
                    "This notebook establishes the formal mathematical definitions and reproducible pipeline for:\n",
                    "1. Target Variables: Absolute Future Peak Temperature $T_{\\text{cell,max}}(t+h)$ vs. Future Temperature Rise $\\Delta T_h(t)$\n",
                    "2. Prediction Horizons: Primary $h=60\\text{s}$ (12 steps) and Secondary $h=300\\text{s}$ (60 steps)\n",
                    "3. Core Baseline Features: Compact, physically measurable inputs available at decision time $t$\n",
                    "4. Derived Features: Non-linear Joule heating proxies, rolling load integrals, and thermal velocity\n",
                    "5. Ablation Features: Downstream calorimetric signals ($\\text{temp\\_outlet}$) isolated for separate sensitivity testing\n",
                    "6. Run-Boundary Audit: Empirical proof that shifting and rolling operations never cross experimental boundaries\n",
                    "\n",
                    "**Strict Rule:** No ML models are trained in this phase."
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
                    "from src.freetwinev_eda import load_all_runs\n",
                    "from src.freetwinev_targets import process_all_runs_targets\n",
                    "from src.freetwinev_features import (\n",
                    "    compute_features_all_runs,\n",
                    "    get_feature_schema,\n",
                    "    CORE_FEATURES,\n",
                    "    DERIVED_FEATURES,\n",
                    "    ABLATION_FEATURES\n",
                    ")\n",
                    "\n",
                    "all_data_dir = project_root / 'data' / 'candidate_freetwinev' / '1s4p Battery Module Measurement' / 'AllData'\n",
                    "print('Loading 14 physical runs from:', all_data_dir)\n",
                    "runs = load_all_runs(str(all_data_dir))\n",
                    "print(f'Loaded {len(runs)} experimental runs successfully.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Feature Taxonomy and Specification Schema\n",
                    "\n",
                    "The complete feature schema defining categories, transformations, window lengths, and leakage status."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_schema = get_feature_schema()\n",
                    "print(f'Total features defined: {len(df_schema)}')\n",
                    "print(f'- Core Baseline Features: {len(CORE_FEATURES)}')\n",
                    "print(f'- Derived Features: {len(DERIVED_FEATURES)}')\n",
                    "print(f'- Ablation Features: {len(ABLATION_FEATURES)}')\n",
                    "df_schema[['Feature', 'Category', 'Unit', 'Availability', 'Window', 'Leakage_Status']]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Target Construction with Strict Run-Boundary Enforcement\n",
                    "\n",
                    "We construct future targets for $h = 60\\text{s}$ (12 steps) and $h = 300\\text{s}$ (60 steps).  \n",
                    "Each experiment is processed independently so that shifting never crosses from one experiment to the next."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "horizons = {'60s': 12, '300s': 60}\n",
                    "runs_with_targets = process_all_runs_targets(runs, horizons=horizons)\n",
                    "runs_featured = compute_features_all_runs(runs_with_targets)\n",
                    "\n",
                    "# Load run boundary audit results\n",
                    "df_boundary = pd.read_csv(project_root / 'results' / 'freetwinev' / 'run_boundary_audit.csv')\n",
                    "df_boundary"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Statistical Analysis of Candidate Targets\n",
                    "\n",
                    "Comparison of Absolute Future Temperature $T_{\\text{cell,max}}(t+h)$ vs Future Temperature Rise $\\Delta T_h(t)$."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_tgt_stats = pd.read_csv(project_root / 'results' / 'freetwinev' / 'target_summary_stats.csv')\n",
                    "df_tgt_stats"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Feature Summary Statistics across All Observations\n",
                    "\n",
                    "Summary statistics for all Core, Derived, and Ablation features."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_feat_stats = pd.read_csv(project_root / 'results' / 'freetwinev' / 'feature_summary_stats.csv')\n",
                    "df_feat_stats"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Feature-Target Correlations\n",
                    "\n",
                    "Correlations between input features and candidate targets, highlighting the physical predictive signal for $\\Delta T_h$."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_corr = pd.read_csv(project_root / 'results' / 'freetwinev' / 'feature_target_correlations.csv')\n",
                    "df_corr.sort_values(by='Corr_DeltaT_60s', key=abs, ascending=False)"
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
    nb_path = 'notebooks/08_target_feature_definition.ipynb'
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {nb_path}")

if __name__ == '__main__':
    run_target_feature_audit()
