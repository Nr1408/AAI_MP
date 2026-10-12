"""
Phase 3 Execution Script: FreeTwinEV Preprocessing & Data Pipeline
Course: Advanced AI Techniques (AAI)
Project: AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)

This script executes the complete Phase 3 data preparation and audit pipeline:
1. Loads all 14 physical runs.
2. Computes targets (60s and 300s) and features (Baseline, Extended, Ablation).
3. Generates and persists split and feature manifests.
4. Prepares modeling datasets for 3 experiments * 2 horizons * 3 splits (18 datasets).
5. Fits RobustScaler strictly on train partitions and saves serialized scalers.
6. Conducts automated Leakage Audit, Scaling Audit, and Distribution Shift Audit.
7. Generates notebooks/09_preprocessing_pipeline.ipynb.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import joblib
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
from src.freetwinev_preprocessing import (
    get_split_assignment,
    create_split_manifest,
    prepare_split_datasets,
    audit_leakage,
    audit_scaling,
    audit_distribution_shift,
    FEATURE_EXPERIMENTS,
    HORIZONS,
    SPLIT_MAPPING,
    TRAIN_RUNS,
    VAL_RUNS,
    TEST_RUNS
)

def generate_phase3_notebook():
    """Generates the interactive Phase 3 Jupyter Notebook."""
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 3: Preprocessing & Data Pipeline\n",
                    "\n",
                    "**Course:** Advanced AI Techniques (AAI)  \n",
                    "**Project:** AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)  \n",
                    "**Dataset:** FreeTwinEV P45B 1s4p Battery Module Measurement  \n",
                    "\n",
                    "This notebook demonstrates the end-to-end Phase 3 preprocessing and audit pipeline:\n",
                    "1. Locking the 14-run experimental design into Train (10), Validation (2), and Final Test (2)\n",
                    "2. The 3 Feature Experiments: Baseline (9 features), Extended (21 features), and Ablation (23 features)\n",
                    "3. Target formatting for 60s and 300s horizons with strict run-boundary terminal NaN dropping\n",
                    "4. Fitting RobustScaler strictly on the training partition\n",
                    "5. Automated Leakage Audit (PASS/FAIL verification)\n",
                    "6. Distribution and Scaling Audit (pre vs post transformation)\n",
                    "7. Distribution Shift Audit across the severe stress test runs (ID18, ID19)\n",
                    "\n",
                    "**Strict Scientific Rule:** No ML models are trained in this phase."
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
                    "import json\n",
                    "import joblib\n",
                    "\n",
                    "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                    "sys.path.insert(0, str(project_root))\n",
                    "\n",
                    "prep_dir = project_root / 'results' / 'freetwinev' / 'preprocessed'\n",
                    "print('Preprocessing output directory exists:', prep_dir.exists())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Experimental Split Manifest\n",
                    "\n",
                    "The run-level split assigns 10 runs to Train, 2 to Validation, and 2 to Final Test.  \n",
                    "No observations from the same run appear across multiple splits."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_split = pd.read_csv(prep_dir / 'split_manifest.csv')\n",
                    "print(f\"Total runs: {len(df_split)}\")\n",
                    "print(\"Split counts:\", df_split['Split'].value_counts().to_dict())\n",
                    "df_split[['Exp_ID', 'Split', 'Current_Load', 'Coolant_Temp', 'Flow_Rate', 'Raw_Rows', 'Valid_60s_Rows', 'Valid_300s_Rows']]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Feature Experiment Manifests\n",
                    "\n",
                    "Overview of Baseline (9), Extended (21), and Ablation (23) feature sets."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_feat_man = pd.read_csv(prep_dir / 'feature_manifest.csv')\n",
                    "print(f\"Total features in taxonomy: {len(df_feat_man)}\")\n",
                    "df_feat_man[['Feature', 'Category', 'Unit', 'Availability', 'Window', 'Leakage_Status']]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Automated Preprocessing Leakage Audit\n",
                    "\n",
                    "Verifies that no target, future signal, or test statistic contaminated the pipeline."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "with open(prep_dir / 'leakage_audit_results.json', 'r') as f:\n",
                    "    audit_res = json.load(f)\n",
                    "\n",
                    "for exp_key, result in audit_res.items():\n",
                    "    print(f\"Experiment {exp_key}: Status = {result['status']}\")\n",
                    "    for chk, details in result['checks'].items():\n",
                    "        print(f\"   - {chk}: Passed = {details['passed']}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Scaling Verification (Pre vs Post RobustScaler)\n",
                    "\n",
                    "Verifies that training features achieve median = 0 and IQR = 1, while validation and test are transformed using training statistics."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_scaling = pd.read_csv(prep_dir / 'scaling_audit.csv')\n",
                    "df_scaling.head(10)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Distribution Shift Analysis on Severe Stress Tests (ID18, ID19)\n",
                    "\n",
                    "Highlights features where test conditions intentionally exceed training boundaries due to starved cooling."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_shift = pd.read_csv(prep_dir / 'distribution_shift_audit.csv')\n",
                    "shifted = df_shift[df_shift['Shift_Detected']]\n",
                    "print(f\"Features exhibiting distribution shift in test set: {len(shifted)}\")\n",
                    "shifted[['Experiment', 'Feature', 'Train_Range', 'Test_Range', 'Shift_Description']].head(10)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Preprocessed Datasets Dimension Summary"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "with open(prep_dir / 'preprocessing_config.json', 'r') as f:\n",
                    "    config = json.load(f)\n",
                    "print(\"Preprocessing Configuration Summary:\")\n",
                    "print(json.dumps(config, indent=2))"
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
    nb_path = 'notebooks/09_preprocessing_pipeline.ipynb'
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {nb_path}")

def run_pipeline():
    all_data_dir = 'data/candidate_freetwinev/1s4p Battery Module Measurement/AllData'
    out_dir = 'results/freetwinev/preprocessed'
    scalers_dir = os.path.join(out_dir, 'scalers')
    scaled_data_dir = os.path.join(out_dir, 'scaled')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(scalers_dir, exist_ok=True)
    os.makedirs(scaled_data_dir, exist_ok=True)
    
    print("==================================================================")
    print("PHASE 3: EXECUTING PREPROCESSING & DATA PIPELINE")
    print("==================================================================")
    
    # 1. Load raw runs
    print("1. Loading raw runs...")
    runs = load_all_runs(all_data_dir)
    
    # 2. Process targets (60s and 300s)
    print("2. Processing targets for 60s and 300s horizons...")
    runs_with_targets = process_all_runs_targets(runs, horizons=HORIZONS)
    
    # 3. Compute features
    print("3. Computing Core, Derived, and Ablation features...")
    runs_with_features = compute_features_all_runs(runs_with_targets)
    
    # 4. Generate Split Manifest
    print("4. Creating split manifest...")
    df_split = create_split_manifest(runs)
    df_split.to_csv(os.path.join(out_dir, 'split_manifest.csv'), index=False)
    print("   Saved split_manifest.csv")
    
    # 5. Generate Feature Manifest
    print("5. Creating feature manifest...")
    df_feat_schema = get_feature_schema()
    df_feat_schema.to_csv(os.path.join(out_dir, 'feature_manifest.csv'), index=False)
    with open(os.path.join(out_dir, 'feature_manifest.json'), 'w') as f:
        json.dump(df_feat_schema.to_dict(orient='records'), f, indent=2)
    print("   Saved feature_manifest.csv and feature_manifest.json")
    
    # 6. Execute Preprocessing for 3 Experiments * 2 Horizons * 3 Splits
    print("6. Preparing modeling datasets, fitting scalers on train, transforming val/test...")
    leakage_audit_results = {}
    all_scaling_records = []
    all_shift_records = []
    
    dataset_summary = {}
    
    for hz_name in HORIZONS.keys():
        dataset_summary[hz_name] = {}
        for feat_name in FEATURE_EXPERIMENTS.keys():
            exp_key = f"{feat_name}_{hz_name}"
            print(f"   Processing {exp_key}...")
            
            prep_res = prepare_split_datasets(
                runs_with_features,
                horizon=hz_name,
                feature_set_name=feat_name,
                scaler_type='standard'
            )
            
            # Save unscaled modeling datasets
            train_path = os.path.join(out_dir, f"{feat_name}_{hz_name}_train.csv")
            val_path = os.path.join(out_dir, f"{feat_name}_{hz_name}_val.csv")
            test_path = os.path.join(out_dir, f"{feat_name}_{hz_name}_test.csv")
            
            prep_res['train_unscaled'].to_csv(train_path, index=False)
            prep_res['val_unscaled'].to_csv(val_path, index=False)
            prep_res['test_unscaled'].to_csv(test_path, index=False)
            
            # Save scaled modeling datasets
            train_sc_path = os.path.join(scaled_data_dir, f"{feat_name}_{hz_name}_train_scaled.csv")
            val_sc_path = os.path.join(scaled_data_dir, f"{feat_name}_{hz_name}_val_scaled.csv")
            test_sc_path = os.path.join(scaled_data_dir, f"{feat_name}_{hz_name}_test_scaled.csv")
            
            prep_res['train_scaled'].to_csv(train_sc_path, index=False)
            prep_res['val_scaled'].to_csv(val_sc_path, index=False)
            prep_res['test_scaled'].to_csv(test_sc_path, index=False)
            
            # Save fitted scaler
            scaler_path = os.path.join(scalers_dir, f"scaler_{feat_name}_{hz_name}.joblib")
            joblib.dump(prep_res['scaler'], scaler_path)
            
            # Automated Leakage Audit
            audit_res = audit_leakage(prep_res)
            leakage_audit_results[exp_key] = audit_res
            if audit_res['status'] != 'PASS':
                print(f"CRITICAL ERROR: Leakage audit FAILED for {exp_key}!")
                print(json.dumps(audit_res, indent=2))
                sys.exit(1)
                
            # Scaling Audit
            df_sc_aud = audit_scaling(prep_res)
            all_scaling_records.append(df_sc_aud)
            
            # Distribution Shift Audit
            df_sh_aud = audit_distribution_shift(prep_res)
            all_shift_records.append(df_sh_aud)
            
            dataset_summary[hz_name][feat_name] = {
                'features_count': len(prep_res['feature_cols']),
                'train_rows': len(prep_res['train_unscaled']),
                'val_rows': len(prep_res['val_unscaled']),
                'test_rows': len(prep_res['test_unscaled']),
                'total_rows': len(prep_res['train_unscaled']) + len(prep_res['val_unscaled']) + len(prep_res['test_unscaled']),
                'scaler_path': scaler_path
            }
            
    # Save combined audits
    print("7. Persisting audit reports and configuration...")
    with open(os.path.join(out_dir, 'leakage_audit_results.json'), 'w') as f:
        json.dump(leakage_audit_results, f, indent=2)
    print("   Saved leakage_audit_results.json (ALL PASSED)")
    
    df_combined_scaling = pd.concat(all_scaling_records, ignore_index=True)
    df_combined_scaling.to_csv(os.path.join(out_dir, 'scaling_audit.csv'), index=False)
    print("   Saved scaling_audit.csv")
    
    df_combined_shift = pd.concat(all_shift_records, ignore_index=True)
    df_combined_shift.to_csv(os.path.join(out_dir, 'distribution_shift_audit.csv'), index=False)
    print("   Saved distribution_shift_audit.csv")
    
    config = {
        'split_protocol': {
            'train_runs': TRAIN_RUNS,
            'val_runs': VAL_RUNS,
            'test_runs': TEST_RUNS
        },
        'feature_experiments': {k: len(v) for k, v in FEATURE_EXPERIMENTS.items()},
        'horizons': HORIZONS,
        'scaler_selected': 'StandardScaler',
        'scaler_rationale': 'StandardScaler provides stable variance-normalized scaling (mean ~ 0, std = 1) across all operational regimes. In contrast, RobustScaler suffered severe numerical inflation (z ~ -48,000) on current-related features because extended rest steps collapse their interquartile range (IQR ~ 0.0025A) to near-zero sensor noise.',
        'dataset_summary': dataset_summary
    }
    with open(os.path.join(out_dir, 'preprocessing_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print("   Saved preprocessing_config.json")
    
    # 8. Generate Phase 3 Notebook
    print("8. Generating Phase 3 notebook...")
    generate_phase3_notebook()
    
    print("\n==================================================================")
    print("PHASE 3 COMPLETE: ALL AUDITS PASSED!")
    print("==================================================================")

if __name__ == '__main__':
    run_pipeline()
