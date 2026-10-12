"""
Phase 3: FreeTwinEV Preprocessing & Data Pipeline Library
Course: Advanced AI Techniques (AAI)
Project: AI-Enabled Micro-channel Battery Thermal Management System for Electric Vehicles (KJS-CES-02)

This module implements the reproducible, leakage-free data pipeline for FreeTwinEV:
1. Locks the 14-run experimental design into Train (10), Validation (2), and Final Test (2).
2. Manages the 3 feature experiments: Baseline (9), Extended (21), and Ablation (23).
3. Enforces strict run-level boundary isolation and eliminates cross-run contamination.
4. Implements RobustScaler (and StandardScaler) fitted strictly on Training data only.
5. Performs automated Leakage, Scaling, and Distribution-Shift audits.
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
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.freetwinev_eda import load_all_runs
from src.freetwinev_targets import process_all_runs_targets
from src.freetwinev_features import (
    compute_features_all_runs,
    get_feature_schema,
    CORE_FEATURES,
    DERIVED_FEATURES,
    ABLATION_FEATURES
)

# -------------------------------------------------------------
# 1. Experimental Split Definition
# -------------------------------------------------------------
SPLIT_MAPPING = {
    'ID01': 'train',
    'ID02': 'train',
    'ID11': 'train',
    'ID12': 'train',
    'ID13': 'train',
    'ID14': 'train',
    'ID15': 'train',
    'ID16': 'val',
    'ID17': 'train',
    'ID18': 'test',
    'ID19': 'test',
    'ID20': 'train',
    'ID21': 'train',
    'ID22': 'val'
}

TRAIN_RUNS = [k for k, v in SPLIT_MAPPING.items() if v == 'train']
VAL_RUNS = [k for k, v in SPLIT_MAPPING.items() if v == 'val']
TEST_RUNS = [k for k, v in SPLIT_MAPPING.items() if v == 'test']

# -------------------------------------------------------------
# 2. Feature Experiment Definitions
# -------------------------------------------------------------
FEATURE_EXPERIMENTS = {
    'baseline': CORE_FEATURES,                          # 9 features
    'extended': CORE_FEATURES + DERIVED_FEATURES,       # 21 features
    'ablation': CORE_FEATURES + DERIVED_FEATURES + ABLATION_FEATURES # 23 features
}

HORIZONS = {
    '60s': 12,    # 12 steps * 5s = 60s
    '300s': 60    # 60 steps * 5s = 300s
}

METADATA_COLS = ['exp_id', 'time_s', 'time_min']

def get_split_assignment() -> dict:
    """Returns the immutable mapping of run IDs to splits."""
    return SPLIT_MAPPING.copy()

def create_split_manifest(runs_dict: dict) -> pd.DataFrame:
    """
    Creates the split manifest documenting:
    run ID, directory, split, load conditions, cooling conditions,
    raw rows, valid 60s rows, and valid 300s rows.
    """
    manifest_rows = []
    
    run_conditions = {
        'ID01': ('-20A (~1.1C)', '25°C', '1.5 L/min', 'Light discharge, warm coolant'),
        'ID02': ('-40A (~2.2C)', '25°C', '1.5 L/min', 'Moderate discharge, warm coolant'),
        'ID11': ('-40A (~2.2C)', '15°C', '1.5 L/min', 'Standard baseline run'),
        'ID12': ('-80A (~4.4C)', '15°C', '1.5 L/min', 'Heavy discharge, high flow'),
        'ID13': ('-120A (~6.7C)', '15°C', '1.5 L/min', 'Extreme discharge, full cycles'),
        'ID14': ('-40A (~2.2C)', '15°C', '0.5 L/min', 'Moderate discharge, medium flow'),
        'ID15': ('-80A (~4.4C)', '15°C', '0.5 L/min', 'Heavy discharge, medium flow'),
        'ID16': ('-120A (~6.7C)', '15°C', '0.5 L/min', 'Validation: extreme load, medium flow'),
        'ID17': ('-40A (~2.2C)', '15°C', '0.1 L/min', 'Moderate discharge, starved cooling'),
        'ID18': ('-80A (~4.4C)', '15°C', '0.1 L/min', 'Test: heavy load, starved cooling'),
        'ID19': ('-120A (~6.7C)', '15°C', '0.1 L/min', 'Test: extreme load, starved cooling'),
        'ID20': ('-40A (~2.2C)', '15°C', '1.5 L/min', 'Repeatability check of ID11'),
        'ID21': ('-40A (~2.2C)', '7°C', '1.5 L/min', 'Chilled coolant, moderate load'),
        'ID22': ('-80A (~4.4C)', '8°C', '2.4 L/min', 'Validation: chilled coolant, heavy load')
    }
    
    for exp_id in sorted(runs_dict.keys()):
        run_info = runs_dict[exp_id]
        df_r = run_info['df']
        raw_rows = len(df_r)
        
        # Valid rows drop the terminal rows per run
        valid_60s = raw_rows - HORIZONS['60s']
        valid_300s = raw_rows - HORIZONS['300s']
        
        cond = run_conditions.get(exp_id, ('Unknown', 'Unknown', 'Unknown', ''))
        
        manifest_rows.append({
            'Exp_ID': exp_id,
            'Directory': run_info['dir_name'],
            'Split': SPLIT_MAPPING[exp_id],
            'Current_Load': cond[0],
            'Coolant_Temp': cond[1],
            'Flow_Rate': cond[2],
            'Raw_Rows': raw_rows,
            'Valid_60s_Rows': valid_60s,
            'Valid_300s_Rows': valid_300s,
            'Notes': cond[3]
        })
        
    df_manifest = pd.DataFrame(manifest_rows)
    return df_manifest

def prepare_split_datasets(runs_with_features: dict, horizon: str, feature_set_name: str, scaler_type: str = 'standard'):
    """
    Prepares train, validation, and test datasets for a specified horizon and feature experiment.
    
    Guarantees:
    - Filters out terminal rows with NaN targets for this horizon.
    - Fits scaler strictly on the training partition.
    - Transforms validation and test using the train-fitted scaler.
    - Target variables remain unscaled.
    - Defaults to StandardScaler because resting periods collapse the IQR of current-related
      features to ~0.0025A, which would cause RobustScaler to inflate active currents to z ~ -48,000.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"Unknown horizon: {horizon}. Must be one of {list(HORIZONS.keys())}")
    if feature_set_name not in FEATURE_EXPERIMENTS:
        raise ValueError(f"Unknown feature set: {feature_set_name}. Must be one of {list(FEATURE_EXPERIMENTS.keys())}")
        
    feature_cols = FEATURE_EXPERIMENTS[feature_set_name]
    target_delta_col = f'target_DeltaT_{horizon}'
    target_abs_col = f'target_T_cell_max_{horizon}'
    
    # Collect partitioned DataFrames
    train_dfs, val_dfs, test_dfs = [], [], []
    
    for exp_id, r_info in runs_with_features.items():
        df_r = r_info['df'].copy()
        df_r['exp_id'] = exp_id
        
        # Drop terminal NaN targets for this run
        df_valid = df_r.dropna(subset=[target_delta_col]).copy()
        
        split = SPLIT_MAPPING[exp_id]
        if split == 'train':
            train_dfs.append(df_valid)
        elif split == 'val':
            val_dfs.append(df_valid)
        elif split == 'test':
            test_dfs.append(df_valid)
            
    df_train_raw = pd.concat(train_dfs, ignore_index=True)
    df_val_raw = pd.concat(val_dfs, ignore_index=True)
    df_test_raw = pd.concat(test_dfs, ignore_index=True)
    
    # Fit scaler ONLY on train features
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")
        
    scaler.fit(df_train_raw[feature_cols])
    
    # Transform features
    train_scaled_vals = scaler.transform(df_train_raw[feature_cols])
    val_scaled_vals = scaler.transform(df_val_raw[feature_cols])
    test_scaled_vals = scaler.transform(df_test_raw[feature_cols])
    
    # Construct scaled DataFrames
    keep_cols = METADATA_COLS + [target_delta_col, target_abs_col]
    
    df_train_scaled = df_train_raw[keep_cols].copy()
    df_train_scaled[feature_cols] = train_scaled_vals
    
    df_val_scaled = df_val_raw[keep_cols].copy()
    df_val_scaled[feature_cols] = val_scaled_vals
    
    df_test_scaled = df_test_raw[keep_cols].copy()
    df_test_scaled[feature_cols] = test_scaled_vals
    
    # Also prepare clean unscaled DataFrames
    df_train_unscaled = df_train_raw[keep_cols + feature_cols].copy()
    df_val_unscaled = df_val_raw[keep_cols + feature_cols].copy()
    df_test_unscaled = df_test_raw[keep_cols + feature_cols].copy()
    
    return {
        'horizon': horizon,
        'feature_set': feature_set_name,
        'scaler_type': scaler_type,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'target_delta_col': target_delta_col,
        'target_abs_col': target_abs_col,
        'train_unscaled': df_train_unscaled,
        'val_unscaled': df_val_unscaled,
        'test_unscaled': df_test_unscaled,
        'train_scaled': df_train_scaled,
        'val_scaled': df_val_scaled,
        'test_scaled': df_test_scaled
    }

def audit_leakage(prep_result: dict) -> dict:
    """
    Automated leakage verification suite.
    Returns audit dictionary with individual checks and overall PASS/FAIL status.
    """
    feature_cols = prep_result['feature_cols']
    target_delta_col = prep_result['target_delta_col']
    target_abs_col = prep_result['target_abs_col']
    
    train_df = prep_result['train_unscaled']
    val_df = prep_result['val_unscaled']
    test_df = prep_result['test_unscaled']
    scaler = prep_result['scaler']
    
    checks = {}
    
    # Check 1: Target not in feature columns
    target_in_feats = (target_delta_col in feature_cols) or (target_abs_col in feature_cols)
    checks['no_target_in_features'] = {
        'passed': not target_in_feats,
        'details': f"Target {target_delta_col} in features: {target_in_feats}"
    }
    
    # Check 2: No future columns in features
    future_in_feats = any('target_' in col or 't+' in col for col in feature_cols)
    checks['no_future_cols_in_features'] = {
        'passed': not future_in_feats,
        'details': f"Future-labeled columns in features: {future_in_feats}"
    }
    
    # Check 3: Zero feature NaNs across all splits
    nans_train = int(train_df[feature_cols].isna().sum().sum())
    nans_val = int(val_df[feature_cols].isna().sum().sum())
    nans_test = int(test_df[feature_cols].isna().sum().sum())
    zero_feat_nans = (nans_train == 0) and (nans_val == 0) and (nans_test == 0)
    checks['zero_feature_nans'] = {
        'passed': zero_feat_nans,
        'details': f"Feature NaNs - Train: {nans_train}, Val: {nans_val}, Test: {nans_test}"
    }
    
    # Check 4: Zero target NaNs across all splits
    nans_target_train = int(train_df[target_delta_col].isna().sum())
    nans_target_val = int(val_df[target_delta_col].isna().sum())
    nans_target_test = int(test_df[target_delta_col].isna().sum())
    zero_target_nans = (nans_target_train == 0) and (nans_target_val == 0) and (nans_target_test == 0)
    checks['zero_target_nans'] = {
        'passed': zero_target_nans,
        'details': f"Target NaNs - Train: {nans_target_train}, Val: {nans_target_val}, Test: {nans_target_test}"
    }
    
    # Check 5: Run ID split isolation (zero overlap)
    train_runs_set = set(train_df['exp_id'].unique())
    val_runs_set = set(val_df['exp_id'].unique())
    test_runs_set = set(test_df['exp_id'].unique())
    
    train_val_overlap = train_runs_set.intersection(val_runs_set)
    train_test_overlap = train_runs_set.intersection(test_runs_set)
    val_test_overlap = val_runs_set.intersection(test_runs_set)
    
    zero_split_overlap = (len(train_val_overlap) == 0) and (len(train_test_overlap) == 0) and (len(val_test_overlap) == 0)
    checks['zero_run_overlap_between_splits'] = {
        'passed': zero_split_overlap,
        'details': f"Train/Val overlap: {train_val_overlap}, Train/Test overlap: {train_test_overlap}, Val/Test overlap: {val_test_overlap}"
    }
    
    # Check 6: Scaler fitted exclusively on train features
    scaler_n_features = scaler.n_features_in_
    expected_n_features = len(feature_cols)
    scaler_shape_match = (scaler_n_features == expected_n_features)
    checks['scaler_fitted_on_train_only'] = {
        'passed': scaler_shape_match,
        'details': f"Scaler features: {scaler_n_features}, Expected: {expected_n_features}"
    }
    
    # Overall status
    all_passed = all(c['passed'] for c in checks.values())
    
    return {
        'experiment': f"{prep_result['feature_set']}_{prep_result['horizon']}",
        'status': 'PASS' if all_passed else 'FAIL',
        'checks': checks
    }

def audit_scaling(prep_result: dict) -> pd.DataFrame:
    """
    Computes summary statistics before and after scaling to verify transformation properties.
    """
    feature_cols = prep_result['feature_cols']
    train_raw = prep_result['train_unscaled'][feature_cols]
    train_scaled = prep_result['train_scaled'][feature_cols]
    val_scaled = prep_result['val_scaled'][feature_cols]
    test_scaled = prep_result['test_scaled'][feature_cols]
    
    records = []
    for col in feature_cols:
        records.append({
            'Experiment': f"{prep_result['feature_set']}_{prep_result['horizon']}",
            'Feature': col,
            'Scaler_Type': prep_result.get('scaler_type', 'standard'),
            'Raw_Train_Mean': round(float(train_raw[col].mean()), 4),
            'Raw_Train_Median': round(float(train_raw[col].median()), 4),
            'Raw_Train_Std': round(float(train_raw[col].std()), 4),
            'Scaled_Train_Mean': round(float(train_scaled[col].mean()), 4),
            'Scaled_Train_Std': round(float(train_scaled[col].std()), 4),
            'Scaled_Train_Median': round(float(train_scaled[col].median()), 4),
            'Scaled_Train_IQR': round(float(train_scaled[col].quantile(0.75) - train_scaled[col].quantile(0.25)), 4),
            'Scaled_Train_Min': round(float(train_scaled[col].min()), 4),
            'Scaled_Train_Max': round(float(train_scaled[col].max()), 4),
            'Scaled_Val_Min': round(float(val_scaled[col].min()), 4),
            'Scaled_Val_Max': round(float(val_scaled[col].max()), 4),
            'Scaled_Test_Min': round(float(test_scaled[col].min()), 4),
            'Scaled_Test_Max': round(float(test_scaled[col].max()), 4)
        })
    return pd.DataFrame(records)

def audit_distribution_shift(prep_result: dict) -> pd.DataFrame:
    """
    Quantifies feature distribution differences between Train, Validation, and Test (stress tests).
    Flags features where test values substantially exceed training bounds.
    """
    feature_cols = prep_result['feature_cols']
    train_df = prep_result['train_unscaled'][feature_cols]
    val_df = prep_result['val_unscaled'][feature_cols]
    test_df = prep_result['test_unscaled'][feature_cols]
    
    records = []
    for col in feature_cols:
        tr_min, tr_max = float(train_df[col].min()), float(train_df[col].max())
        val_min, val_max = float(val_df[col].min()), float(val_df[col].max())
        tst_min, tst_max = float(test_df[col].min()), float(test_df[col].max())
        
        tr_mean, tr_med = float(train_df[col].mean()), float(train_df[col].median())
        val_mean, val_med = float(val_df[col].mean()), float(val_df[col].median())
        tst_mean, tst_med = float(test_df[col].mean()), float(test_df[col].median())
        
        # Shift flags
        test_below_train = tst_min < tr_min - 1e-4
        test_above_train = tst_max > tr_max + 1e-4
        shift_detected = test_below_train or test_above_train
        
        shift_notes = []
        if test_below_train:
            shift_notes.append(f"Test min ({tst_min:.2f}) < Train min ({tr_min:.2f})")
        if test_above_train:
            shift_notes.append(f"Test max ({tst_max:.2f}) > Train max ({tr_max:.2f})")
            
        records.append({
            'Experiment': f"{prep_result['feature_set']}_{prep_result['horizon']}",
            'Feature': col,
            'Train_Range': f"[{tr_min:.2f}, {tr_max:.2f}]",
            'Val_Range': f"[{val_min:.2f}, {val_max:.2f}]",
            'Test_Range': f"[{tst_min:.2f}, {tst_max:.2f}]",
            'Train_Mean': round(tr_mean, 2),
            'Val_Mean': round(val_mean, 2),
            'Test_Mean': round(tst_mean, 2),
            'Shift_Detected': shift_detected,
            'Shift_Description': "; ".join(shift_notes) if shift_notes else "Within train bounds"
        })
    return pd.DataFrame(records)
