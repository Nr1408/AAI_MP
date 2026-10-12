"""Frozen Phase 6 held-out evaluation for the FreeTwinEV project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

TRAIN_RUNS = ["ID01", "ID02", "ID11", "ID12", "ID13", "ID14", "ID15", "ID17", "ID20", "ID21"]
VAL_RUNS = ["ID16", "ID22"]
TEST_RUNS = ["ID18", "ID19"]

EXTENDED_21_FEATURES = [
    "current", "voltage", "SOC", "liquid_flow", "temp_inlet_C",
    "temp_ambient_C", "T_cell_max", "T_cell_spread", "T_plate_mean",
    "current_squared", "current_abs", "power_gross", "current_mean_60s",
    "current_mean_300s", "current_sq_mean_60s", "current_sq_mean_300s",
    "T_cell_max_diff_30s", "T_cell_max_diff_60s", "T_cell_to_plate_diff",
    "T_cell_to_inlet_diff", "flow_x_inlet_temp",
]

SELECTED_MODELS = {
    "60s": {
        "model": "Ridge Regression",
        "specification": "Ridge(alpha=1.0)",
        "target": "target_DeltaT_60s",
        "absolute_target": "target_T_cell_max_60s",
    },
    "300s": {
        "model": "Linear Regression",
        "specification": "LinearRegression()",
        "target": "target_DeltaT_300s",
        "absolute_target": "target_T_cell_max_300s",
    },
}

FROZEN_VALIDATION_EVIDENCE = {
    "60s": {"mae": 0.1598, "rmse": 0.7228, "r2": 0.7890},
    "300s": {"mae": 2.2582, "rmse": 4.5396, "r2": 0.5027},
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_temperature: np.ndarray,
    absolute_true: np.ndarray,
) -> Dict[str, float]:
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    absolute_pred = np.asarray(current_temperature, dtype=float) + np.asarray(y_pred, dtype=float)
    absolute_errors = absolute_pred - np.asarray(absolute_true, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(errors)),
        "Max_AE": float(np.max(np.abs(errors))),
        "Abs_MAE": float(np.mean(np.abs(absolute_errors))),
        "Abs_RMSE": float(np.sqrt(np.mean(absolute_errors ** 2))),
        "Abs_R2": float(r2_score(absolute_true, absolute_pred)),
        "Abs_Bias": float(np.mean(absolute_errors)),
        "Abs_Max_AE": float(np.max(np.abs(absolute_errors))),
    }


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_feature_manifest(manifest_path: Path) -> Dict[str, Any]:
    manifest = load_json(manifest_path)
    manifest_features = [row["Feature"] for row in manifest]
    if manifest_features[:21] != EXTENDED_21_FEATURES:
        raise AssertionError("Extended feature order differs from feature_manifest.json")
    if len(EXTENDED_21_FEATURES) != 21:
        raise AssertionError("Extended feature count is not 21")
    return {
        "status": "PASS",
        "feature_count": len(EXTENDED_21_FEATURES),
        "feature_columns": EXTENDED_21_FEATURES,
        "manifest_path": str(manifest_path),
    }


def verify_frozen_protocol(preprocessed_dir: Path, results_dir: Path) -> Dict[str, Any]:
    preprocessing = load_json(preprocessed_dir / "preprocessing_config.json")
    phase5_config = load_json(results_dir / "phase5_config.json")
    phase5_audit = load_json(results_dir / "phase5_audit.json")
    split = pd.read_csv(preprocessed_dir / "split_manifest.csv")
    split_map = dict(zip(split["Exp_ID"], split["Split"]))
    expected_map = {rid: "train" for rid in TRAIN_RUNS}
    expected_map.update({rid: "val" for rid in VAL_RUNS})
    expected_map.update({rid: "test" for rid in TEST_RUNS})
    if split_map != expected_map:
        raise AssertionError("split_manifest.csv does not match the frozen partition")
    if preprocessing["scaler_selected"] != "StandardScaler":
        raise AssertionError("Frozen preprocessing scaler is not StandardScaler")
    if phase5_config["train_runs"] != TRAIN_RUNS or phase5_config["val_runs"] != VAL_RUNS:
        raise AssertionError("Phase 5 partition configuration differs from frozen partition")
    if phase5_config["test_runs_locked"] != TEST_RUNS:
        raise AssertionError("Phase 5 locked test runs differ from frozen partition")
    if phase5_audit["overall_status"] != "PASS":
        raise AssertionError("Phase 5 audit is not PASS")
    return {
        "status": "PASS",
        "scaler": "StandardScaler",
        "train_runs": TRAIN_RUNS,
        "validation_runs": VAL_RUNS,
        "test_runs": TEST_RUNS,
        "split_manifest": str(preprocessed_dir / "split_manifest.csv"),
        "preprocessing_config": str(preprocessed_dir / "preprocessing_config.json"),
    }


def verify_selection_evidence(results_dir: Path) -> Dict[str, Any]:
    baseline = pd.read_csv(results_dir / "baseline_results.csv")
    required = {
        ("60s", "Ridge Regression (alpha=1.0)", "Extended (21 feats)"): FROZEN_VALIDATION_EVIDENCE["60s"],
        ("300s", "Linear Regression", "Extended (21 feats)"): FROZEN_VALIDATION_EVIDENCE["300s"],
    }
    evidence_rows = []
    for key, expected in required.items():
        horizon, model, feature_set = key
        row = baseline[(baseline["Horizon"] == horizon) & (baseline["Model"] == model) & (baseline["Feature_Set"] == feature_set)]
        if len(row) != 1:
            raise AssertionError(f"Missing frozen selection evidence for {key}")
        actual = row.iloc[0]
        for metric, column in [("mae", "Val_MAE"), ("rmse", "Val_RMSE"), ("r2", "Val_R2")]:
            if not np.isclose(float(actual[column]), expected[metric], atol=1e-4):
                raise AssertionError(f"Selection evidence mismatch for {key} {metric}")
        evidence_rows.append({
            "horizon": horizon,
            "model": model,
            "feature_set": feature_set,
            "validation_mae": float(actual["Val_MAE"]),
            "validation_rmse": float(actual["Val_RMSE"]),
            "validation_r2": float(actual["Val_R2"]),
        })
    return {
        "status": "PASS",
        "selection_rule": ["lowest validation MAE", "lowest validation RMSE", "highest validation R2"],
        "evidence_source": str(results_dir / "baseline_results.csv"),
        "evidence": evidence_rows,
        "selection_based_only_on_training_and_validation": True,
    }


def create_models() -> Dict[str, Any]:
    return {
        "60s": Ridge(alpha=1.0),
        "300s": LinearRegression(),
    }


def load_training_data(preprocessed_dir: Path, horizon: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = preprocessed_dir / "scaled" / f"extended_{horizon}_train_scaled.csv"
    train = pd.read_csv(path)
    if set(train["exp_id"].unique()) != set(TRAIN_RUNS):
        raise AssertionError(f"Unexpected training runs in {path}")
    if list(train[EXTENDED_21_FEATURES].columns) != EXTENDED_21_FEATURES:
        raise AssertionError("Training feature columns do not match Extended manifest")
    target = SELECTED_MODELS[horizon]["target"]
    return train, train[EXTENDED_21_FEATURES].to_numpy(), train[target].to_numpy()


def load_test_data(preprocessed_dir: Path, horizon: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    scaled_path = preprocessed_dir / "scaled" / f"extended_{horizon}_test_scaled.csv"
    raw_path = preprocessed_dir / f"extended_{horizon}_test.csv"
    scaled = pd.read_csv(scaled_path)
    raw = pd.read_csv(raw_path)
    if set(scaled["exp_id"].unique()) != set(TEST_RUNS) or set(raw["exp_id"].unique()) != set(TEST_RUNS):
        raise AssertionError("Test data contains unexpected run membership")
    alignment_cols = ["exp_id", "time_s", "time_min"]
    if not scaled[alignment_cols].reset_index(drop=True).equals(raw[alignment_cols].reset_index(drop=True)):
        raise AssertionError("Scaled and unscaled test rows are not aligned")
    if list(scaled[EXTENDED_21_FEATURES].columns) != EXTENDED_21_FEATURES:
        raise AssertionError("Test feature columns do not match Extended manifest")
    target = SELECTED_MODELS[horizon]["target"]
    return scaled, raw, scaled[target].to_numpy()


def build_pre_test_audit(preprocessed_dir: Path, results_dir: Path) -> Dict[str, Any]:
    feature_check = verify_feature_manifest(preprocessed_dir / "feature_manifest.json")
    protocol_check = verify_frozen_protocol(preprocessed_dir, results_dir)
    selection_check = verify_selection_evidence(results_dir)
    for horizon in ("60s", "300s"):
        train, _, _ = load_training_data(preprocessed_dir, horizon)
        if train[EXTENDED_21_FEATURES].isna().any().any() or train[SELECTED_MODELS[horizon]["target"]].isna().any():
            raise AssertionError(f"Missing training values for {horizon}")
    return {
        "status": "PASS",
        "test_rows_loaded": False,
        "test_predictions_generated": False,
        "selected_models": SELECTED_MODELS,
        "feature_check": feature_check,
        "protocol_check": protocol_check,
        "selection_check": selection_check,
        "preprocessing_frozen": True,
        "test_tuning": False,
        "test_metrics_used_for_selection": False,
    }
