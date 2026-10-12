"""Temporal diagnostics for the frozen Phase 4-6 forecasting models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

EXTENDED_21_FEATURES = [
    "current", "voltage", "SOC", "liquid_flow", "temp_inlet_C",
    "temp_ambient_C", "T_cell_max", "T_cell_spread", "T_plate_mean",
    "current_squared", "current_abs", "power_gross", "current_mean_60s",
    "current_mean_300s", "current_sq_mean_60s", "current_sq_mean_300s",
    "T_cell_max_diff_30s", "T_cell_max_diff_60s", "T_cell_to_plate_diff",
    "T_cell_to_inlet_diff", "flow_x_inlet_temp",
]
TRAIN_RUNS = ["ID01", "ID02", "ID11", "ID12", "ID13", "ID14", "ID15", "ID17", "ID20", "ID21"]
VAL_RUNS = ["ID16", "ID22"]
TEST_RUNS = ["ID18", "ID19"]
KEY_COLUMNS = ["exp_id", "time_s", "time_min"]
HORIZONS = ["60s", "300s"]
SELECTED_PREDICTION_COLUMNS = {
    "60s": "pred_DeltaT_Ridge Regression (alpha=1.0)_Extended",
    "300s": "pred_DeltaT_Linear Regression_Extended",
}
SELECTED_RESIDUAL_COLUMNS = {
    "60s": "residual_Ridge Regression (alpha=1.0)_Extended",
    "300s": "residual_Linear Regression_Extended",
}
TARGET_COLUMNS = {"60s": "target_DeltaT_60s", "300s": "target_DeltaT_300s"}
ABS_TARGET_COLUMNS = {"60s": "target_T_cell_max_60s", "300s": "target_T_cell_max_300s"}
ROLLING_SAMPLES = 12


def _check_features(frame: pd.DataFrame) -> None:
    if list(frame[EXTENDED_21_FEATURES].columns) != EXTENDED_21_FEATURES:
        raise AssertionError("Extended feature order/count differs from frozen contract")


def _merge_features(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    feature_frame = features[KEY_COLUMNS + EXTENDED_21_FEATURES].copy()
    predictions["_time_key"] = predictions["time_s"].round(3)
    feature_frame["_time_key"] = feature_frame["time_s"].round(3)
    merge_keys = ["exp_id", "_time_key"]
    if feature_frame.duplicated(merge_keys).any():
        raise AssertionError("Duplicate feature rows in temporal input")
    merged = predictions.merge(
        feature_frame[merge_keys + EXTENDED_21_FEATURES],
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )
    merged = merged.drop(columns=["_time_key"])
    if merged[EXTENDED_21_FEATURES].isna().any().any():
        raise AssertionError("Temporal feature merge produced missing values")
    _check_features(merged)
    return merged


def load_validation_predictions(results_dir: Path, preprocessed_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(results_dir / "baseline_predictions_validation.csv")
    frames = []
    for horizon in HORIZONS:
        subset = source[source["horizon"] == horizon].copy()
        prediction = SELECTED_PREDICTION_COLUMNS[horizon]
        residual = SELECTED_RESIDUAL_COLUMNS[horizon]
        subset = subset[KEY_COLUMNS + ["T_cell_max", TARGET_COLUMNS[horizon], ABS_TARGET_COLUMNS[horizon], prediction, residual]].copy()
        subset = subset.rename(columns={
            "T_cell_max": "current_T_cell_max",
            TARGET_COLUMNS[horizon]: "target_actual",
            ABS_TARGET_COLUMNS[horizon]: "absolute_actual",
            prediction: "target_predicted",
            residual: "residual",
        })
        subset["absolute_predicted"] = subset["current_T_cell_max"] + subset["target_predicted"]
        subset["absolute_residual"] = subset["absolute_predicted"] - subset["absolute_actual"]
        subset["horizon"] = horizon
        subset["split"] = "validation"
        features = pd.read_csv(preprocessed_dir / f"extended_{horizon}_val.csv")
        frames.append(_merge_features(subset, features))
    return add_temporal_columns(pd.concat(frames, ignore_index=True))


def load_test_predictions(results_dir: Path, preprocessed_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(results_dir / "phase6_predictions_test.csv")
    frames = []
    for horizon in HORIZONS:
        subset = source[source["horizon"] == horizon].copy()
        subset = subset.rename(columns={
            "current_T_cell_max": "current_T_cell_max",
            "target_actual": "target_actual",
            "absolute_actual": "absolute_actual",
            "target_predicted": "target_predicted",
            "residual": "residual",
            "absolute_predicted": "absolute_predicted",
            "absolute_residual": "absolute_residual",
        })
        subset["split"] = "test_posthoc"
        features = pd.read_csv(preprocessed_dir / f"extended_{horizon}_test.csv")
        frames.append(_merge_features(subset, features))
    return add_temporal_columns(pd.concat(frames, ignore_index=True))


def add_temporal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["horizon", "exp_id", "time_s"]).reset_index(drop=True)
    frame["abs_error"] = frame["residual"].abs()
    frame["rolling_mae_60s"] = frame.groupby(["horizon", "exp_id"])["abs_error"].transform(
        lambda values: values.rolling(ROLLING_SAMPLES, min_periods=ROLLING_SAMPLES).mean()
    )
    frame["rolling_rmse_60s"] = frame.groupby(["horizon", "exp_id"])["residual"].transform(
        lambda values: np.sqrt((values ** 2).rolling(ROLLING_SAMPLES, min_periods=ROLLING_SAMPLES).mean())
    )
    frame["prediction_step_change"] = frame.groupby(["horizon", "exp_id"])["target_predicted"].diff()
    frame["actual_step_change"] = frame.groupby(["horizon", "exp_id"])["target_actual"].diff()
    frame["abs_thermal_change_30s"] = frame["T_cell_max_diff_30s"].abs()
    frame["abs_power_gross"] = frame["power_gross"].abs()
    return frame


def validate_temporal_order(frame: pd.DataFrame, expected_runs: Iterable[str]) -> Dict[str, object]:
    checks = {
        "expected_runs": sorted(frame["exp_id"].unique().tolist()) == sorted(expected_runs),
        "no_duplicate_temporal_rows": not frame.duplicated(["exp_id", "time_s", "horizon"]).any(),
        "strict_time_order_within_run": True,
        "both_horizons_present": set(frame["horizon"].unique()) == set(HORIZONS),
        "no_missing_predictions": int(frame["target_predicted"].isna().sum()) == 0,
    }
    for (_, _), group in frame.groupby(["horizon", "exp_id"]):
        if not group["time_s"].is_monotonic_increasing or group["time_s"].duplicated().any():
            checks["strict_time_order_within_run"] = False
    if not all(checks.values()):
        raise AssertionError(f"Temporal ordering validation failed: {checks}")
    return {"status": "PASS", **checks}


def temporal_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, horizon, run_id), group in frame.groupby(["split", "horizon", "exp_id"]):
        group = group.sort_values("time_s")
        rows.append({
            "Split": split,
            "Horizon": horizon,
            "Run": run_id,
            "N": len(group),
            "MAE": mean_absolute_error(group["target_actual"], group["target_predicted"]),
            "RMSE": root_mean_squared_error(group["target_actual"], group["target_predicted"]),
            "R2": r2_score(group["target_actual"], group["target_predicted"]),
            "Bias": group["residual"].mean(),
            "Max_AE": group["abs_error"].max(),
            "P95_AE": group["abs_error"].quantile(0.95),
            "Mean_Rolling_MAE_60s": group["rolling_mae_60s"].mean(),
            "Max_Rolling_MAE_60s": group["rolling_mae_60s"].max(),
            "Prediction_Step_STD": group["prediction_step_change"].std(),
            "Actual_Step_STD": group["actual_step_change"].std(),
            "Prediction_Step_MAE": (group["prediction_step_change"] - group["actual_step_change"]).abs().mean(),
        })
    return pd.DataFrame(rows)


def regime_metrics(validation: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float]]:
    thresholds = {
        horizon: float(validation.loc[validation["horizon"] == horizon, "current_abs"].quantile(0.75))
        for horizon in HORIZONS
    }
    rows = []
    for split, frame in [("validation", validation), ("test_posthoc", test)]:
        for horizon in HORIZONS:
            subset = frame[frame["horizon"] == horizon].copy()
            variables = ["current_abs", "abs_power_gross", "SOC", "liquid_flow", "temp_inlet_C", "T_cell_max", "T_cell_spread", "abs_thermal_change_30s"]
            for variable in variables:
                rows.append({
                    "Split": split,
                    "Horizon": horizon,
                    "Analysis": "Spearman correlation",
                    "Variable": variable,
                    "Value": subset["abs_error"].corr(subset[variable], method="spearman"),
                    "N": len(subset),
                })
            high_load = subset[subset["current_abs"] >= thresholds[horizon]]
            rows.append({
                "Split": split,
                "Horizon": horizon,
                "Analysis": "High-load descriptive subset MAE",
                "Variable": "current_abs >= validation 75th percentile",
                "Value": high_load["abs_error"].mean(),
                "N": len(high_load),
            })
    return pd.DataFrame(rows), thresholds


def horizon_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, horizon), group in frame.groupby(["split", "horizon"]):
        rows.append({
            "Split": split,
            "Horizon": horizon,
            "N": len(group),
            "MAE": group["abs_error"].mean(),
            "RMSE": np.sqrt((group["residual"] ** 2).mean()),
            "Bias": group["residual"].mean(),
            "P95_AE": group["abs_error"].quantile(0.95),
            "Mean_Rolling_MAE_60s": group["rolling_mae_60s"].mean(),
            "Max_Rolling_MAE_60s": group["rolling_mae_60s"].max(),
        })
    return pd.DataFrame(rows)
