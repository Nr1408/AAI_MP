"""Interpretable thermal-risk layer over frozen Phase 6/7 forecasts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

TRAIN_RUNS = ["ID01", "ID02", "ID11", "ID12", "ID13", "ID14", "ID15", "ID17", "ID20", "ID21"]
VAL_RUNS = ["ID16", "ID22"]
TEST_RUNS = ["ID18", "ID19"]
HORIZONS = ["60s", "300s"]
PERSISTENCE_SAMPLES = 12
SAMPLING_INTERVAL_SECONDS = 5
ENGINEERING_CUTOFF_C = 70.0
REQUIRED_COLUMNS = [
    "exp_id", "time_s", "time_min", "horizon", "split", "current_T_cell_max",
    "target_predicted", "T_cell_spread", "T_cell_max_diff_30s", "current_abs",
    "power_gross", "SOC", "liquid_flow", "temp_inlet_C", "absolute_predicted",
]


def load_phase7_predictions(path: Path, expected_split: str, expected_runs: Iterable[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise AssertionError(f"Missing Phase 7 risk inputs: {missing}")
    if set(frame["split"].unique()) != {expected_split}:
        raise AssertionError(f"Unexpected split labels: {frame['split'].unique().tolist()}")
    if set(frame["exp_id"].unique()) != set(expected_runs):
        raise AssertionError(f"Unexpected runs: {frame['exp_id'].unique().tolist()}")
    if set(frame["horizon"].unique()) != set(HORIZONS):
        raise AssertionError("Both frozen horizons are required")
    if frame.duplicated(["split", "exp_id", "time_s", "horizon"]).any():
        raise AssertionError("Duplicate Phase 7 temporal rows")
    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise AssertionError("Missing Phase 7 risk inputs")
    for (_, _), group in frame.groupby(["horizon", "exp_id"]):
        if not group["time_s"].is_monotonic_increasing:
            raise AssertionError("Non-monotonic temporal ordering")
    frame = frame.sort_values(["horizon", "exp_id", "time_s"]).reset_index(drop=True)
    frame["predicted_future_temperature"] = frame["current_T_cell_max"] + frame["target_predicted"]
    if not np.allclose(frame["predicted_future_temperature"], frame["absolute_predicted"], atol=1e-8):
        raise AssertionError("Future-temperature reconstruction differs from Phase 7")
    return frame


def derive_validation_thresholds(validation: pd.DataFrame) -> Dict[str, dict]:
    thresholds = {}
    for horizon, group in validation.groupby("horizon"):
        thresholds[horizon] = {
            "future_temperature_p75_c": float(group["predicted_future_temperature"].quantile(0.75)),
            "future_temperature_p90_c": float(group["predicted_future_temperature"].quantile(0.90)),
            "future_temperature_p99_c": float(group["predicted_future_temperature"].quantile(0.99)),
            "predicted_delta_p90_c": float(group["target_predicted"].quantile(0.90)),
            "recent_trend_p90_c": float(group["T_cell_max_diff_30s"].quantile(0.90)),
            "spread_p90_c": float(group["T_cell_spread"].quantile(0.90)),
            "source_rows": int(len(group)),
            "source_runs": sorted(group["exp_id"].unique().tolist()),
        }
    return {
        "method": "Validation-only descriptive percentiles",
        "validation_runs": VAL_RUNS,
        "temperature_units": "deg C",
        "engineering_cutoff_c": ENGINEERING_CUTOFF_C,
        "engineering_cutoff_source": "reports/freetwinev_dataset_assessment.md and reports/freetwinev_eda.md",
        "percentile_interpretation": "P75/P90/P99 describe the validation forecast distribution; they are not certified safety limits.",
        "persistence_samples": PERSISTENCE_SAMPLES,
        "persistence_seconds": PERSISTENCE_SAMPLES * SAMPLING_INTERVAL_SECONDS,
        "thresholds_by_horizon": thresholds,
    }


def apply_risk_logic(frame: pd.DataFrame, thresholds: Dict[str, dict]) -> pd.DataFrame:
    frame = frame.copy().sort_values(["horizon", "exp_id", "time_s"]).reset_index(drop=True)
    frame["temperature_band"] = "Normal"
    frame["temperature_band"] = np.where(frame.apply(lambda r: r["predicted_future_temperature"] >= thresholds["thresholds_by_horizon"][r["horizon"]]["future_temperature_p75_c"], axis=1), "Elevated", frame["temperature_band"])
    frame["temperature_band"] = np.where(frame.apply(lambda r: r["predicted_future_temperature"] >= thresholds["thresholds_by_horizon"][r["horizon"]]["future_temperature_p90_c"], axis=1), "High", frame["temperature_band"])
    frame["temperature_band"] = np.where(frame["predicted_future_temperature"] >= ENGINEERING_CUTOFF_C, "Critical", frame["temperature_band"])

    frame["rapid_predicted_rise"] = frame.apply(lambda r: r["target_predicted"] >= thresholds["thresholds_by_horizon"][r["horizon"]]["predicted_delta_p90_c"], axis=1)
    frame["rapid_recent_trend"] = frame.apply(lambda r: r["T_cell_max_diff_30s"] >= thresholds["thresholds_by_horizon"][r["horizon"]]["recent_trend_p90_c"], axis=1)
    frame["wide_cell_spread"] = frame.apply(lambda r: r["T_cell_spread"] >= thresholds["thresholds_by_horizon"][r["horizon"]]["spread_p90_c"], axis=1)
    frame["secondary_signal_count"] = frame[["rapid_predicted_rise", "rapid_recent_trend", "wide_cell_spread"]].sum(axis=1)
    elevated = frame["predicted_future_temperature"] >= frame["horizon"].map({h: thresholds["thresholds_by_horizon"][h]["future_temperature_p75_c"] for h in HORIZONS})
    frame["sustained_elevated_60s"] = elevated.groupby([frame["horizon"], frame["exp_id"]]).transform(
        lambda values: values.rolling(PERSISTENCE_SAMPLES, min_periods=PERSISTENCE_SAMPLES).sum() == PERSISTENCE_SAMPLES
    ).fillna(False)

    frame["risk_category"] = frame["temperature_band"]
    frame.loc[(frame["risk_category"] == "Normal") & ((frame["secondary_signal_count"] >= 1) | frame["sustained_elevated_60s"]), "risk_category"] = "Elevated"
    frame.loc[(frame["risk_category"] == "Elevated") & ((frame["secondary_signal_count"] >= 2) | frame["sustained_elevated_60s"]), "risk_category"] = "High"
    frame.loc[frame["predicted_future_temperature"] >= ENGINEERING_CUTOFF_C, "risk_category"] = "Critical"
    frame["risk_category_basis"] = "Validation-derived forecast-temperature bands plus documented 70 deg C cutoff; secondary flags are transparent escalation indicators."
    threshold_map = thresholds["thresholds_by_horizon"]
    for key, column in [("future_temperature_p75_c", "threshold_temperature_p75_c"), ("future_temperature_p90_c", "threshold_temperature_p90_c"), ("predicted_delta_p90_c", "threshold_delta_p90_c"), ("recent_trend_p90_c", "threshold_trend_p90_c"), ("spread_p90_c", "threshold_spread_p90_c")]:
        frame[column] = frame["horizon"].map({h: threshold_map[h][key] for h in HORIZONS})
    return frame


def risk_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    category_order = {"Normal": 0, "Elevated": 1, "High": 2, "Critical": 3}
    for (split, horizon, run_id), group in frame.groupby(["split", "horizon", "exp_id"]):
        group = group.sort_values("time_s").copy()
        severity = group["risk_category"].map(category_order)
        non_normal = severity > 0
        transitions = int((group["risk_category"] != group["risk_category"].shift()).sum() - 1)
        max_streak = 0
        current_streak = 0
        for value in non_normal:
            current_streak = current_streak + 1 if value else 0
            max_streak = max(max_streak, current_streak)
        counts = group["risk_category"].value_counts()
        rows.append({
            "Split": split,
            "Horizon": horizon,
            "Run": run_id,
            "N": len(group),
            "Max_Predicted_Future_Temperature_C": group["predicted_future_temperature"].max(),
            "Max_Predicted_DeltaT_C": group["target_predicted"].max(),
            "Max_T_cell_spread_C": group["T_cell_spread"].max(),
            "Max_Recent_Trend_C": group["T_cell_max_diff_30s"].max(),
            "Normal_Rows": int(counts.get("Normal", 0)),
            "Elevated_Rows": int(counts.get("Elevated", 0)),
            "High_Rows": int(counts.get("High", 0)),
            "Critical_Rows": int(counts.get("Critical", 0)),
            "Risk_Transitions": transitions,
            "Longest_NonNormal_Seconds": max_streak * SAMPLING_INTERVAL_SECONDS,
            "Any_Critical": bool((group["risk_category"] == "Critical").any()),
        })
    return pd.DataFrame(rows)
