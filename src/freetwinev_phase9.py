"""Validation-first decision-support cooling recommendations over frozen Phase 8 risk."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

TRAIN_RUNS = ["ID01", "ID02", "ID11", "ID12", "ID13", "ID14", "ID15", "ID17", "ID20", "ID21"]
VAL_RUNS = ["ID16", "ID22"]
TEST_RUNS = ["ID18", "ID19"]
HORIZONS = ["60s", "300s"]
ACTIVE_LOAD_MIN_CURRENT_ABS_A = 20.0
REQUIRED_COLUMNS = [
    "exp_id", "time_s", "time_min", "horizon", "split", "liquid_flow",
    "temp_inlet_C", "temp_ambient_C", "current_T_cell_max", "T_cell_max",
    "T_cell_spread", "current_abs", "power_gross", "target_predicted",
    "predicted_future_temperature", "risk_category", "temperature_band",
    "T_cell_max_diff_30s", "sustained_elevated_60s",
]


def load_risk_predictions(path: Path, expected_split: str, expected_runs: Iterable[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise AssertionError(f"Missing Phase 8 recommendation inputs: {missing}")
    if set(frame["split"].unique()) != {expected_split}:
        raise AssertionError("Unexpected split labels")
    if set(frame["exp_id"].unique()) != set(expected_runs):
        raise AssertionError("Unexpected run membership")
    if set(frame["horizon"].unique()) != set(HORIZONS):
        raise AssertionError("Both frozen horizons are required")
    if frame.duplicated(["split", "exp_id", "time_s", "horizon"]).any():
        raise AssertionError("Duplicate recommendation input rows")
    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise AssertionError("Missing required recommendation inputs")
    for (_, _), group in frame.groupby(["horizon", "exp_id"]):
        if not group["time_s"].is_monotonic_increasing:
            raise AssertionError("Non-monotonic temporal ordering")
    return frame.sort_values(["horizon", "exp_id", "time_s"]).reset_index(drop=True)


def derive_validation_config(validation: pd.DataFrame) -> dict:
    active = validation[(validation["current_abs"] >= ACTIVE_LOAD_MIN_CURRENT_ABS_A) & (validation["liquid_flow"] > 0.05)]
    if active.empty:
        raise AssertionError("No validation rows support the active-load flow reference")
    flow_reference = {
        horizon: float(group["liquid_flow"].quantile(0.75))
        for horizon, group in active.groupby("horizon")
    }
    return {
        "method": "Validation-derived decision-support rules over frozen Phase 8 risk",
        "validation_runs": VAL_RUNS,
        "engineering_cutoff_reference": "Phase 8 Critical risk category, based on documented 70 deg C cutoff",
        "active_load_min_current_abs_a": ACTIVE_LOAD_MIN_CURRENT_ABS_A,
        "active_load_assumption": "20 A is the lowest documented substantive discharge regime; this is an operating-regime assumption, not a certified control limit.",
        "low_flow_reference_method": "75th percentile of liquid_flow among validation rows with current_abs >= 20 A and liquid_flow > 0.05",
        "low_flow_reference_by_horizon": flow_reference,
        "rules": {
            "CRITICAL_PROTECTIVE_ACTION": "risk_category == Critical",
            "HIGH_COOLING_ATTENTION": "risk_category == High, or Elevated without validated low-flow active-load evidence",
            "INCREASE_COOLING": "risk_category == Elevated AND current_abs >= 20 A AND liquid_flow < validation low-flow reference",
            "MAINTAIN": "risk_category == Normal",
        },
        "causal_status": "Decision support only; no causal cooling-control effect is claimed.",
    }


def apply_recommendations(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = frame.copy().sort_values(["horizon", "exp_id", "time_s"]).reset_index(drop=True)
    frame["active_load"] = frame["current_abs"] >= config["active_load_min_current_abs_a"]
    frame["below_validation_flow_reference"] = frame.apply(
        lambda row: row["liquid_flow"] < config["low_flow_reference_by_horizon"][row["horizon"]], axis=1
    )
    frame["cooling_association_context"] = frame.apply(
        lambda row: "active_load_low_flow" if row["active_load"] and row["below_validation_flow_reference"] else "not_supported_by_validation_rule",
        axis=1,
    )
    frame["recommendation"] = "MAINTAIN"
    frame.loc[frame["risk_category"].isin(["Elevated", "High"]), "recommendation"] = "HIGH_COOLING_ATTENTION"
    frame.loc[
        (frame["risk_category"] == "Elevated")
        & frame["active_load"]
        & frame["below_validation_flow_reference"],
        "recommendation",
    ] = "INCREASE_COOLING"
    frame.loc[frame["risk_category"] == "Critical", "recommendation"] = "CRITICAL_PROTECTIVE_ACTION"
    frame["recommendation_basis"] = frame.apply(
        lambda row: {
            "MAINTAIN": "Normal predicted thermal state",
            "INCREASE_COOLING": "Elevated predicted risk with active load and below-reference validation flow",
            "HIGH_COOLING_ATTENTION": "High risk or elevated risk without sufficient validation support for an increase-flow action",
            "CRITICAL_PROTECTIVE_ACTION": "Frozen Phase 8 Critical risk category",
        }[row["recommendation"]], axis=1
    )
    frame["recommendation_transition"] = frame.groupby(["horizon", "exp_id"])["recommendation"].transform(lambda values: values != values.shift())
    frame["recommendation_transition"] = frame["recommendation_transition"].fillna(False)
    return frame


def recommendation_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, horizon, run_id), group in frame.groupby(["split", "horizon", "exp_id"]):
        counts = group["recommendation"].value_counts()
        transitions = int(group["recommendation_transition"].sum())
        rows.append({
            "Split": split,
            "Horizon": horizon,
            "Run": run_id,
            "N": len(group),
            "Maintain_Rows": int(counts.get("MAINTAIN", 0)),
            "Increase_Cooling_Rows": int(counts.get("INCREASE_COOLING", 0)),
            "High_Cooling_Attention_Rows": int(counts.get("HIGH_COOLING_ATTENTION", 0)),
            "Critical_Protective_Action_Rows": int(counts.get("CRITICAL_PROTECTIVE_ACTION", 0)),
            "Recommendation_Transitions": transitions,
            "Max_Predicted_Future_Temperature_C": group["predicted_future_temperature"].max(),
            "Mean_Liquid_Flow": group["liquid_flow"].mean(),
            "Max_T_cell_spread_C": group["T_cell_spread"].max(),
        })
    return pd.DataFrame(rows)


def association_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, horizon), group in frame.groupby(["split", "horizon"]):
        for variable in ["liquid_flow", "temp_inlet_C", "current_abs", "T_cell_spread", "predicted_future_temperature"]:
            rows.append({
                "Split": split,
                "Horizon": horizon,
                "Variable": variable,
                "Spearman_Correlation_With_Predicted_Future_Temperature": group[variable].corr(group["predicted_future_temperature"], method="spearman"),
                "N": len(group),
            })
    return pd.DataFrame(rows)
