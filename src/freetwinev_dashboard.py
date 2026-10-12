"""Phase 10 read-only dashboard over persisted FreeTwinEV artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "freetwinev"
RUNS = ["ID18", "ID19"]
HORIZONS = ["60s", "300s"]
REQUIRED_ARTIFACTS = [
    "phase6_results.csv",
    "phase6_results_per_run.csv",
    "phase6_predictions_test.csv",
    "phase7_temporal_predictions_test_posthoc.csv",
    "phase8_risk_predictions_test.csv",
    "phase8_risk_metrics_test.csv",
    "phase9_recommendations_test.csv",
    "phase9_metrics_test.csv",
    "phase8_thresholds.json",
    "phase9_recommendation_config.json",
    "phase6_audit.json",
    "phase7_audit.json",
    "phase8_audit.json",
    "phase9_audit.json",
]
KEY_COLUMNS = ["exp_id", "time_s", "horizon"]
REQUIRED_DATA_COLUMNS = [
    "exp_id",
    "time_s",
    "time_min",
    "horizon",
    "T_cell_max",
    "T_cell_spread",
    "liquid_flow",
    "temp_inlet_C",
    "current_abs",
    "predicted_future_temperature",
    "risk_category",
    "recommendation",
]


def _read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing persisted artifact: {path}")
    return pd.read_csv(path)


def _read_json(name: str) -> dict[str, Any]:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing persisted artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_artifacts() -> dict[str, Any]:
    """Load only persisted Phase 6-9 data; this function never fits or predicts."""
    return {
        "metrics": _read_csv("phase6_results.csv"),
        "metrics_per_run": _read_csv("phase6_results_per_run.csv"),
        "predictions": _read_csv("phase6_predictions_test.csv"),
        "temporal": _read_csv("phase7_temporal_predictions_test_posthoc.csv"),
        "risk": _read_csv("phase8_risk_predictions_test.csv"),
        "risk_metrics": _read_csv("phase8_risk_metrics_test.csv"),
        "recommendations": _read_csv("phase9_recommendations_test.csv"),
        "recommendation_metrics": _read_csv("phase9_metrics_test.csv"),
        "thresholds": _read_json("phase8_thresholds.json"),
        "recommendation_config": _read_json("phase9_recommendation_config.json"),
        "audits": {
            phase: _read_json(f"{phase}_audit.json")
            for phase in ["phase6", "phase7", "phase8", "phase9"]
        },
    }


def validate_artifacts(data: dict[str, Any]) -> dict[str, Any]:
    """Validate the dashboard contract without modifying any input artifact."""
    checks: dict[str, bool] = {}
    for name in REQUIRED_ARTIFACTS:
        checks[f"artifact_exists:{name}"] = (RESULTS_DIR / name).exists()

    recommendations = data["recommendations"]
    checks["required_columns"] = set(REQUIRED_DATA_COLUMNS).issubset(recommendations.columns)
    checks["expected_runs"] = set(recommendations["exp_id"].unique()) == set(RUNS)
    checks["expected_horizons"] = set(recommendations["horizon"].unique()) == set(HORIZONS)
    checks["no_duplicate_keys"] = not recommendations.duplicated(KEY_COLUMNS).any()
    checks["no_missing_required_data"] = not recommendations[REQUIRED_DATA_COLUMNS].isna().any().any()

    metrics = data["metrics"]
    checks["frozen_metrics_present"] = set(metrics["Horizon"]) == set(HORIZONS)
    checks["critical_boundary_loaded"] = data["thresholds"].get("engineering_cutoff_c") == 70.0
    checks["percentile_bands_loaded"] = set(data["thresholds"].get("thresholds_by_horizon", {})) == set(HORIZONS)
    checks["flow_reference_loaded"] = set(data["recommendation_config"].get("low_flow_reference_by_horizon", {})) == set(HORIZONS)
    checks["all_prior_audits_pass"] = all(
        audit.get("overall_status", audit.get("status")) == "PASS"
        for audit in data["audits"].values()
    )
    checks["timestamps_preserved"] = recommendations["time_s"].equals(data["risk"]["time_s"])

    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "row_count": int(len(recommendations)),
        "runs": sorted(recommendations["exp_id"].unique().tolist()),
        "horizons": sorted(recommendations["horizon"].unique().tolist()),
        "time_min": float(recommendations["time_s"].min()),
        "time_max": float(recommendations["time_s"].max()),
    }


def _filtered_frame(data: dict[str, Any], runs: list[str], horizons: list[str]) -> pd.DataFrame:
    frame = data["recommendations"]
    return frame[frame["exp_id"].isin(runs) & frame["horizon"].isin(horizons)].copy()


def _latest_row(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values("time_s", kind="mergesort").iloc[-1]


def _format_value(value: Any, unit: str = "") -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}{unit}"


def render_dashboard(data: dict[str, Any]) -> None:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="FreeTwinEV Thermal Dashboard", layout="wide")
    st.title("FreeTwinEV Thermal Forecast Dashboard")
    st.caption("Phase 10 visualization and decision support over held-out Phase 6-9 artifacts")

    with st.sidebar:
        st.header("Filters")
        selected_runs = st.multiselect("Run", RUNS, default=RUNS)
        horizon_choice = st.radio("Forecast horizon", ["Both", *HORIZONS], index=0)
        selected_horizons = HORIZONS if horizon_choice == "Both" else [horizon_choice]
        full_time = data["recommendations"]["time_s"]
        time_range = st.slider("Time range (s)", float(full_time.min()), float(full_time.max()), (float(full_time.min()), float(full_time.max())))
        risk_options = ["Normal", "Elevated", "High", "Critical"]
        selected_risks = st.multiselect("Risk categories", risk_options, default=risk_options)
        recommendation_options = sorted(data["recommendations"]["recommendation"].dropna().unique())
        selected_recommendations = st.multiselect("Recommendations", recommendation_options, default=recommendation_options)

    if not selected_runs or not selected_horizons:
        st.warning("Select at least one run and one horizon.")
        return
    frame = _filtered_frame(data, selected_runs, selected_horizons)
    frame = frame[frame["time_s"].between(*time_range)]
    frame = frame[frame["risk_category"].isin(selected_risks) & frame["recommendation"].isin(selected_recommendations)]
    if frame.empty:
        st.warning("No persisted rows match the selected filters.")
        return

    st.subheader("Current thermal state")
    latest = _latest_row(frame)
    kpis = st.columns(8)
    values = [
        ("Max cell temperature", _format_value(latest["T_cell_max"], " °C")),
        ("Predicted at 60s", _format_value(_latest_horizon_value(frame, "60s"), " °C")),
        ("Predicted at 300s", _format_value(_latest_horizon_value(frame, "300s"), " °C")),
        ("Risk", str(latest["risk_category"])),
        ("Recommendation", str(latest["recommendation"])),
        ("Liquid flow", _format_value(latest["liquid_flow"], " L/min")),
        ("Current magnitude", _format_value(latest["current_abs"], " A")),
        ("Temperature spread", _format_value(latest["T_cell_spread"], " °C")),
    ]
    for column, (label, value) in zip(kpis, values):
        column.metric(label, value)

    st.subheader("Forecast trajectories")
    trajectory = frame.sort_values(["exp_id", "horizon", "time_s"], kind="mergesort")
    figure = go.Figure()
    for (run, horizon), group in trajectory.groupby(["exp_id", "horizon"], sort=True):
        figure.add_trace(go.Scatter(x=group["time_s"], y=group["T_cell_max"], name=f"{run} observed", legendgroup=run, mode="lines"))
        figure.add_trace(go.Scatter(x=group["time_s"], y=group["predicted_future_temperature"], name=f"{run} predicted {horizon}", legendgroup=f"{run}-{horizon}", mode="lines"))
    figure.add_hline(y=data["thresholds"]["engineering_cutoff_c"], line_dash="dash", line_color="#d1495b", annotation_text="70 °C documented cutoff")
    figure.update_layout(xaxis_title="Exact persisted time (s)", yaxis_title="Temperature (°C)", hovermode="x unified")
    st.plotly_chart(figure, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Risk distribution")
        counts = frame.groupby(["horizon", "risk_category"], observed=True).size().reset_index(name="rows")
        st.plotly_chart(px.bar(counts, x="horizon", y="rows", color="risk_category", category_orders={"risk_category": ["Normal", "Elevated", "High", "Critical"]}), use_container_width=True)
    with right:
        st.subheader("Risk timeline")
        st.plotly_chart(px.scatter(frame, x="time_s", y="predicted_future_temperature", color="risk_category", symbol="horizon", hover_data=["exp_id", "recommendation"]), use_container_width=True)

    st.subheader("Critical-event summary")
    risk_metrics = data["risk_metrics"]
    st.dataframe(risk_metrics[["Horizon", "Run", "Critical_Rows", "Risk_Transitions", "Any_Critical"]], hide_index=True, use_container_width=True)

    st.subheader("Descriptive validation bands")
    bands = []
    for horizon in HORIZONS:
        values = data["thresholds"]["thresholds_by_horizon"][horizon]
        bands.append({"Horizon": horizon, "P75 (°C)": values["future_temperature_p75_c"], "P90 (°C)": values["future_temperature_p90_c"], "P99 (°C)": values["future_temperature_p99_c"]})
    st.dataframe(pd.DataFrame(bands), hide_index=True, use_container_width=True)
    st.caption("P75/P90/P99 are validation-derived descriptive forecast bands, not certified safety limits.")

    st.subheader("Cooling conditions and transitions")
    flow_reference = data["recommendation_config"]["low_flow_reference_by_horizon"]
    st.caption(f"Validation P75 low-flow references loaded from Phase 9 config: 60s {flow_reference['60s']:.4f} L/min; 300s {flow_reference['300s']:.4f} L/min.")
    cooling_columns = ["exp_id", "time_s", "horizon", "liquid_flow", "temp_inlet_C", "current_abs", "risk_category", "recommendation", "recommendation_transition"]
    st.dataframe(frame.sort_values("time_s")[cooling_columns], hide_index=True, use_container_width=True)

    st.subheader("Frozen held-out model performance")
    metrics = data["metrics"].rename(columns={"Horizon": "Horizon", "Test_MAE": "MAE", "Test_RMSE": "RMSE", "Test_R2": "R²"})
    st.dataframe(metrics[["Horizon", "Model", "Feature_Set", "MAE", "RMSE", "R²", "N"]], hide_index=True, use_container_width=True)
    st.dataframe(data["metrics_per_run"], hide_index=True, use_container_width=True)

    st.subheader("Scope and limitations")
    st.warning("Forecasts are model predictions, not guarantees. The 70 °C boundary is the documented project cutoff. Cooling recommendations are decision-support guidance; Phase 9 does not establish causal cooling effectiveness. Test observations are held-out evaluation data, and filters do not change the frozen methodology.")


def _latest_horizon_value(frame: pd.DataFrame, horizon: str) -> Any:
    horizon_frame = frame[frame["horizon"] == horizon].sort_values("time_s", kind="mergesort")
    return horizon_frame["predicted_future_temperature"].iloc[-1] if not horizon_frame.empty else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="Validate persisted dashboard inputs and print JSON")
    args = parser.parse_args()
    data = load_artifacts()
    validation = validate_artifacts(data)
    if args.validate:
        print(json.dumps(validation, indent=2))
        raise SystemExit(0 if validation["status"] == "PASS" else 1)
    render_dashboard(data)


if __name__ == "__main__":
    main()