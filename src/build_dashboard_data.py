"""
Convert Phase 6-9 result artifacts to compact JSON files for the React dashboard.
Reads from results/freetwinev/, writes to dashboard/public/data/.
Does NOT modify any frozen phase artifact.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "freetwinev"
OUT = ROOT / "dashboard" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(name: str) -> list[dict]:
    path = RESULTS / name
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def coerce_row(row: dict, float_cols: list[str], int_cols: list[str] = []) -> dict:
    out = {}
    for k, v in row.items():
        if k in float_cols:
            try:
                out[k] = float(v) if v not in ("", "nan", "None") else None
            except ValueError:
                out[k] = None
        elif k in int_cols:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = None
        else:
            out[k] = v
    return out


# ── 1. Model performance ──────────────────────────────────────────────────────
perf_rows = read_csv("phase6_results.csv")
perf = []
for r in perf_rows:
    perf.append({
        "horizon": r["Horizon"],
        "model": r["Model"],
        "specification": r["Specification"],
        "feature_set": r["Feature_Set"],
        "n": int(r["N"]),
        "mae": round(float(r["Test_MAE"]), 4),
        "rmse": round(float(r["Test_RMSE"]), 4),
        "r2": round(float(r["Test_R2"]), 4),
        "bias": round(float(r["Test_Bias"]), 4),
        "max_ae": round(float(r["Test_Max_AE"]), 4),
        "abs_r2": round(float(r["Test_Abs_R2"]), 4),
    })
(OUT / "model_performance.json").write_text(json.dumps(perf, indent=2))

per_run = read_csv("phase6_results_per_run.csv")
per_run_out = []
for r in per_run:
    per_run_out.append({
        "horizon": r["Horizon"],
        "model": r["Model"],
        "run": r["Run"],
        "n": int(r["N"]),
        "mae": round(float(r["MAE"]), 4),
        "rmse": round(float(r["RMSE"]), 4),
        "r2": round(float(r["R2"]), 4),
        "bias": round(float(r["Bias"]), 4),
        "max_ae": round(float(r["Max_AE"]), 4),
    })
(OUT / "model_performance_per_run.json").write_text(json.dumps(per_run_out, indent=2))

# ── 2. Thresholds & config ────────────────────────────────────────────────────
thresh = json.loads((RESULTS / "phase8_thresholds.json").read_text())
(OUT / "thresholds.json").write_text(json.dumps(thresh, indent=2))

rec_cfg = json.loads((RESULTS / "phase9_recommendation_config.json").read_text())
(OUT / "recommendation_config.json").write_text(json.dumps(rec_cfg, indent=2))

# ── 3. Validation metrics (from handoff — read from phase5_results/phase6) ───
val_metrics = [
    {"horizon": "60s",  "model": "Ridge α=1.0", "val_mae": 0.1598, "val_rmse": 0.7228, "val_r2": 0.7890},
    {"horizon": "300s", "model": "OLS",          "val_mae": 2.2582, "val_rmse": 4.5396, "val_r2": 0.5027},
]
(OUT / "validation_metrics.json").write_text(json.dumps(val_metrics, indent=2))

# ── 4. Risk metrics ───────────────────────────────────────────────────────────
risk_m = read_csv("phase8_risk_metrics_test.csv")
risk_metrics_out = []
for r in risk_m:
    risk_metrics_out.append({
        "horizon": r["Horizon"],
        "run": r["Run"],
        "n": int(r["N"]),
        "normal": int(r["Normal_Rows"]),
        "elevated": int(r["Elevated_Rows"]),
        "high": int(r["High_Rows"]),
        "critical": int(r["Critical_Rows"]),
        "transitions": int(r["Risk_Transitions"]),
        "any_critical": r["Any_Critical"] == "True",
        "max_predicted_temp": round(float(r["Max_Predicted_Future_Temperature_C"]), 2),
        "longest_non_normal_s": int(r["Longest_NonNormal_Seconds"]),
    })
(OUT / "risk_metrics.json").write_text(json.dumps(risk_metrics_out, indent=2))

# ── 5. Recommendation metrics ─────────────────────────────────────────────────
rec_m = read_csv("phase9_metrics_test.csv")
rec_metrics_out = []
for r in rec_m:
    rec_metrics_out.append({
        "horizon": r["Horizon"],
        "run": r["Run"],
        "n": int(r["N"]),
        "maintain": int(r["Maintain_Rows"]),
        "increase_cooling": int(r["Increase_Cooling_Rows"]),
        "high_attention": int(r["High_Cooling_Attention_Rows"]),
        "critical_action": int(r["Critical_Protective_Action_Rows"]),
        "transitions": int(r["Recommendation_Transitions"]),
        "mean_flow": round(float(r["Mean_Liquid_Flow"]), 4),
        "max_spread": round(float(r["Max_T_cell_spread_C"]), 3),
    })
(OUT / "recommendation_metrics.json").write_text(json.dumps(rec_metrics_out, indent=2))

# ── 6. Main time-series (sampled for performance) ────────────────────────────
FLOAT_COLS = [
    "time_s", "time_min", "target_actual", "target_predicted", "residual",
    "current_T_cell_max", "absolute_actual", "absolute_predicted", "absolute_residual",
    "current", "voltage", "SOC", "liquid_flow", "temp_inlet_C", "T_cell_max",
    "T_cell_spread", "current_abs", "power_gross", "abs_error",
    "T_cell_max_diff_30s", "predicted_future_temperature",
    "threshold_temperature_p75_c", "threshold_temperature_p90_c",
    "rolling_mae_60s",
]

rec_rows = read_csv("phase9_recommendations_test.csv")
ts_out = []
for r in rec_rows:
    c = coerce_row(r, FLOAT_COLS)
    ts_out.append({
        "exp_id": c["exp_id"],
        "time_s": c["time_s"],
        "time_min": c["time_min"],
        "horizon": c["horizon"],
        "T_cell_max": c["T_cell_max"],
        "predicted_future_temperature": c["predicted_future_temperature"],
        "target_actual": c["target_actual"],
        "target_predicted": c["target_predicted"],
        "liquid_flow": c["liquid_flow"],
        "current_abs": c["current_abs"],
        "temp_inlet_C": c["temp_inlet_C"],
        "T_cell_spread": c["T_cell_spread"],
        "risk_category": c["risk_category"],
        "recommendation": c["recommendation"],
        "recommendation_transition": c["recommendation_transition"],
        "abs_error": c["abs_error"],
        "rolling_mae_60s": c["rolling_mae_60s"],
        "threshold_p75": c["threshold_temperature_p75_c"],
        "threshold_p90": c["threshold_temperature_p90_c"],
    })

(OUT / "timeseries.json").write_text(json.dumps(ts_out))
print(f"timeseries.json: {len(ts_out)} rows")

# ── 7. Summary stats per run/horizon for quick KPIs ──────────────────────────
from collections import defaultdict

summary: dict[tuple, dict] = {}
for row in ts_out:
    key = (row["exp_id"], row["horizon"])
    if key not in summary:
        summary[key] = {
            "exp_id": row["exp_id"], "horizon": row["horizon"],
            "n": 0, "max_temp": -999, "max_flow": -999,
            "risk_counts": defaultdict(int), "rec_counts": defaultdict(int),
        }
    s = summary[key]
    s["n"] += 1
    if row["T_cell_max"] is not None:
        s["max_temp"] = max(s["max_temp"], row["T_cell_max"])
    if row["liquid_flow"] is not None:
        s["max_flow"] = max(s["max_flow"], row["liquid_flow"])
    s["risk_counts"][row["risk_category"]] += 1
    s["rec_counts"][row["recommendation"]] += 1

summary_out = []
for (exp, h), s in summary.items():
    summary_out.append({
        "exp_id": s["exp_id"],
        "horizon": s["horizon"],
        "n": s["n"],
        "max_temp": round(s["max_temp"], 2),
        "max_flow": round(s["max_flow"], 4),
        "risk_counts": dict(s["risk_counts"]),
        "rec_counts": dict(s["rec_counts"]),
    })
(OUT / "summary.json").write_text(json.dumps(summary_out, indent=2))

print("Done. Files written to dashboard/public/data/:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
