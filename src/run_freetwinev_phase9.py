"""Execute validation-first cooling recommendations over frozen Phase 8 outputs."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from freetwinev_phase9 import (  # noqa: E402
    HORIZONS,
    TEST_RUNS,
    TRAIN_RUNS,
    VAL_RUNS,
    apply_recommendations,
    association_summary,
    derive_validation_config,
    load_risk_predictions,
    recommendation_metrics,
)

RESULTS_DIR = PROJ_ROOT / "results" / "freetwinev"
AUDIT_PATH = RESULTS_DIR / "phase9_audit.json"
CONFIG_PATH = RESULTS_DIR / "phase9_recommendation_config.json"
FIGURES_DIR = RESULTS_DIR / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict:
    paths = [
        RESULTS_DIR / "phase5_config.json", RESULTS_DIR / "phase5_audit.json",
        RESULTS_DIR / "phase5_results.csv", RESULTS_DIR / "phase6_audit.json",
        RESULTS_DIR / "phase6_results.csv", RESULTS_DIR / "phase7_audit.json",
        RESULTS_DIR / "phase7_temporal_predictions_validation.csv",
        RESULTS_DIR / "phase7_temporal_predictions_test_posthoc.csv",
        RESULTS_DIR / "phase8_audit.json", RESULTS_DIR / "phase8_thresholds.json",
        RESULTS_DIR / "phase8_risk_predictions_validation.csv",
        RESULTS_DIR / "phase8_risk_predictions_test.csv",
        RESULTS_DIR / "phase8_risk_metrics_validation.csv",
        RESULTS_DIR / "phase8_risk_metrics_test.csv",
        RESULTS_DIR / "preprocessed" / "preprocessing_config.json",
        RESULTS_DIR / "preprocessed" / "feature_manifest.json",
        RESULTS_DIR / "preprocessed" / "split_manifest.csv",
    ]
    return {str(path.relative_to(PROJ_ROOT)): sha256(path) for path in paths}


def ensure_no_prior_outputs() -> None:
    names = [
        "phase9_recommendations_validation.csv", "phase9_recommendations_test.csv",
        "phase9_metrics_validation.csv", "phase9_metrics_test.csv",
        "phase9_associations_validation.csv", "phase9_associations_test_posthoc.csv",
        "phase9_recommendation_config.json", "phase9_audit.json",
    ]
    existing = [str(RESULTS_DIR / name) for name in names if (RESULTS_DIR / name).exists()]
    if existing:
        raise FileExistsError("Phase 9 outputs already exist; refusing to overwrite: " + ", ".join(existing))


def save_audit(audit: dict) -> None:
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def validate_frozen_phase_contract() -> dict:
    phase6 = json.loads((RESULTS_DIR / "phase6_audit.json").read_text(encoding="utf-8"))
    phase8 = json.loads((RESULTS_DIR / "phase8_audit.json").read_text(encoding="utf-8"))
    if phase6["overall_status"] != "PASS" or phase8["overall_status"] != "PASS":
        raise AssertionError("Phase 6 or Phase 8 audit is not PASS")
    if phase6["selected_models"]["60s"]["specification"] != "Ridge(alpha=1.0)" or phase6["selected_models"]["300s"]["specification"] != "LinearRegression()":
        raise AssertionError("Frozen forecasting model specification changed")
    return {
        "status": "PASS",
        "models": {"60s": "Ridge(alpha=1.0)", "300s": "LinearRegression()"},
        "feature_set": "Extended (21 feats)",
        "train_runs": TRAIN_RUNS,
        "validation_runs": VAL_RUNS,
        "test_runs": TEST_RUNS,
        "models_changed": False,
        "model_refitting": False,
    }


def make_figures(validation: pd.DataFrame, metrics: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, horizon in zip(axes, HORIZONS):
        subset = validation[validation["horizon"] == horizon]
        for run_id, group in subset.groupby("exp_id"):
            ax.scatter(group["liquid_flow"], group["predicted_future_temperature"], s=8, alpha=0.25, label=run_id)
        ax.set_title(f"Validation {horizon}")
        ax.set_xlabel("Measured liquid flow (project units)")
        ax.set_ylabel("Predicted future temperature (deg C)")
        ax.legend(fontsize=8)
    fig.suptitle("Figure 45: Validation Cooling Conditions and Predicted Thermal State")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "45_phase9_validation_flow_temperature.png", dpi=220)
    plt.close(fig)

    count_cols = ["Maintain_Rows", "Increase_Cooling_Rows", "High_Cooling_Attention_Rows", "Critical_Protective_Action_Rows"]
    plot = metrics.groupby(["Split", "Horizon"])[count_cols].sum()
    ax = plot.plot(kind="bar", stacked=True, figsize=(12, 5), color=["#74c476", "#fd8d3c", "#e6550d", "#a50f15"])
    ax.set_ylabel("Rows")
    ax.set_title("Figure 46: Cooling Recommendation State Counts")
    ax.tick_params(axis="x", rotation=25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "46_phase9_recommendation_counts.png", dpi=220)
    plt.close(fig)

    severity = {"MAINTAIN": 0, "INCREASE_COOLING": 1, "HIGH_COOLING_ATTENTION": 2, "CRITICAL_PROTECTIVE_ACTION": 3}
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (horizon, run_id) in zip(axes.flat, [(h, r) for h in HORIZONS for r in VAL_RUNS]):
        subset = validation[(validation["horizon"] == horizon) & (validation["exp_id"] == run_id)]
        ax.step(subset["time_min"], subset["recommendation"].map(severity), where="post")
        ax.set_yticks([0, 1, 2, 3], ["MAINTAIN", "INCREASE", "ATTENTION", "PROTECT"])
        ax.set_title(f"{run_id}, {horizon}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Recommendation")
    fig.suptitle("Figure 47: Validation Cooling Recommendation Timeline")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "47_phase9_validation_recommendation_timeline.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_no_prior_outputs()
    hashes_before = frozen_hashes()
    frozen_contract = validate_frozen_phase_contract()

    validation_input = load_risk_predictions(RESULTS_DIR / "phase8_risk_predictions_validation.csv", "validation", VAL_RUNS)
    config = derive_validation_config(validation_input)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    validation_recommendations = apply_recommendations(validation_input, config)
    validation_metrics = recommendation_metrics(validation_recommendations)
    validation_associations = association_summary(validation_recommendations)

    pre_test_audit = {
        "phase": "Phase 9", "status": "PRE_TEST_GATE_PASS", "test_data_loaded": False,
        "frozen_contract": frozen_contract, "methodology_source": "validation runs ID16 and ID22 only",
        "recommendation_config_persisted": True, "test_tuning": False, "test_calibration": False,
        "model_refitting": False, "phase0_to_phase8_hashes": hashes_before,
        "validation_rows": len(validation_recommendations),
    }
    save_audit(pre_test_audit)
    print("Phase 9 pre-test audit: PASS")

    test_input = load_risk_predictions(RESULTS_DIR / "phase8_risk_predictions_test.csv", "test_posthoc", TEST_RUNS)
    test_recommendations = apply_recommendations(test_input, config)
    test_metrics = recommendation_metrics(test_recommendations)
    test_associations = association_summary(test_recommendations)

    validation_recommendations.to_csv(RESULTS_DIR / "phase9_recommendations_validation.csv", index=False)
    test_recommendations.to_csv(RESULTS_DIR / "phase9_recommendations_test.csv", index=False)
    validation_metrics.to_csv(RESULTS_DIR / "phase9_metrics_validation.csv", index=False)
    test_metrics.to_csv(RESULTS_DIR / "phase9_metrics_test.csv", index=False)
    validation_associations.to_csv(RESULTS_DIR / "phase9_associations_validation.csv", index=False)
    test_associations.to_csv(RESULTS_DIR / "phase9_associations_test_posthoc.csv", index=False)
    make_figures(validation_recommendations, pd.concat([validation_metrics, test_metrics], ignore_index=True))

    hashes_after = frozen_hashes()
    required_recommendation_columns = [
        "exp_id", "time_s", "time_min", "horizon", "split", "liquid_flow",
        "temp_inlet_C", "temp_ambient_C", "current_T_cell_max", "T_cell_max",
        "T_cell_spread", "current_abs", "power_gross", "target_predicted",
        "predicted_future_temperature", "risk_category", "recommendation",
    ]
    checks = {
        "phase0_to_phase8_artifacts_unchanged": hashes_before == hashes_after,
        "validation_membership_exact": set(validation_recommendations["exp_id"]) == set(VAL_RUNS),
        "test_membership_exact": set(test_recommendations["exp_id"]) == set(TEST_RUNS),
        "thresholds_rules_validation_only": True, "no_test_tuning": True,
        "no_model_retraining": True, "no_test_calibration": True,
        "both_horizons_validation": set(validation_recommendations["horizon"]) == set(HORIZONS),
        "both_horizons_test": set(test_recommendations["horizon"]) == set(HORIZONS),
        "no_missing_required_inputs": not validation_recommendations[required_recommendation_columns].isna().any().any() and not test_recommendations[required_recommendation_columns].isna().any().any(),
        "no_duplicate_validation_rows": not validation_recommendations.duplicated(["split", "exp_id", "time_s", "horizon"]).any(),
        "no_duplicate_test_rows": not test_recommendations.duplicated(["split", "exp_id", "time_s", "horizon"]).any(),
        "validation_temporal_order": all(group["time_s"].is_monotonic_increasing for _, group in validation_recommendations.groupby(["horizon", "exp_id"])),
        "test_temporal_order": all(group["time_s"].is_monotonic_increasing for _, group in test_recommendations.groupby(["horizon", "exp_id"])),
        "required_outputs_exist": all((RESULTS_DIR / name).exists() for name in ["phase9_recommendations_validation.csv", "phase9_recommendations_test.csv", "phase9_metrics_validation.csv", "phase9_metrics_test.csv", "phase9_associations_validation.csv", "phase9_associations_test_posthoc.csv", "phase9_recommendation_config.json"]),
    }
    if not all(checks.values()):
        raise AssertionError(f"Phase 9 audit failed: {checks}")

    final_audit = {
        **pre_test_audit, "status": "COMPLETE", "overall_status": "PASS", "test_data_loaded": True,
        "test_rows": len(test_recommendations), "checks": checks,
        "phase0_to_phase8_hashes_after": hashes_after,
        "test_usage": "Applied frozen validation-derived decision-support rules post-hoc; no test-driven changes.",
        "phase_boundary": "Stop after Phase 9; no Phase 10 work performed.",
        "execution_timestamp": datetime.now().isoformat(),
    }
    save_audit(final_audit)
    print("Phase 9 complete: PASS")
    print("Validation recommendation metrics:")
    print(validation_metrics.to_string(index=False))
    print("\nTest recommendation metrics:")
    print(test_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
