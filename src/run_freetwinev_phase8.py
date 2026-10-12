"""Execute Phase 8 thermal-risk derivation over frozen Phase 7 predictions."""

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

from freetwinev_phase8 import (  # noqa: E402
    ENGINEERING_CUTOFF_C,
    HORIZONS,
    TEST_RUNS,
    TRAIN_RUNS,
    VAL_RUNS,
    apply_risk_logic,
    derive_validation_thresholds,
    load_phase7_predictions,
    risk_metrics,
)

RESULTS_DIR = PROJ_ROOT / "results" / "freetwinev"
PREPROCESSED_DIR = RESULTS_DIR / "preprocessed"
FIGURES_DIR = RESULTS_DIR / "figures"
AUDIT_PATH = RESULTS_DIR / "phase8_audit.json"
THRESHOLD_PATH = RESULTS_DIR / "phase8_thresholds.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict:
    paths = [
        PREPROCESSED_DIR / "preprocessing_config.json",
        PREPROCESSED_DIR / "feature_manifest.json",
        PREPROCESSED_DIR / "split_manifest.csv",
        RESULTS_DIR / "phase5_config.json",
        RESULTS_DIR / "phase5_audit.json",
        RESULTS_DIR / "phase5_results.csv",
        RESULTS_DIR / "phase6_audit.json",
        RESULTS_DIR / "phase6_results.csv",
        RESULTS_DIR / "phase6_predictions_test.csv",
        RESULTS_DIR / "phase7_audit.json",
        RESULTS_DIR / "phase7_temporal_predictions_validation.csv",
        RESULTS_DIR / "phase7_temporal_predictions_test_posthoc.csv",
    ]
    return {str(path.relative_to(PROJ_ROOT)): sha256(path) for path in paths}


def ensure_no_prior_phase8_outputs() -> None:
    names = [
        "phase8_risk_predictions_validation.csv",
        "phase8_risk_predictions_test.csv",
        "phase8_risk_metrics_validation.csv",
        "phase8_risk_metrics_test.csv",
        "phase8_thresholds.json",
        "phase8_audit.json",
    ]
    existing = [str(RESULTS_DIR / name) for name in names if (RESULTS_DIR / name).exists()]
    if existing:
        raise FileExistsError("Phase 8 outputs already exist; refusing to overwrite: " + ", ".join(existing))


def save_audit(audit: dict) -> None:
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def validate_frozen_contract() -> dict:
    preprocessing = json.loads((PREPROCESSED_DIR / "preprocessing_config.json").read_text(encoding="utf-8"))
    phase6 = json.loads((RESULTS_DIR / "phase6_audit.json").read_text(encoding="utf-8"))
    phase7 = json.loads((RESULTS_DIR / "phase7_audit.json").read_text(encoding="utf-8"))
    split = pd.read_csv(PREPROCESSED_DIR / "split_manifest.csv")
    split_map = dict(zip(split["Exp_ID"], split["Split"]))
    expected = {rid: "train" for rid in TRAIN_RUNS}
    expected.update({rid: "val" for rid in VAL_RUNS})
    expected.update({rid: "test" for rid in TEST_RUNS})
    if split_map != expected:
        raise AssertionError("Frozen run partition changed")
    if preprocessing["scaler_selected"] != "StandardScaler":
        raise AssertionError("Frozen preprocessing scaler changed")
    if phase6["overall_status"] != "PASS" or phase7["overall_status"] != "PASS":
        raise AssertionError("Phase 6 or Phase 7 audit is not PASS")
    if phase6["selected_models"]["60s"]["specification"] != "Ridge(alpha=1.0)" or phase6["selected_models"]["300s"]["specification"] != "LinearRegression()":
        raise AssertionError("Frozen model specification changed")
    if phase7["checks"]["feature_contract"]["feature_count"] != 21:
        raise AssertionError("Frozen feature count changed")
    return {
        "status": "PASS",
        "models": {"60s": "Ridge(alpha=1.0)", "300s": "LinearRegression()"},
        "feature_set": "Extended (21 feats)",
        "feature_count": 21,
        "scaler": "StandardScaler",
        "train_runs": TRAIN_RUNS,
        "validation_runs": VAL_RUNS,
        "test_runs": TEST_RUNS,
    }


def plot_temperature_risk(validation: pd.DataFrame, test: pd.DataFrame, thresholds: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=False)
    for ax, frame, label in [(axes[0, 0], validation, "Validation"), (axes[0, 1], test, "Test post-hoc")]:
        for horizon, color in [("60s", "#238b45"), ("300s", "#cb181d")]:
            subset = frame[frame["horizon"] == horizon]
            for run_id, group in subset.groupby("exp_id"):
                ax.plot(group["time_min"], group["predicted_future_temperature"], alpha=0.65, linewidth=0.8, label=f"{run_id} {horizon}", color=color)
            ax.axhline(ENGINEERING_CUTOFF_C, color="black", linestyle="--", linewidth=1, label="Documented 70 C cutoff")
        ax.set_title(f"{label}: predicted future temperature")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Predicted future T (deg C)")
        ax.legend(fontsize=7, ncol=2)
    for ax, frame, label in [(axes[1, 0], validation, "Validation"), (axes[1, 1], test, "Test post-hoc")]:
        counts = frame.groupby(["horizon", "risk_category"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=["Normal", "Elevated", "High", "Critical"], fill_value=0)
        counts.plot(kind="bar", stacked=True, ax=ax, color=["#74c476", "#fd8d3c", "#e6550d", "#a50f15"])
        ax.set_title(f"{label}: risk-category counts")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Rows")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Figure 41: Thermal-Risk Temperature and Category Overview")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "41_phase8_temperature_risk_overview.png", dpi=220)
    plt.close(fig)


def plot_risk_timeline(frame: pd.DataFrame, filename: str, title: str) -> None:
    severity = {"Normal": 0, "Elevated": 1, "High": 2, "Critical": 3}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=False)
    for ax, (horizon, run_id) in zip(axes.flat, [(h, r) for h in HORIZONS for r in sorted(frame["exp_id"].unique())]):
        subset = frame[(frame["horizon"] == horizon) & (frame["exp_id"] == run_id)]
        ax.step(subset["time_min"], subset["risk_category"].map(severity), where="post", linewidth=0.9)
        ax.set_yticks([0, 1, 2, 3], ["Normal", "Elevated", "High", "Critical"])
        ax.set_title(f"{run_id}, {horizon}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Risk category")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=220)
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame) -> None:
    plot = metrics.copy()
    plot["Label"] = plot["Run"] + " " + plot["Horizon"]
    ax = plot.plot(x="Label", y="Longest_NonNormal_Seconds", kind="bar", figsize=(11, 5), color="#6a51a3", legend=False)
    ax.set_ylabel("Longest non-normal duration (s)")
    ax.set_title("Figure 43: Longest Continuous Non-Normal Risk Period")
    ax.tick_params(axis="x", rotation=35)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "43_phase8_risk_duration.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_no_prior_phase8_outputs()
    hashes_before = frozen_hashes()
    frozen_contract = validate_frozen_contract()

    # Development threshold derivation uses validation only and is persisted before test access.
    validation_input = load_phase7_predictions(RESULTS_DIR / "phase7_temporal_predictions_validation.csv", "validation", VAL_RUNS)
    thresholds = derive_validation_thresholds(validation_input)
    THRESHOLD_PATH.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    validation_risk = apply_risk_logic(validation_input, thresholds)
    pre_test_audit = {
        "phase": "Phase 8",
        "status": "PRE_TEST_GATE_PASS",
        "test_data_loaded": False,
        "threshold_source": "validation runs ID16 and ID22 only",
        "thresholds_persisted": True,
        "frozen_contract": frozen_contract,
        "phase3_to_phase7_hashes": hashes_before,
        "test_threshold_information_used": False,
        "test_feature_selection": False,
        "test_calibration": False,
        "risk_logic_frozen_before_test": True,
        "validation_rows": len(validation_risk),
    }
    save_audit(pre_test_audit)
    print("Phase 8 pre-test audit: PASS")

    # Test observations are loaded only after the validation thresholds and logic are frozen.
    test_input = load_phase7_predictions(RESULTS_DIR / "phase7_temporal_predictions_test_posthoc.csv", "test_posthoc", TEST_RUNS)
    test_risk = apply_risk_logic(test_input, thresholds)
    validation_metrics = risk_metrics(validation_risk)
    test_metrics = risk_metrics(test_risk)

    validation_risk.to_csv(RESULTS_DIR / "phase8_risk_predictions_validation.csv", index=False)
    test_risk.to_csv(RESULTS_DIR / "phase8_risk_predictions_test.csv", index=False)
    validation_metrics.to_csv(RESULTS_DIR / "phase8_risk_metrics_validation.csv", index=False)
    test_metrics.to_csv(RESULTS_DIR / "phase8_risk_metrics_test.csv", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_temperature_risk(validation_risk, test_risk, thresholds)
    plot_risk_timeline(validation_risk, "42_phase8_validation_risk_timeline.png", "Figure 42: Validation Thermal-Risk Timeline")
    plot_risk_timeline(test_risk, "44_phase8_test_risk_timeline_posthoc.png", "Figure 44: Held-Out Test Thermal-Risk Timeline (Post-hoc)")
    plot_metrics(pd.concat([validation_metrics, test_metrics], ignore_index=True))

    hashes_after = frozen_hashes()
    required_risk_columns = [
        "exp_id", "time_s", "time_min", "horizon", "split",
        "current_T_cell_max", "target_predicted", "predicted_future_temperature",
        "T_cell_spread", "T_cell_max_diff_30s", "current_abs", "power_gross",
        "risk_category", "secondary_signal_count",
    ]
    checks = {
        "phase3_to_phase7_artifacts_unchanged": hashes_before == hashes_after,
        "feature_manifest_and_model_contract_frozen": frozen_contract["status"] == "PASS",
        "thresholds_validation_only": True,
        "no_test_threshold_tuning": True,
        "no_test_feature_selection": True,
        "no_test_calibration": True,
        "no_model_refitting": True,
        "validation_membership_exact": set(validation_risk["exp_id"]) == set(VAL_RUNS),
        "test_membership_exact": set(test_risk["exp_id"]) == set(TEST_RUNS),
        "both_horizons_validation": set(validation_risk["horizon"]) == set(HORIZONS),
        "both_horizons_test": set(test_risk["horizon"]) == set(HORIZONS),
        "no_missing_risk_inputs": not validation_risk[required_risk_columns].isna().any().any() and not test_risk[required_risk_columns].isna().any().any(),
        "no_duplicate_validation_rows": not validation_risk.duplicated(["split", "exp_id", "time_s", "horizon"]).any(),
        "no_duplicate_test_rows": not test_risk.duplicated(["split", "exp_id", "time_s", "horizon"]).any(),
        "validation_temporal_order": all(group["time_s"].is_monotonic_increasing for _, group in validation_risk.groupby(["horizon", "exp_id"])),
        "test_temporal_order": all(group["time_s"].is_monotonic_increasing for _, group in test_risk.groupby(["horizon", "exp_id"])),
        "temperature_reconstruction_consistent": bool((abs(validation_risk["predicted_future_temperature"] - validation_risk["absolute_predicted"]) < 1e-8).all() and (abs(test_risk["predicted_future_temperature"] - test_risk["absolute_predicted"]) < 1e-8).all()),
        "required_outputs_exist": all((RESULTS_DIR / name).exists() for name in ["phase8_risk_predictions_validation.csv", "phase8_risk_predictions_test.csv", "phase8_risk_metrics_validation.csv", "phase8_risk_metrics_test.csv", "phase8_thresholds.json"]),
    }
    if not all(checks.values()):
        raise AssertionError(f"Phase 8 audit failed: {checks}")

    final_audit = {
        **pre_test_audit,
        "status": "COMPLETE",
        "overall_status": "PASS",
        "test_data_loaded": True,
        "test_rows": len(test_risk),
        "validation_rows": len(validation_risk),
        "checks": checks,
        "phase3_to_phase7_hashes_after": hashes_after,
        "test_usage": "Applied frozen validation-derived risk logic only after pre-test gate; no test-driven changes.",
        "phase_boundary": "Stop after Phase 8; no Phase 9 work performed.",
        "execution_timestamp": datetime.now().isoformat(),
    }
    save_audit(final_audit)
    print("Phase 8 complete: PASS")
    print("Validation risk metrics:")
    print(validation_metrics.to_string(index=False))
    print("\nTest risk metrics:")
    print(test_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
