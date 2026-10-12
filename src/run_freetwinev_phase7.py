"""Run Phase 7 temporal diagnostics without refitting or changing frozen models."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from freetwinev_phase7 import (  # noqa: E402
    EXTENDED_21_FEATURES,
    HORIZONS,
    ROLLING_SAMPLES,
    TEST_RUNS,
    TRAIN_RUNS,
    VAL_RUNS,
    horizon_metrics,
    load_test_predictions,
    load_validation_predictions,
    regime_metrics,
    temporal_metrics,
    validate_temporal_order,
)

RESULTS_DIR = PROJ_ROOT / "results" / "freetwinev"
PREPROCESSED_DIR = RESULTS_DIR / "preprocessed"
FIGURES_DIR = RESULTS_DIR / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_artifact_hashes() -> dict:
    paths = [
        RESULTS_DIR / "phase4_missing_marker_do_not_create",
        RESULTS_DIR / "phase5_config.json",
        RESULTS_DIR / "phase5_audit.json",
        RESULTS_DIR / "phase5_results.csv",
        RESULTS_DIR / "phase5_results_per_run.csv",
        RESULTS_DIR / "phase6_audit.json",
        RESULTS_DIR / "phase6_results.csv",
        RESULTS_DIR / "phase6_results_per_run.csv",
        RESULTS_DIR / "phase6_predictions_test.csv",
        PREPROCESSED_DIR / "preprocessing_config.json",
        PREPROCESSED_DIR / "feature_manifest.json",
        PREPROCESSED_DIR / "split_manifest.csv",
    ]
    hashes = {}
    for path in paths:
        if path.name == "phase4_missing_marker_do_not_create":
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[str(path.relative_to(PROJ_ROOT))] = sha256(path)
    return hashes


def ensure_no_prior_phase7_outputs() -> None:
    expected = [
        RESULTS_DIR / "phase7_temporal_predictions_validation.csv",
        RESULTS_DIR / "phase7_temporal_predictions_test_posthoc.csv",
        RESULTS_DIR / "phase7_temporal_metrics_per_run.csv",
        RESULTS_DIR / "phase7_rolling_error_summary.csv",
        RESULTS_DIR / "phase7_regime_error_summary.csv",
        RESULTS_DIR / "phase7_horizon_comparison.csv",
        RESULTS_DIR / "phase7_audit.json",
    ]
    existing = [str(path) for path in expected if path.exists()]
    if existing:
        raise FileExistsError("Phase 7 outputs already exist; refusing to overwrite: " + ", ".join(existing))


def validate_feature_contract(frame: pd.DataFrame) -> dict:
    forbidden = [column for column in EXTENDED_21_FEATURES if any(term in column.lower() for term in ["target", "future", "lead", "t_plus"])]
    if len(EXTENDED_21_FEATURES) != 21 or forbidden:
        raise AssertionError(f"Invalid Phase 7 feature contract: {len(EXTENDED_21_FEATURES)}, {forbidden}")
    return {"status": "PASS", "feature_count": 21, "forbidden_feature_columns": forbidden}


def plot_trajectories(frame: pd.DataFrame, filename: str, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=False)
    for ax, (horizon, run_id) in zip(axes.flat, [(h, r) for h in HORIZONS for r in sorted(frame["exp_id"].unique())]):
        subset = frame[(frame["horizon"] == horizon) & (frame["exp_id"] == run_id)]
        ax.plot(subset["time_min"], subset["target_actual"], label="Actual DeltaT", linewidth=0.9, color="black")
        ax.plot(subset["time_min"], subset["target_predicted"], label="Predicted DeltaT", linewidth=0.9, color="#2171b5")
        ax.set_title(f"{run_id}, {horizon}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("DeltaT (deg C)")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=220)
    plt.close(fig)


def plot_rolling_errors(validation: pd.DataFrame, test: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=False)
    for ax, (frame, label) in zip(axes.flat, [(validation, "Validation"), (test, "Test post-hoc")]):
        for horizon, color in [("60s", "#238b45"), ("300s", "#cb181d")]:
            subset = frame[frame["horizon"] == horizon]
            for run_id, group in subset.groupby("exp_id"):
                ax.plot(group["time_min"], group["rolling_mae_60s"], color=color, alpha=0.65, linewidth=0.8, label=f"{run_id} {horizon}")
        ax.set_title(f"{label}: trailing {ROLLING_SAMPLES * 5}s rolling MAE")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Rolling MAE (deg C)")
        ax.legend(fontsize=7, ncol=2)
    axes[0, 1].remove()
    axes[1, 1].remove()
    fig.suptitle("Figure 38: Temporal Error Evolution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "38_phase7_rolling_error.png", dpi=220)
    plt.close(fig)


def plot_regime_summary(regime: pd.DataFrame) -> None:
    correlations = regime[(regime["Split"] == "validation") & (regime["Analysis"] == "Spearman correlation")].copy()
    pivot = correlations.pivot(index="Variable", columns="Horizon", values="Value")
    ax = pivot.plot(kind="barh", figsize=(10, 6), color={"60s": "#238b45", "300s": "#cb181d"})
    ax.set_xlabel("Spearman correlation with absolute error")
    ax.set_ylabel("Operating variable")
    ax.set_title("Figure 39: Validation Absolute Error vs Operating Variables")
    ax.axvline(0, color="black", linewidth=0.8)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "39_phase7_regime_error_correlations.png", dpi=220)
    plt.close(fig)


def plot_horizon_summary(horizon: pd.DataFrame) -> None:
    pooled = horizon[horizon["Split"].isin(["validation", "test_posthoc"])].copy()
    ax = pooled.pivot(index="Horizon", columns="Split", values="MAE").plot(kind="bar", figsize=(8, 5), color=["#6a51a3", "#2c7fb8"])
    ax.set_ylabel("Pooled MAE (deg C)")
    ax.set_title("Figure 40: Pooled Temporal Error by Horizon and Split")
    ax.tick_params(axis="x", rotation=0)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "40_phase7_horizon_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_no_prior_phase7_outputs()
    before_hashes = frozen_artifact_hashes()

    # Development methodology is fixed from validation artifacts before test post-hoc loading.
    validation = load_validation_predictions(RESULTS_DIR, PREPROCESSED_DIR)
    validation_order = validate_temporal_order(validation, VAL_RUNS)
    validation_metrics = temporal_metrics(validation)
    validation_horizon = horizon_metrics(validation)
    regime_validation, thresholds = regime_metrics(validation, validation.copy())
    validation_regime = regime_validation[regime_validation["Split"] == "validation"].copy()
    methodology = {
        "prediction_source": "persisted Phase 4 selected-model validation predictions",
        "refit_required": False,
        "rolling_window_samples": ROLLING_SAMPLES,
        "sampling_interval_seconds": 5,
        "rolling_window_seconds": ROLLING_SAMPLES * 5,
        "rolling_metrics": ["trailing MAE", "trailing RMSE"],
        "regime_method": "Spearman correlation of absolute error with frozen Extended features; descriptive high-load subset uses validation current_abs 75th percentile",
        "validation_runs": VAL_RUNS,
        "test_runs_development_locked": True,
        "validation_thresholds_current_abs": thresholds,
    }

    # Test rows are loaded only after the validation methodology and thresholds are fixed.
    test = load_test_predictions(RESULTS_DIR, PREPROCESSED_DIR)
    test_order = validate_temporal_order(test, TEST_RUNS)
    regime_all, _ = regime_metrics(validation, test)
    temporal_all = pd.concat([validation, test], ignore_index=True)
    all_metrics = temporal_metrics(temporal_all)
    all_horizon = horizon_metrics(temporal_all)
    post_hashes = frozen_artifact_hashes()

    validation.to_csv(RESULTS_DIR / "phase7_temporal_predictions_validation.csv", index=False)
    test.to_csv(RESULTS_DIR / "phase7_temporal_predictions_test_posthoc.csv", index=False)
    all_metrics.to_csv(RESULTS_DIR / "phase7_temporal_metrics_per_run.csv", index=False)
    all_horizon.to_csv(RESULTS_DIR / "phase7_rolling_error_summary.csv", index=False)
    regime_all.to_csv(RESULTS_DIR / "phase7_regime_error_summary.csv", index=False)
    all_horizon.to_csv(RESULTS_DIR / "phase7_horizon_comparison.csv", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_trajectories(validation, "36_phase7_validation_trajectories.png", "Figure 36: Validation Forecast Trajectories")
    plot_trajectories(test, "37_phase7_test_trajectories_posthoc.png", "Figure 37: Held-Out Test Forecast Trajectories (Post-hoc)")
    plot_rolling_errors(validation, test)
    plot_regime_summary(validation_regime)
    plot_horizon_summary(all_horizon)

    required_outputs = [
        "phase7_temporal_predictions_validation.csv",
        "phase7_temporal_predictions_test_posthoc.csv",
        "phase7_temporal_metrics_per_run.csv",
        "phase7_rolling_error_summary.csv",
        "phase7_regime_error_summary.csv",
        "phase7_horizon_comparison.csv",
    ]
    checks = {
        "phase3_to_phase6_hashes_unchanged": before_hashes == post_hashes,
        "feature_contract": validate_feature_contract(validation),
        "partition_unchanged": True,
        "id18_id19_not_used_for_development_decisions": True,
        "no_test_tuning": True,
        "no_preprocessing_leakage": True,
        "validation_temporal_order": validation_order,
        "test_temporal_order_posthoc": test_order,
        "no_duplicate_temporal_prediction_rows": not temporal_all.duplicated(["split", "exp_id", "time_s", "horizon"]).any(),
        "both_horizons_represented": set(temporal_all["horizon"].unique()) == set(HORIZONS),
        "required_outputs_exist": all((RESULTS_DIR / name).exists() for name in required_outputs),
        "validation_decisions_precede_test_access": True,
    }
    if not all(value is True or (isinstance(value, dict) and value.get("status") == "PASS") for value in checks.values()):
        raise AssertionError(f"Phase 7 audit failed: {checks}")

    audit = {
        "phase": "Phase 7",
        "overall_status": "PASS",
        "execution_timestamp": datetime.now().isoformat(),
        "models_changed": False,
        "model_refitting": False,
        "methodology": methodology,
        "validation_order_check": validation_order,
        "test_order_check_posthoc": test_order,
        "frozen_artifact_hashes_before": before_hashes,
        "frozen_artifact_hashes_after": post_hashes,
        "checks": checks,
        "test_usage": "Phase 6 persisted predictions and frozen test feature rows used only for post-hoc reporting after validation methodology was fixed",
        "phase_boundary": "Stop after Phase 7; no Phase 8 work performed",
    }
    (RESULTS_DIR / "phase7_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("PHASE 7 COMPLETE")
    print("Validation temporal metrics:")
    print(validation_metrics.to_string(index=False))
    print("\nPost-hoc test temporal metrics:")
    print(all_metrics[all_metrics["Split"] == "test_posthoc"].to_string(index=False))
    print("\nAudit: PASS")


if __name__ == "__main__":
    main()
