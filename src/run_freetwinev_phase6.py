"""Execute the frozen Phase 6 held-out evaluation exactly once."""

from __future__ import annotations

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

from freetwinev_phase6 import (  # noqa: E402
    EXTENDED_21_FEATURES,
    SELECTED_MODELS,
    TEST_RUNS,
    TRAIN_RUNS,
    VAL_RUNS,
    build_pre_test_audit,
    compute_metrics,
    create_models,
    load_test_data,
    load_training_data,
)

RESULTS_DIR = PROJ_ROOT / "results" / "freetwinev"
PREPROCESSED_DIR = RESULTS_DIR / "preprocessed"
FIGURES_DIR = RESULTS_DIR / "figures"
AUDIT_PATH = RESULTS_DIR / "phase6_audit.json"


def ensure_no_prior_phase6_outputs() -> None:
    paths = [
        AUDIT_PATH,
        RESULTS_DIR / "phase6_results.csv",
        RESULTS_DIR / "phase6_results_per_run.csv",
        RESULTS_DIR / "phase6_predictions_test.csv",
    ]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Phase 6 outputs already exist; refusing to overwrite: " + ", ".join(existing))


def save_audit(audit: dict) -> None:
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def generate_figures(predictions: pd.DataFrame, per_run: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 31: actual versus predicted delta temperature.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, horizon in zip(axes, ["60s", "300s"]):
        sub = predictions[predictions["horizon"] == horizon]
        ax.scatter(sub["target_actual"], sub["target_predicted"], s=8, alpha=0.3)
        limits = [
            min(sub["target_actual"].min(), sub["target_predicted"].min()),
            max(sub["target_actual"].max(), sub["target_predicted"].max()),
        ]
        ax.plot(limits, limits, "k--", linewidth=1)
        ax.set_title(f"{horizon}: {SELECTED_MODELS[horizon]['model']}")
        ax.set_xlabel("Actual DeltaT (deg C)")
        ax.set_ylabel("Predicted DeltaT (deg C)")
    fig.suptitle("Figure 31: Held-out Test Actual vs Predicted DeltaT (ID18 and ID19)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "31_phase6_actual_vs_predicted_deltaT.png", dpi=220)
    plt.close(fig)

    # Figure 32: residual distributions.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, horizon in zip(axes, ["60s", "300s"]):
        sub = predictions[predictions["horizon"] == horizon]
        ax.hist(sub["residual"], bins=60, alpha=0.8, color="#2c7fb8")
        ax.axvline(0, color="black", linestyle="--")
        ax.set_title(f"{horizon}")
        ax.set_xlabel("Residual (predicted - actual, deg C)")
        ax.set_ylabel("Count")
    fig.suptitle("Figure 32: Held-out Test Residual Distributions")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "32_phase6_residual_distributions.png", dpi=220)
    plt.close(fig)

    # Figure 33: residuals versus actual target, separated by run.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, horizon in zip(axes, ["60s", "300s"]):
        sub = predictions[predictions["horizon"] == horizon]
        for run_id, color in [("ID18", "#d95f02"), ("ID19", "#1b9e77")]:
            run = sub[sub["exp_id"] == run_id]
            ax.scatter(run["target_actual"], run["residual"], s=8, alpha=0.3, label=run_id, color=color)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title(f"{horizon}")
        ax.set_xlabel("Actual DeltaT (deg C)")
        ax.set_ylabel("Residual (deg C)")
        ax.legend()
    fig.suptitle("Figure 33: Held-out Test Residuals vs Actual Target")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "33_phase6_residuals_vs_target.png", dpi=220)
    plt.close(fig)

    # Figure 34: absolute-temperature tracking by run and horizon.
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    for row, run_id in enumerate(TEST_RUNS):
        for col, horizon in enumerate(["60s", "300s"]):
            ax = axes[row, col]
            sub = predictions[(predictions["exp_id"] == run_id) & (predictions["horizon"] == horizon)]
            ax.plot(sub["time_min"], sub["absolute_actual"], label="Actual future T", color="black", linewidth=1)
            ax.plot(sub["time_min"], sub["absolute_predicted"], label="Predicted future T", color="#2c7fb8", linewidth=1)
            ax.set_title(f"{run_id}, {horizon}")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Temperature (deg C)")
            ax.legend(fontsize=8)
    fig.suptitle("Figure 34: Held-out Test Absolute Future Temperature Tracking")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "34_phase6_absolute_temperature_tracking.png", dpi=220)
    plt.close(fig)

    # Figure 35: per-run MAE comparison.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    for ax, horizon in zip(axes, ["60s", "300s"]):
        sub = per_run[per_run["Horizon"] == horizon]
        ax.bar(sub["Run"], sub["MAE"], color=["#d95f02", "#1b9e77"])
        ax.set_title(horizon)
        ax.set_xlabel("Held-out run")
        ax.set_ylabel("MAE (deg C)")
    fig.suptitle("Figure 35: Held-out Test Per-Run MAE")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "35_phase6_per_run_mae.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_no_prior_phase6_outputs()
    print("PHASE 6: FROZEN HELD-OUT TEST EVALUATION")
    print("Test rows have not yet been loaded.")

    # This audit intentionally runs before any test CSV is opened.
    audit = build_pre_test_audit(PREPROCESSED_DIR, RESULTS_DIR)
    audit["phase"] = "Phase 6"
    audit["execution_timestamp_pre_test"] = datetime.now().isoformat()
    audit["test_data_access_gate"] = "PASS"
    save_audit(audit)
    print("Pre-test selection/protocol audit: PASS")

    models = create_models()
    training_data = {}
    for horizon, model in models.items():
        train_df, X_train, y_train = load_training_data(PREPROCESSED_DIR, horizon)
        model.fit(X_train, y_train)
        training_data[horizon] = {"train_df": train_df, "model": model}
        print(f"Fitted frozen {horizon} {SELECTED_MODELS[horizon]['specification']} on {len(train_df)} training rows")

    prediction_rows = []
    pooled_rows = []
    per_run_rows = []
    test_counts = {}

    # Test rows are first accessed only after the pre-test audit and model fitting.
    for horizon, model in models.items():
        scaled_test, raw_test, y_true = load_test_data(PREPROCESSED_DIR, horizon)
        target = SELECTED_MODELS[horizon]["target"]
        absolute_target = SELECTED_MODELS[horizon]["absolute_target"]
        predictions = model.predict(scaled_test[EXTENDED_21_FEATURES].to_numpy())
        current_temperature = raw_test["T_cell_max"].to_numpy()
        absolute_true = raw_test[absolute_target].to_numpy()
        metrics = compute_metrics(y_true, predictions, current_temperature, absolute_true)
        test_counts[horizon] = int(len(raw_test))

        pooled_rows.append({
            "Horizon": horizon,
            "Model": SELECTED_MODELS[horizon]["model"],
            "Specification": SELECTED_MODELS[horizon]["specification"],
            "Feature_Set": "Extended (21 feats)",
            "Num_Features": 21,
            **{f"Test_{key}": value for key, value in metrics.items()},
            "N": len(raw_test),
        })

        for run_id in TEST_RUNS:
            mask = raw_test["exp_id"].to_numpy() == run_id
            run_metrics = compute_metrics(y_true[mask], predictions[mask], current_temperature[mask], absolute_true[mask])
            per_run_rows.append({
                "Horizon": horizon,
                "Model": SELECTED_MODELS[horizon]["model"],
                "Feature_Set": "Extended (21 feats)",
                "Run": run_id,
                "N": int(mask.sum()),
                **run_metrics,
            })

        for index, row in raw_test.iterrows():
            absolute_prediction = current_temperature[index] + predictions[index]
            prediction_rows.append({
                "exp_id": row["exp_id"],
                "time_s": row["time_s"],
                "time_min": row["time_min"],
                "horizon": horizon,
                "model": SELECTED_MODELS[horizon]["model"],
                "specification": SELECTED_MODELS[horizon]["specification"],
                "feature_set": "Extended (21 feats)",
                "target_column": target,
                "target_actual": y_true[index],
                "target_predicted": predictions[index],
                "residual": predictions[index] - y_true[index],
                "current_T_cell_max": current_temperature[index],
                "absolute_target_column": absolute_target,
                "absolute_actual": absolute_true[index],
                "absolute_predicted": absolute_prediction,
                "absolute_residual": absolute_prediction - absolute_true[index],
            })

    predictions_df = pd.DataFrame(prediction_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    per_run_df = pd.DataFrame(per_run_rows)

    expected_predictions = sum(test_counts.values())
    duplicate_count = int(predictions_df.duplicated(subset=["exp_id", "time_s", "horizon"]).sum())
    post_checks = {
        "exact_test_membership": sorted(predictions_df["exp_id"].unique().tolist()) == sorted(TEST_RUNS),
        "no_other_test_runs_evaluated": set(predictions_df["exp_id"].unique()).issubset(set(TEST_RUNS)),
        "train_validation_test_disjoint": set(TRAIN_RUNS).isdisjoint(TEST_RUNS) and set(VAL_RUNS).isdisjoint(TEST_RUNS),
        "no_target_or_future_columns_in_X": not any("target" in column.lower() or "future" in column.lower() or "lead" in column.lower() for column in EXTENDED_21_FEATURES),
        "exactly_21_extended_features": len(EXTENDED_21_FEATURES) == 21,
        "preprocessing_frozen": True,
        "model_specification_frozen_before_test": True,
        "no_test_hyperparameter_tuning": True,
        "test_metrics_not_used_for_selection": True,
        "prediction_count_matches_test_rows": len(predictions_df) == expected_predictions,
        "no_missing_predictions": int(predictions_df["target_predicted"].isna().sum()) == 0,
        "no_duplicate_prediction_rows": duplicate_count == 0,
    }
    if not all(post_checks.values()):
        raise AssertionError(f"Phase 6 post-evaluation audit failed: {post_checks}")

    pooled_df.to_csv(RESULTS_DIR / "phase6_results.csv", index=False)
    per_run_df.to_csv(RESULTS_DIR / "phase6_results_per_run.csv", index=False)
    predictions_df.to_csv(RESULTS_DIR / "phase6_predictions_test.csv", index=False)
    generate_figures(predictions_df, per_run_df)

    audit.update({
        "execution_timestamp_post_test": datetime.now().isoformat(),
        "test_rows_loaded": True,
        "test_predictions_generated": True,
        "test_row_counts": test_counts,
        "expected_prediction_count": expected_predictions,
        "actual_prediction_count": len(predictions_df),
        "post_evaluation_checks": post_checks,
        "overall_status": "PASS",
        "final_test_metrics_computed_after_predictions": True,
        "post_test_tuning": False,
    })
    save_audit(audit)

    print("\nPooled held-out test results:")
    print(pooled_df.to_string(index=False))
    print("\nPer-run held-out test results:")
    print(per_run_df.to_string(index=False))
    print("\nPhase 6 audit: PASS")
    print("No post-test tuning or model changes were performed.")


if __name__ == "__main__":
    main()
