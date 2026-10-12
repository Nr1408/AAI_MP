"""
run_freetwinev_baselines.py
===========================
Executable script to run Phase 4: Baseline Models & Benchmark Evaluation.

Executes:
1. Scientific protocol & leakage audit (test runs untouched, no future leakage).
2. Fits baselines on Training partition ONLY (ID01, ID02, ID11, ID12, ID13, ID14, ID15, ID17, ID20, ID21).
3. Evaluates baselines on Validation partition (ID16, ID22) pooled and per-run:
   - Naive Persistence (DeltaT = 0)
   - Mean Baseline (Training set mean)
   - Linear Regression (Baseline 9 features)
   - Ridge Regression (Baseline 9 features, alpha=1.0)
   - Linear Regression (Extended 21 features)
   - Ridge Regression (Extended 21 features, alpha=1.0)
4. Computes both DeltaT metrics and reconstructed absolute future temperature metrics.
5. Saves all tables and JSON audit.
6. Generates 5 high-resolution diagnostic figures in results/freetwinev/figures/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freetwinev_baselines import run_baseline_pipeline, TRAIN_RUNS, VAL_RUNS, TEST_RUNS


def generate_diagnostic_figures(
    predictions_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    per_run_df: pd.DataFrame,
    figures_dir: Path
):
    """
    Generates 5 publication-grade diagnostic figures for baseline model evaluation.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Set common font parameters
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    # -------------------------------------------------------------------------
    # Figure 11: Actual vs Predicted Delta-T Scatter Plots
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=False, sharey=False)
    fig.suptitle(
        "Figure 11: Actual vs. Predicted Temperature Rise (Delta-T) on Validation Runs (ID16, ID22)",
        fontsize=14, fontweight="bold", y=0.98
    )

    models_to_plot = [
        ("Naive Persistence", "None", "#7f8c8d"),
        ("Mean Baseline", "None", "#e67e22"),
        ("Linear Regression", "Baseline", "#2980b9"),
        ("Linear Regression", "Extended", "#27ae60")
    ]

    for row_idx, h in enumerate(["60s", "300s"]):
        sub_df = predictions_df[predictions_df["horizon"] == h].copy()
        target_col = f"target_DeltaT_{h}"
        y_act = sub_df[target_col].values

        for col_idx, (m_name, feat_prefix, color) in enumerate(models_to_plot):
            ax = axes[row_idx, col_idx]
            pred_col = f"pred_DeltaT_{m_name}_{feat_prefix}"
            y_pred = sub_df[pred_col].values

            # Scatter points
            ax.scatter(y_act, y_pred, alpha=0.35, s=12, color=color, edgecolors="none")

            # 1:1 Reference Line
            min_val = min(np.min(y_act), np.min(y_pred)) - 0.5
            max_val = max(np.max(y_act), np.max(y_pred)) + 0.5
            ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.2, alpha=0.75, label="Ideal 1:1")

            # Horizontal zero line
            ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
            ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

            # Look up metrics
            metric_row = pooled_df[
                (pooled_df["Horizon"] == h) &
                (pooled_df["Model"] == m_name) &
                (pooled_df["Feature_Set"].str.startswith(feat_prefix))
            ]
            if not metric_row.empty:
                mae_v = metric_row["Val_MAE"].values[0]
                rmse_v = metric_row["Val_RMSE"].values[0]
                r2_v = metric_row["Val_R2"].values[0]
                title_txt = f"{m_name} ({feat_prefix})\nMAE: {mae_v:.2f}°C | RMSE: {rmse_v:.2f}°C | R²: {r2_v:.2f}"
            else:
                title_txt = f"{m_name} ({feat_prefix})"

            ax.set_title(title_txt, fontsize=10, fontweight="bold")
            ax.set_xlabel(f"Actual {target_col} (°C)", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"Horizon {h}\nPredicted ΔT (°C)", fontsize=9, fontweight="bold")
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig11_path = figures_dir / "11_baseline_actual_vs_predicted.png"
    plt.savefig(fig11_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {fig11_path}")

    # -------------------------------------------------------------------------
    # Figure 12: Residual Distributions (Error Spread & Bias)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Figure 12: Residual Distributions (e = Predicted - Actual) on Pooled Validation Set",
        fontsize=14, fontweight="bold", y=0.98
    )

    for idx, h in enumerate(["60s", "300s"]):
        ax = axes[idx]
        sub_df = predictions_df[predictions_df["horizon"] == h].copy()

        for m_name, feat_prefix, color in models_to_plot:
            res_col = f"residual_{m_name}_{feat_prefix}"
            residuals = sub_df[res_col].values

            # Plot histogram / density curve
            ax.hist(
                residuals, bins=60, density=True, alpha=0.35, color=color,
                label=f"{m_name} ({feat_prefix})"
            )

        ax.axvline(0, color="black", linestyle="--", lw=1.2, label="Zero Bias (Ideal)")
        ax.set_title(f"Horizon: {h}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Residual (Predicted - Actual ΔT in °C)", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        if h == "60s":
            ax.set_xlim(-4, 4)
        else:
            ax.set_xlim(-15, 15)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig12_path = figures_dir / "12_baseline_residual_distributions.png"
    plt.savefig(fig12_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {fig12_path}")

    # -------------------------------------------------------------------------
    # Figure 13: Residuals vs Actual Target (Heteroscedasticity Diagnosis)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Figure 13: Residuals vs. Actual Target ΔT (Diagnostic for Dynamic Range Bias)",
        fontsize=14, fontweight="bold", y=0.98
    )

    models_to_show = [
        ("Naive Persistence", "None", "#7f8c8d"),
        ("Linear Regression", "Baseline", "#2980b9"),
        ("Linear Regression", "Extended", "#27ae60")
    ]

    for row_idx, h in enumerate(["60s", "300s"]):
        sub_df = predictions_df[predictions_df["horizon"] == h].copy()
        target_col = f"target_DeltaT_{h}"
        y_act = sub_df[target_col].values

        for col_idx, (m_name, feat_prefix, color) in enumerate(models_to_show):
            ax = axes[row_idx, col_idx]
            res_col = f"residual_{m_name}_{feat_prefix}"
            residuals = sub_df[res_col].values

            ax.scatter(y_act, residuals, alpha=0.3, s=12, color=color, edgecolors="none")
            ax.axhline(0, color="red", linestyle="--", lw=1.2, label="Zero Error")

            ax.set_title(f"{m_name} ({feat_prefix}) — Horizon {h}", fontsize=10, fontweight="bold")
            ax.set_xlabel(f"Actual {target_col} (°C)", fontsize=9)
            ax.set_ylabel("Residual (Pred - Act in °C)", fontsize=9)
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig13_path = figures_dir / "13_baseline_residuals_vs_target.png"
    plt.savefig(fig13_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {fig13_path}")

    # -------------------------------------------------------------------------
    # Figure 14: Time-Series Tracking of Absolute Future Temperature
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False)
    fig.suptitle(
        "Figure 14: Time-Series Tracking of Reconstructed Absolute Temperature on Validation Runs",
        fontsize=14, fontweight="bold", y=0.98
    )

    for row_idx, rid in enumerate(["ID16", "ID22"]):
        run_desc = "ID16 (-120A Extreme Discharge, 15°C)" if rid == "ID16" else "ID22 (-80A Discharge, 8°C Chilled Coolant)"
        for col_idx, h in enumerate(["60s", "300s"]):
            ax = axes[row_idx, col_idx]
            sub_df = predictions_df[(predictions_df["exp_id"] == rid) & (predictions_df["horizon"] == h)].copy()
            t_min = sub_df["time_min"].values
            target_abs_col = f"target_T_cell_max_{h}"
            y_abs_act = sub_df[target_abs_col].values

            # Plot actual future temperature
            ax.plot(t_min, y_abs_act, "k-", lw=1.8, label="Actual Future T_cell_max", zorder=4)

            # Plot persistence
            pers_abs = sub_df["pred_T_future_Naive Persistence_None"].values
            ax.plot(t_min, pers_abs, color="#95a5a6", linestyle=":", lw=1.4, label="Persistence (T_current)")

            # Plot Linear Regression Baseline 9
            lr_base_abs = sub_df["pred_T_future_Linear Regression_Baseline"].values
            ax.plot(t_min, lr_base_abs, color="#2980b9", linestyle="--", lw=1.4, label="Linear Reg (Baseline 9)")

            # Plot Linear Regression Extended 21
            lr_ext_abs = sub_df["pred_T_future_Linear Regression_Extended"].values
            ax.plot(t_min, lr_ext_abs, color="#27ae60", linestyle="-.", lw=1.4, label="Linear Reg (Extended 21)")

            ax.set_title(f"{run_desc} — Horizon {h}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Time (minutes)", fontsize=9)
            ax.set_ylabel("Absolute Temp (°C)", fontsize=9)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig14_path = figures_dir / "14_baseline_absolute_temp_tracking.png"
    plt.savefig(fig14_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {fig14_path}")

    # -------------------------------------------------------------------------
    # Figure 15: Per-Run Error Comparison (ID16 vs ID22)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Figure 15: Run-Level Validation MAE & RMSE Comparison Across Baselines",
        fontsize=14, fontweight="bold", y=0.98
    )

    models_order = [
        "Naive Persistence",
        "Mean Baseline",
        "Linear Regression (Baseline 9)",
        "Linear Regression (Extended 21)"
    ]

    for idx, h in enumerate(["60s", "300s"]):
        ax = axes[idx]
        sub_per_run = per_run_df[per_run_df["Horizon"] == h].copy()

        # Format label names
        sub_per_run["Model_Label"] = sub_per_run["Model"] + " (" + sub_per_run["Feature_Set"].str.split().str[0] + ")"

        # Filter to selected models
        labels = [
            "Naive Persistence (None)",
            "Mean Baseline (None)",
            "Linear Regression (Baseline)",
            "Linear Regression (Extended)"
        ]

        id16_mae = []
        id16_rmse = []
        id22_mae = []
        id22_rmse = []

        for lbl in labels:
            m_id16 = sub_per_run[(sub_per_run["Model_Label"] == lbl) & (sub_per_run["Run"] == "ID16")]
            m_id22 = sub_per_run[(sub_per_run["Model_Label"] == lbl) & (sub_per_run["Run"] == "ID22")]

            id16_mae.append(m_id16["MAE"].values[0] if not m_id16.empty else 0)
            id16_rmse.append(m_id16["RMSE"].values[0] if not m_id16.empty else 0)
            id22_mae.append(m_id22["MAE"].values[0] if not m_id22.empty else 0)
            id22_rmse.append(m_id22["RMSE"].values[0] if not m_id22.empty else 0)

        x = np.arange(len(labels))
        width = 0.2

        ax.bar(x - 1.5 * width, id16_mae, width, label="ID16 MAE (-120A)", color="#e74c3c", alpha=0.85)
        ax.bar(x - 0.5 * width, id16_rmse, width, label="ID16 RMSE (-120A)", color="#c0392b", alpha=0.85)
        ax.bar(x + 0.5 * width, id22_mae, width, label="ID22 MAE (-80A Chilled)", color="#3498db", alpha=0.85)
        ax.bar(x + 1.5 * width, id22_rmse, width, label="ID22 RMSE (-80A Chilled)", color="#2980b9", alpha=0.85)

        ax.set_title(f"Horizon: {h}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Persistence", "Mean", "Linear (Base 9)", "Linear (Ext 21)"],
            rotation=15, ha="right", fontsize=9
        )
        ax.set_ylabel("Error (°C)", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig15_path = figures_dir / "15_baseline_per_run_comparison.png"
    plt.savefig(fig15_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {fig15_path}")


def main():
    print("=" * 80)
    print("PHASE 4: BASELINE MODELS & BENCHMARK EVALUATION")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent.parent
    preprocessed_dir = base_dir / "results" / "freetwinev" / "preprocessed"
    results_dir = base_dir / "results" / "freetwinev"
    figures_dir = results_dir / "figures"

    print(f"Preprocessed Data Dir: {preprocessed_dir}")
    print(f"Results Output Dir:    {results_dir}")
    print(f"Figures Output Dir:    {figures_dir}\n")

    # Step 1: Run baseline pipeline
    print("Step 1: Running baseline pipeline across 60s and 300s horizons...")
    pooled_df, per_run_df, val_preds_df, audit_summary = run_baseline_pipeline(
        preprocessed_dir, results_dir
    )

    # Step 2: Save tables and audit JSON
    print("\nStep 2: Persisting evaluation tables & machine-readable audit...")
    results_csv_path = results_dir / "baseline_results.csv"
    per_run_csv_path = results_dir / "baseline_results_per_run.csv"
    preds_csv_path = results_dir / "baseline_predictions_validation.csv"
    audit_json_path = results_dir / "baseline_audit.json"

    pooled_df.to_csv(results_csv_path, index=False)
    per_run_df.to_csv(per_run_csv_path, index=False)
    val_preds_df.to_csv(preds_csv_path, index=False)

    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"  [SAVED] {results_csv_path}")
    print(f"  [SAVED] {per_run_csv_path}")
    print(f"  [SAVED] {preds_csv_path}")
    print(f"  [SAVED] {audit_json_path}")

    # Step 3: Generate diagnostic figures
    print("\nStep 3: Generating diagnostic figures...")
    generate_diagnostic_figures(val_preds_df, pooled_df, per_run_df, figures_dir)

    # Step 4: Display terminal tables
    print("\n" + "=" * 80)
    print("TABLE 1: POOLED VALIDATION BENCHMARK SUMMARY")
    print("=" * 80)
    display_cols = [
        "Horizon", "Model", "Feature_Set", "Val_MAE", "Val_RMSE", "Val_R2", "Val_Bias", "Val_Max_AE"
    ]
    print(pooled_df[display_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 2: PER-RUN VALIDATION BREAKDOWN (ID16 vs ID22)")
    print("=" * 80)
    per_run_cols = ["Horizon", "Model", "Feature_Set", "Run", "MAE", "RMSE", "R2", "Bias", "Max_AE"]
    print(per_run_df[per_run_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 3: RECONSTRUCTED ABSOLUTE TEMPERATURE BENCHMARK (POOLED VAL)")
    print("=" * 80)
    abs_cols = ["Horizon", "Model", "Feature_Set", "Val_Abs_MAE", "Val_Abs_RMSE", "Val_Abs_R2"]
    print(pooled_df[abs_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 4: SCIENTIFIC PROTOCOL & LEAKAGE AUDIT STATUS")
    print("=" * 80)
    for h in ["60s", "300s"]:
        for ftype in ["baseline_audit", "extended_audit"]:
            audit_obj = audit_summary[h][ftype]
            print(f"  [{audit_obj['overall_status']}] Horizon {h} — {ftype}:")
            for cname, cdata in audit_obj["checks"].items():
                print(f"       - {cname}: {cdata['status']}")

    print("\n" + "=" * 80)
    print("PHASE 4 BASELINE EVALUATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
