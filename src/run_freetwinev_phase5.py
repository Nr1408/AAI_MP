"""
run_freetwinev_phase5.py
========================
Standalone Phase-5 execution pipeline.

Runs ML model exploration & Group-CV hyperparameter selection, saves all
required artifacts, and generates publication-quality diagnostic figures.

Nothing in this script touches ID18 or ID19.
Execution halts with a hard AssertionError if test-set leakage is detected.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── project path setup ─────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from freetwinev_phase5 import (
    EXTENDED_21_FEATURES,
    BASELINE_9_FEATURES,
    N_CV_FOLDS,
    RANDOM_SEED,
    TRAIN_RUNS,
    VAL_RUNS,
    TEST_RUNS,
    MODEL_REGISTRY,
    assert_no_test_runs,
    build_feature_importance_df,
    build_per_run_df,
    build_summary_df,
    compute_metrics,
    run_full_phase5_pipeline,
    load_datasets,
)

# ── output directories ────────────────────────────────────────────────────────
RESULTS_DIR    = PROJ_ROOT / "results" / "freetwinev"
PREPROC_DIR    = RESULTS_DIR / "preprocessed"
FIGURES_DIR    = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 Reference Benchmarks (from completed Phase 4 evaluation)
# ─────────────────────────────────────────────────────────────────────────────
PHASE4_REF = {
    "60s": {
        "Extended (21 feats)": {
            "Ridge Regression (alpha=1.0)": {"Val_MAE": 0.1598, "Val_RMSE": 0.7228, "Val_R2": 0.7890},
            "Linear Regression":            {"Val_MAE": 0.1601, "Val_RMSE": 0.7232, "Val_R2": 0.7888},
        },
        "Baseline (9 feats)": {
            "Ridge Regression (alpha=1.0)": {"Val_MAE": 0.5811, "Val_RMSE": 1.0129, "Val_R2": 0.5857},
            "Linear Regression":            {"Val_MAE": 0.5806, "Val_RMSE": 1.0128, "Val_R2": 0.5858},
        },
    },
    "300s": {
        "Extended (21 feats)": {
            "Ridge Regression (alpha=1.0)": {"Val_MAE": 2.2629, "Val_RMSE": 4.5423, "Val_R2": 0.5021},
            "Linear Regression":            {"Val_MAE": 2.2582, "Val_RMSE": 4.5396, "Val_R2": 0.5027},
        },
        "Baseline (9 feats)": {
            "Ridge Regression (alpha=1.0)": {"Val_MAE": 2.8758, "Val_RMSE": 5.0108, "Val_R2": 0.3941},
            "Linear Regression":            {"Val_MAE": 2.8754, "Val_RMSE": 5.0110, "Val_R2": 0.3941},
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic Figure Generation
# ─────────────────────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "axes.edgecolor":  "#333333",
        "axes.linewidth":  0.8,
    })
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass


def generate_figures(
    all_results,
    summary_df: pd.DataFrame,
    per_run_df: pd.DataFrame,
    fi_df: pd.DataFrame,
    val_preds_df: pd.DataFrame,
    preprocessed_dir: Path,
):
    _style()

    # ── Figure 21: Actual vs Predicted ΔT for each model (60s) ───────────────
    for h_name in ["60s", "300s"]:
        target_col = f"target_DeltaT_{h_name}"
        h_results = [r for r in all_results if r["horizon"] == h_name and
                     r["feature_set"] == "Extended (21 feats)"]

        n_models = len(h_results)
        if n_models == 0:
            continue

        ncols = min(n_models, 4)
        nrows = (n_models + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 4.5 * nrows),
                                 squeeze=False)
        fig.suptitle(
            f"Figure 21/22: Actual vs Predicted ΔT — Extended 21 Features — Horizon {h_name}",
            fontsize=13, fontweight="bold", y=0.99,
        )

        colors = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]
        for col_idx, r in enumerate(h_results):
            row_idx = col_idx // ncols
            ci      = col_idx % ncols
            ax      = axes[row_idx][ci]

            sub = val_preds_df[
                (val_preds_df["horizon"] == h_name) &
                (val_preds_df["model_name"] == r["model_name"]) &
                (val_preds_df["feature_set"] == r["feature_set"])
            ]
            y_act  = sub["y_actual"].values
            y_pred = sub["y_pred"].values

            lim = (
                min(y_act.min(), y_pred.min()) - 0.5,
                max(y_act.max(), y_pred.max()) + 0.5,
            )
            ax.scatter(y_act, y_pred, alpha=0.3, s=10,
                       color=colors[col_idx % len(colors)], edgecolors="none")
            ax.plot(lim, lim, "k--", lw=1.2, alpha=0.7)
            ax.axhline(0, color="gray", ls=":", lw=0.8)
            ax.axvline(0, color="gray", ls=":", lw=0.8)

            vm = r["val_metrics"]
            ax.set_title(
                f"{r['model_name']}\nMAE={vm['MAE']:.3f}°C  RMSE={vm['RMSE']:.3f}°C  R²={vm['R2']:.3f}",
                fontsize=9, fontweight="bold",
            )
            ax.set_xlabel(f"Actual {target_col} (°C)", fontsize=8)
            ax.set_ylabel("Predicted ΔT (°C)", fontsize=8)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)

        # hide unused subplots
        for extra_idx in range(n_models, nrows * ncols):
            axes[extra_idx // ncols][extra_idx % ncols].set_visible(False)

        fig_num = 21 if h_name == "60s" else 22
        fpath = FIGURES_DIR / f"{fig_num}_actual_vs_predicted_{h_name}.png"
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {fpath.name}")

    # ── Figure 23: Residual Distributions (60s & 300s side-by-side) ──────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 23: Residual Distributions on Pooled Validation — Extended 21 Features",
                 fontsize=12, fontweight="bold")
    palette = {"RandomForest": "#e74c3c", "GradientBoosting": "#2980b9",
               "HistGradBoosting": "#27ae60", "XGBoost": "#8e44ad"}
    for idx, h_name in enumerate(["60s", "300s"]):
        ax = axes[idx]
        for r in all_results:
            if r["horizon"] != h_name or r["feature_set"] != "Extended (21 feats)":
                continue
            sub = val_preds_df[
                (val_preds_df["horizon"] == h_name) &
                (val_preds_df["model_name"] == r["model_name"]) &
                (val_preds_df["feature_set"] == r["feature_set"])
            ]
            residuals = sub["y_pred"].values - sub["y_actual"].values
            color = palette.get(r["model_name"], "#555555")
            ax.hist(residuals, bins=60, density=True, alpha=0.4,
                    color=color, label=r["model_name"])
        ax.axvline(0, color="black", ls="--", lw=1.2, label="Zero bias")
        ax.set_title(f"Horizon: {h_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Residual (Pred − Actual ΔT) °C", fontsize=9)
        ax.set_ylabel("Probability Density", fontsize=9)
        ax.legend(fontsize=8)
    fpath = FIGURES_DIR / "23_residual_distributions.png"
    plt.tight_layout()
    plt.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {fpath.name}")

    # ── Figure 24: Residuals vs Actual Target ─────────────────────────────────
    for h_name in ["60s", "300s"]:
        ext_results = [r for r in all_results
                       if r["horizon"] == h_name and r["feature_set"] == "Extended (21 feats)"]
        if not ext_results:
            continue
        n = len(ext_results)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
        fig.suptitle(f"Figure 24: Residuals vs Actual ΔT — Horizon {h_name} — Extended 21",
                     fontsize=11, fontweight="bold")
        colors = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]
        for ci, r in enumerate(ext_results):
            ax = axes[0][ci]
            sub = val_preds_df[
                (val_preds_df["horizon"] == h_name) &
                (val_preds_df["model_name"] == r["model_name"]) &
                (val_preds_df["feature_set"] == r["feature_set"])
            ]
            y_act  = sub["y_actual"].values
            resid  = sub["y_pred"].values - y_act
            ax.scatter(y_act, resid, alpha=0.25, s=8,
                       color=colors[ci % len(colors)], edgecolors="none")
            ax.axhline(0, color="red", ls="--", lw=1.2)
            ax.set_title(r["model_name"], fontsize=9, fontweight="bold")
            ax.set_xlabel(f"Actual ΔT_{h_name} (°C)", fontsize=8)
            ax.set_ylabel("Residual (°C)", fontsize=8)
        fig_num = "24a" if h_name == "60s" else "24b"
        fpath = FIGURES_DIR / f"{fig_num}_residuals_vs_target_{h_name}.png"
        plt.tight_layout()
        plt.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {fpath.name}")

    # ── Figure 25: Prediction Error vs Time (ID16 & ID22, best model 60s) ────
    if all_results:
        best_60 = min(
            [r for r in all_results if r["horizon"] == "60s" and
             r["feature_set"] == "Extended (21 feats)"],
            key=lambda r: r["val_metrics"]["RMSE"],
            default=None,
        )
        if best_60:
            _, val_df_60 = load_datasets(PREPROC_DIR, "60s", "extended")
            y_pred_all = np.array(best_60["val_predictions"])
            val_df_60 = val_df_60.copy()
            val_df_60["y_pred"]   = y_pred_all
            val_df_60["residual"] = y_pred_all - val_df_60["target_DeltaT_60s"]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                f"Figure 25: Error vs Time — Best Model {best_60['model_name']} "
                f"Extended 21 — Horizon 60s",
                fontsize=11, fontweight="bold",
            )
            for idx, rid in enumerate(VAL_RUNS):
                ax = axes[idx]
                sub = val_df_60[val_df_60["exp_id"] == rid]
                ax.plot(sub["time_min"], sub["residual"],
                        color=["#e74c3c", "#2980b9"][idx], lw=0.8, alpha=0.75)
                ax.axhline(0, color="black", ls="--", lw=1)
                ax.set_title(f"Run {rid}", fontsize=10, fontweight="bold")
                ax.set_xlabel("Time (min)", fontsize=9)
                ax.set_ylabel("Residual (°C)", fontsize=9)
            fpath = FIGURES_DIR / "25_error_vs_time_60s_best.png"
            plt.tight_layout()
            plt.savefig(fpath, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  [SAVED] {fpath.name}")

    # ── Figure 26: Per-run MAE bar chart ──────────────────────────────────────
    for h_name in ["60s", "300s"]:
        sub = per_run_df[per_run_df["Horizon"] == h_name].copy()
        ext = sub[sub["Feature_Set"] == "Extended (21 feats)"].copy()
        if ext.empty:
            continue
        models = ext["Model"].unique()
        x = np.arange(len(models))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        id16_mae = [ext[(ext["Model"] == m) & (ext["Run"] == "ID16")]["MAE"].values[0]
                    if not ext[(ext["Model"] == m) & (ext["Run"] == "ID16")].empty else np.nan
                    for m in models]
        id22_mae = [ext[(ext["Model"] == m) & (ext["Run"] == "ID22")]["MAE"].values[0]
                    if not ext[(ext["Model"] == m) & (ext["Run"] == "ID22")].empty else np.nan
                    for m in models]
        ax.bar(x - width / 2, id16_mae, width, label="ID16 (-120A, 15°C, 0.5 L/min)",
               color="#e74c3c", alpha=0.85)
        ax.bar(x + width / 2, id22_mae, width, label="ID22 (-80A, 8°C chilled, 2.4 L/min)",
               color="#2980b9", alpha=0.85)
        ax.set_title(f"Figure 26: Per-Run MAE — Extended 21 Features — Horizon {h_name}",
                     fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("MAE (°C)", fontsize=10)
        ax.legend(fontsize=9)
        fig_num = "26a" if h_name == "60s" else "26b"
        fpath = FIGURES_DIR / f"{fig_num}_per_run_mae_{h_name}.png"
        plt.tight_layout()
        plt.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {fpath.name}")

    # ── Figure 27: Model comparison across horizons (Val RMSE) ────────────────
    ext_df = summary_df[summary_df["Feature_Set"] == "Extended (21 feats)"].copy()
    if not ext_df.empty:
        models = ext_df["Model"].unique()
        horizons = ["60s", "300s"]
        x = np.arange(len(models))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        for hi, h_name in enumerate(horizons):
            rmse_vals = [
                ext_df[(ext_df["Model"] == m) & (ext_df["Horizon"] == h_name)]["Val_RMSE"].values[0]
                if not ext_df[(ext_df["Model"] == m) & (ext_df["Horizon"] == h_name)].empty else np.nan
                for m in models
            ]
            offset = (hi - 0.5) * width
            ax.bar(x + offset, rmse_vals, width,
                   label=f"Horizon {h_name}",
                   color=["#e74c3c", "#2980b9"][hi], alpha=0.85)
        ax.set_title("Figure 27: Validation RMSE by Model — Extended 21 Features",
                     fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Validation RMSE (°C)", fontsize=10)
        ax.legend(fontsize=9)
        fpath = FIGURES_DIR / "27_model_comparison_horizons.png"
        plt.tight_layout()
        plt.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {fpath.name}")

    # ── Figure 28: Feature Importance Heatmap ─────────────────────────────────
    if not fi_df.empty:
        for h_name in ["60s", "300s"]:
            sub = fi_df[
                (fi_df["Horizon"] == h_name) &
                (fi_df["Feature_Set"] == "Extended (21 feats)")
            ].copy()
            if sub.empty:
                continue
            pivot = sub.pivot(index="Feature", columns="Model",
                              values="Importance_Impurity").fillna(0)
            # Normalise each model column to sum to 1
            pivot = pivot.div(pivot.sum(axis=0), axis=1)
            fig_h = max(6, len(pivot) * 0.42)
            fig, ax = plt.subplots(figsize=(8, fig_h))
            im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            plt.colorbar(im, ax=ax, label="Normalised Impurity Importance")
            ax.set_title(
                f"Figure 28: Feature Importance Heatmap — Extended 21 — Horizon {h_name}",
                fontsize=10, fontweight="bold",
            )
            fig_num = "28a" if h_name == "60s" else "28b"
            fpath = FIGURES_DIR / f"{fig_num}_feature_importance_{h_name}.png"
            plt.tight_layout()
            plt.savefig(fpath, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  [SAVED] {fpath.name}")

    # ── Figure 29: CV Performance Comparison ─────────────────────────────────
    ext_df2 = summary_df[summary_df["Feature_Set"] == "Extended (21 feats)"].copy()
    if not ext_df2.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Figure 29: CV MAE vs Validation MAE — Extended 21 Features",
                     fontsize=12, fontweight="bold")
        colors_map = {"RandomForest": "#e74c3c", "GradientBoosting": "#2980b9",
                      "HistGradBoosting": "#27ae60", "XGBoost": "#8e44ad"}
        for idx, h_name in enumerate(["60s", "300s"]):
            ax = axes[idx]
            sub = ext_df2[ext_df2["Horizon"] == h_name]
            for _, row in sub.iterrows():
                color = colors_map.get(row["Model"], "#555")
                ax.scatter(row["CV_MAE"], row["Val_MAE"], s=180,
                           color=color, zorder=3, label=row["Model"])
                ax.annotate(row["Model"], (row["CV_MAE"], row["Val_MAE"]),
                            textcoords="offset points", xytext=(6, 4), fontsize=8)
            ax.plot([0, sub["CV_MAE"].max() * 1.2],
                    [0, sub["CV_MAE"].max() * 1.2], "k--", lw=1, alpha=0.5)
            ax.set_xlabel("CV MAE (°C) — training folds only", fontsize=9)
            ax.set_ylabel("Validation MAE (°C) — pooled ID16+ID22", fontsize=9)
            ax.set_title(f"Horizon {h_name}", fontsize=10, fontweight="bold")
        fpath = FIGURES_DIR / "29_cv_vs_val_mae.png"
        plt.tight_layout()
        plt.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {fpath.name}")

    # ── Figure 30: Absolute Temperature Tracking (best model, 60s) ────────────
    if all_results:
        best_60 = min(
            [r for r in all_results if r["horizon"] == "60s" and
             r["feature_set"] == "Extended (21 feats)"],
            key=lambda r: r["val_metrics"]["RMSE"],
            default=None,
        )
        if best_60:
            _, val_df_60 = load_datasets(PREPROC_DIR, "60s", "extended")
            y_pred_all = np.array(best_60["val_predictions"])
            val_df_60 = val_df_60.copy()
            val_df_60["y_pred_abs"] = val_df_60["T_cell_max"] + y_pred_all

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                f"Figure 30: Absolute Future Temp Reconstruction — "
                f"{best_60['model_name']} Extended 21 — 60s Horizon",
                fontsize=11, fontweight="bold",
            )
            for idx, rid in enumerate(VAL_RUNS):
                ax = axes[idx]
                sub = val_df_60[val_df_60["exp_id"] == rid]
                ax.plot(sub["time_min"], sub["target_T_cell_max_60s"],
                        "k-", lw=1.5, label="Actual T_cell_max(t+60s)")
                ax.plot(sub["time_min"], sub["y_pred_abs"],
                        color=["#e74c3c", "#2980b9"][idx], ls="--", lw=1.3,
                        label=f"Predicted ({best_60['model_name']})")
                ax.set_title(f"Run {rid}", fontsize=10, fontweight="bold")
                ax.set_xlabel("Time (min)", fontsize=9)
                ax.set_ylabel("Temperature (°C)", fontsize=9)
                if idx == 0:
                    ax.legend(fontsize=8)
            fpath = FIGURES_DIR / "30_abs_temp_reconstruction_60s.png"
            plt.tight_layout()
            plt.savefig(fpath, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  [SAVED] {fpath.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0_total = time.time()
    print("=" * 76)
    print("PHASE 5: MACHINE LEARNING MODEL EXPLORATION & CROSS-VALIDATION")
    print("=" * 76)
    print(f"Timestamp:        {datetime.now().isoformat()}")
    print(f"Project root:     {PROJ_ROOT}")
    print(f"Preprocessed dir: {PREPROC_DIR}")
    print(f"Results dir:      {RESULTS_DIR}")
    print(f"Figures dir:      {FIGURES_DIR}")
    print(f"Random seed:      {RANDOM_SEED}")
    print(f"CV folds:         {N_CV_FOLDS} (GroupKFold, group=exp_id)")
    print(f"Models to run:    {list(MODEL_REGISTRY.keys())}")
    print(f"TEST RUNS LOCKED: {TEST_RUNS}  <-- never touched")

    # ── Phase 5 config for reproducibility ───────────────────────────────────
    import sklearn, sys as _sys
    phase5_config = {
        "python_version":   _sys.version,
        "sklearn_version":  sklearn.__version__,
        "random_seed":      RANDOM_SEED,
        "cv_folds":         N_CV_FOLDS,
        "cv_type":          "GroupKFold",
        "group_key":        "exp_id",
        "models":           list(MODEL_REGISTRY.keys()),
        "hyperparameter_grids": {
            name: factory()[1] for name, factory in MODEL_REGISTRY.items()
        },
        "horizons":         ["60s", "300s"],
        "feature_sets":     {
            "Baseline (9 feats)":  BASELINE_9_FEATURES,
            "Extended (21 feats)": EXTENDED_21_FEATURES,
        },
        "train_runs":       TRAIN_RUNS,
        "val_runs":         VAL_RUNS,
        "test_runs_locked": TEST_RUNS,
        "preprocessed_dir": str(PREPROC_DIR),
        "execution_timestamp": datetime.now().isoformat(),
    }
    config_path = RESULTS_DIR / "phase5_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(phase5_config, f, indent=2)
    print(f"\n[SAVED] {config_path.name}")

    # ── Run full pipeline ─────────────────────────────────────────────────────
    print("\nStep 1: Running ML model cross-validation pipeline …")
    all_results, audit_list = run_full_phase5_pipeline(PREPROC_DIR)

    # ── Build and save validation predictions for figures ─────────────────────
    pred_rows = []
    for r in all_results:
        h_name = r["horizon"]
        feat_key = "baseline" if "Baseline" in r["feature_set"] else "extended"
        _, val_df = load_datasets(PREPROC_DIR, h_name, feat_key)
        target_col = f"target_DeltaT_{h_name}"
        preds = np.array(r["val_predictions"])
        for i, (_, row) in enumerate(val_df.iterrows()):
            pred_rows.append({
                "horizon":     h_name,
                "model_name":  r["model_name"],
                "feature_set": r["feature_set"],
                "exp_id":      row["exp_id"],
                "time_min":    row["time_min"],
                "y_actual":    row[target_col],
                "y_pred":      preds[i],
            })
    val_preds_df = pd.DataFrame(pred_rows)

    # ── Build summary tables ───────────────────────────────────────────────────
    print("\nStep 2: Building result tables …")
    summary_df  = build_summary_df(all_results)
    per_run_df  = build_per_run_df(all_results)
    fi_df       = build_feature_importance_df(all_results)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    summary_path  = RESULTS_DIR / "phase5_results.csv"
    per_run_path  = RESULTS_DIR / "phase5_results_per_run.csv"
    fi_path       = RESULTS_DIR / "phase5_feature_importance.csv"
    preds_path    = RESULTS_DIR / "phase5_predictions_validation.csv"

    summary_df.to_csv(summary_path,  index=False)
    per_run_df.to_csv(per_run_path,  index=False)
    fi_df.to_csv(fi_path,            index=False)
    val_preds_df.to_csv(preds_path,  index=False)
    print(f"  [SAVED] {summary_path.name}")
    print(f"  [SAVED] {per_run_path.name}")
    print(f"  [SAVED] {fi_path.name}")
    print(f"  [SAVED] {preds_path.name}")

    # ── Save audit ────────────────────────────────────────────────────────────
    audit_path = RESULTS_DIR / "phase5_audit.json"
    audit_document = {
        "overall_status": "PASS" if all(
            a["overall_status"] == "PASS" for a in audit_list
        ) else "FAIL",
        "test_runs_locked": TEST_RUNS,
        "experiments": audit_list,
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_document, f, indent=2)
    print(f"  [SAVED] {audit_path.name}")

    # ── Generate diagnostic figures ───────────────────────────────────────────
    print("\nStep 3: Generating diagnostic figures …")
    generate_figures(all_results, summary_df, per_run_df, fi_df,
                     val_preds_df, PREPROC_DIR)

    # ── Print terminal summary tables ─────────────────────────────────────────
    disp_cols = ["Horizon", "Model", "Feature_Set",
                 "CV_MAE", "CV_RMSE", "CV_R2", "Val_MAE", "Val_RMSE", "Val_R2",
                 "Val_Bias", "Val_Max_AE", "Training_Time_s"]
    print("\n" + "=" * 76)
    print("TABLE 1: POOLED VALIDATION RESULTS — ALL MODELS")
    print("=" * 76)
    print(summary_df[disp_cols].to_string(index=False))

    print("\n" + "=" * 76)
    print("TABLE 2: PER-RUN VALIDATION (ID16 vs ID22)")
    print("=" * 76)
    pr_cols = ["Horizon", "Model", "Feature_Set", "Run", "MAE", "RMSE", "R2", "Bias", "Max_AE"]
    print(per_run_df[pr_cols].to_string(index=False))

    # ── Phase 4 comparison ────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("TABLE 3: COMPARISON WITH PHASE 4 EXTENDED RIDGE / OLS BENCHMARKS")
    print("=" * 76)
    for h in ["60s", "300s"]:
        ref_ext = PHASE4_REF[h]["Extended (21 feats)"]["Ridge Regression (alpha=1.0)"]
        print(f"\n  Horizon {h}  |  Phase 4 Ridge Extended: "
              f"MAE={ref_ext['Val_MAE']:.4f}°C  RMSE={ref_ext['Val_RMSE']:.4f}°C  R²={ref_ext['Val_R2']:.4f}")
        sub = summary_df[
            (summary_df["Horizon"] == h) &
            (summary_df["Feature_Set"] == "Extended (21 feats)")
        ]
        for _, row in sub.iterrows():
            delta_mae  = row["Val_MAE"]  - ref_ext["Val_MAE"]
            delta_rmse = row["Val_RMSE"] - ref_ext["Val_RMSE"]
            delta_r2   = row["Val_R2"]   - ref_ext["Val_R2"]
            verdict = "IMPROVES" if row["Val_MAE"] < ref_ext["Val_MAE"] else "DOES NOT IMPROVE"
            print(f"    {row['Model']:26s}  Val MAE={row['Val_MAE']:.4f} "
                  f"(Δ={delta_mae:+.4f})  Val R²={row['Val_R2']:.4f} "
                  f"(Δ={delta_r2:+.4f})  → {verdict}")

    # ── Audit summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("TABLE 4: PHASE 5 SCIENTIFIC PROTOCOL & LEAKAGE AUDIT STATUS")
    print("=" * 76)
    for a in audit_list:
        print(f"  [{a['overall_status']}] {a['horizon']} | {a['feature_set']}")
        for cname, cdata in a["checks"].items():
            print(f"       - {cname}: {cdata['status']}")

    elapsed = time.time() - t0_total
    print(f"\nTotal pipeline time: {elapsed:.1f}s")
    print("\n" + "=" * 76)
    print("PHASE 5 COMPLETE. No test evaluation was performed.")
    print("Final test runs ID18 and ID19 remain strictly untouched.")
    print("=" * 76)


if __name__ == "__main__":
    main()
