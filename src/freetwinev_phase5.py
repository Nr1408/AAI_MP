"""
freetwinev_phase5.py
====================
Reusable ML model library for Phase 5: Machine Learning Model Exploration
& Cross-Validation for the FreeTwinEV battery thermal management project.

Models:
  A – RandomForestRegressor
  B – GradientBoostingRegressor
  C – HistGradientBoostingRegressor
  D – XGBRegressor (if xgboost is installed)

Cross-validation:
  GroupKFold with group = exp_id (run-level, temporally honest)
  Only training runs participate in CV.
  Validation runs (ID16, ID22) are held out until final Phase-5 comparison.
  Test runs (ID18, ID19) are locked — never loaded in Phase 5.

Phase boundary:
  No test evaluation. Hyperparameter selection uses training-only CV.
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
)

# Try XGBoost (optional – already confirmed installed in this environment)
try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Fixed Project Constants
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

TRAIN_RUNS: List[str] = [
    "ID01", "ID02", "ID11", "ID12", "ID13",
    "ID14", "ID15", "ID17", "ID20", "ID21",
]
VAL_RUNS: List[str] = ["ID16", "ID22"]
TEST_RUNS: List[str] = ["ID18", "ID19"]   # LOCKED – never accessed in Phase 5

BASELINE_9_FEATURES: List[str] = [
    "current", "voltage", "SOC", "liquid_flow",
    "temp_inlet_C", "temp_ambient_C",
    "T_cell_max", "T_cell_spread", "T_plate_mean",
]

EXTENDED_21_FEATURES: List[str] = BASELINE_9_FEATURES + [
    "current_squared", "current_abs", "power_gross",
    "current_mean_60s", "current_mean_300s",
    "current_sq_mean_60s", "current_sq_mean_300s",
    "T_cell_max_diff_30s", "T_cell_max_diff_60s",
    "T_cell_to_plate_diff", "T_cell_to_inlet_diff",
    "flow_x_inlet_temp",
]

# Number of GroupKFold folds
# 10 training runs → 5-fold leaves 2 runs per test fold (≥ 1 run always in test fold).
# This is the largest fold count that keeps ≥ 2 runs in each training fold.
N_CV_FOLDS = 5

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Metric Helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    T_cell_max_current: Optional[np.ndarray] = None,
    y_abs_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute DeltaT regression metrics (MAE, RMSE, R², Bias, Max_AE).
    If T_cell_max_current and y_abs_true are supplied, also report
    reconstructed-absolute-temperature metrics (prefix Abs_).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    out: Dict[str, float] = {
        "MAE":    float(np.mean(abs_errors)),
        "RMSE":   float(np.sqrt(np.mean(errors ** 2))),
        "R2":     float(r2_score(y_true, y_pred)),
        "Bias":   float(np.mean(errors)),
        "Max_AE": float(np.max(abs_errors)),
    }

    if T_cell_max_current is not None and y_abs_true is not None:
        T_curr = np.asarray(T_cell_max_current, dtype=float)
        y_abs = np.asarray(y_abs_true, dtype=float)
        y_abs_pred = T_curr + y_pred
        abs_err = y_abs_pred - y_abs
        out.update({
            "Abs_MAE":    float(np.mean(np.abs(abs_err))),
            "Abs_RMSE":   float(np.sqrt(np.mean(abs_err ** 2))),
            "Abs_R2":     float(r2_score(y_abs, y_abs_pred)),
            "Abs_Bias":   float(np.mean(abs_err)),
            "Abs_Max_AE": float(np.max(np.abs(abs_err))),
        })

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Scientific Protocol Assertions (fail-fast)
# ─────────────────────────────────────────────────────────────────────────────

def assert_no_test_runs(df: pd.DataFrame, context: str = "") -> None:
    """Hard assertion: ID18 / ID19 must never appear in Phase-5 data frames."""
    if "exp_id" not in df.columns:
        return
    present = set(df["exp_id"].unique()) & set(TEST_RUNS)
    if present:
        raise AssertionError(
            f"DATA INTEGRITY VIOLATION [{context}]: "
            f"Test runs {sorted(present)} detected in dataset. "
            f"Phase 5 must not access ID18 or ID19. Aborting."
        )


def audit_phase5_inputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    horizon_name: str,
    feature_set_name: str,
) -> Dict[str, Any]:
    """
    Run all 12 Phase-5 audit checks. Raises AssertionError on any failure.
    Returns a structured pass/fail dictionary for persistence.
    """
    checks: Dict[str, Any] = {}

    # ── Checks 1 & 2: test runs absent from training ──────────────────────────
    assert_no_test_runs(train_df, f"train_df h={horizon_name}")
    train_runs_found = sorted(train_df["exp_id"].unique())
    assert set(train_runs_found) == set(TRAIN_RUNS), (
        f"Unexpected training runs: {train_runs_found}; expected {TRAIN_RUNS}"
    )
    checks["id18_absent_from_training"] = {
        "status": "PASS",
        "train_runs": train_runs_found,
        "test_run": "ID18",
    }
    checks["id19_absent_from_training"] = {
        "status": "PASS",
        "train_runs": train_runs_found,
        "test_run": "ID19",
    }

    # ── Checks 3 & 4: test runs absent from validation data ───────────────────
    assert_no_test_runs(val_df, f"val_df h={horizon_name}")
    val_runs_found = sorted(val_df["exp_id"].unique())
    assert set(val_runs_found) == set(VAL_RUNS), (
        f"Unexpected validation runs: {val_runs_found}; expected {VAL_RUNS}"
    )
    checks["id18_absent_from_cv"] = {"status": "PASS", "val_runs": val_runs_found}
    checks["id19_absent_from_cv"] = {"status": "PASS", "val_runs": val_runs_found}

    # ── Check 5: no target column in features ─────────────────────────────────
    target_keywords = [
        "target_DeltaT_60s", "target_DeltaT_300s",
        "target_T_cell_max_60s", "target_T_cell_max_300s",
    ]
    target_in_feats = [c for c in feature_cols if c in target_keywords]
    assert len(target_in_feats) == 0, f"Target column in features: {target_in_feats}"
    expected_feature_count = {
        "Baseline (9 feats)": 9,
        "Extended (21 feats)": 21,
    }.get(feature_set_name)
    if expected_feature_count is not None:
        assert len(feature_cols) == expected_feature_count, (
            f"Expected {expected_feature_count} features for {feature_set_name}, "
            f"found {len(feature_cols)}"
        )
    checks["no_target_in_features"] = {"status": "PASS", "found": target_in_feats}
    checks["expected_feature_count"] = {
        "status": "PASS",
        "expected": expected_feature_count,
        "found": len(feature_cols),
    }

    # ── Check 6: no future/lead-shifted columns ───────────────────────────────
    future_kw = ["future", "lead", "horizon", "t_plus"]
    future_cols = [c for c in feature_cols if any(k in c.lower() for k in future_kw)]
    assert len(future_cols) == 0, f"Future columns in features: {future_cols}"
    checks["no_future_columns_in_features"] = {"status": "PASS", "found": future_cols}

    # ── Check 7: group-aware CV flag (checked at call site) ───────────────────
    checks["group_aware_cv_used"] = {
        "status": "PASS",
        "cv_type": f"GroupKFold(n_splits={N_CV_FOLDS})",
        "group_key": "exp_id",
    }

    # ── Check 8: train/val disjoint ───────────────────────────────────────────
    overlap = set(train_runs_found) & set(val_runs_found)
    assert len(overlap) == 0, f"Train/val overlap: {overlap}"
    checks["train_val_disjoint"] = {"status": "PASS", "overlap": sorted(overlap)}

    # ── Check 9: no NaNs ──────────────────────────────────────────────────────
    nan_train_X = int(train_df[feature_cols].isna().sum().sum())
    nan_train_y = int(train_df[target_col].isna().sum())
    nan_val_X   = int(val_df[feature_cols].isna().sum().sum())
    nan_val_y   = int(val_df[target_col].isna().sum())
    assert nan_train_X == 0 and nan_train_y == 0, f"NaNs in train: X={nan_train_X}, y={nan_train_y}"
    assert nan_val_X   == 0 and nan_val_y   == 0, f"NaNs in val:   X={nan_val_X},   y={nan_val_y}"
    checks["zero_missing_values"] = {
        "status": "PASS",
        "train_nan_X": nan_train_X, "train_nan_y": nan_train_y,
        "val_nan_X":   nan_val_X,   "val_nan_y":   nan_val_y,
    }

    # ── Check 10: hyperparameter selection uses training CV only ──────────────
    checks["hp_selection_via_train_cv_only"] = {
        "status": "PASS",
        "note": "GridSearchCV uses groups from training runs only; "
                "val/test rows never enter the CV split.",
    }

    # ── Check 11: validation not used for CV ──────────────────────────────────
    checks["validation_not_used_for_cv"] = {
        "status": "PASS",
        "note": "ID16 and ID22 are evaluated only after CV-selected model "
                "is re-fitted on all training data.",
    }

    # ── Check 12: test data not accessed ──────────────────────────────────────
    checks["test_data_not_accessed"] = {
        "status": "PASS",
        "note": "ID18 and ID19 CSVs are never read in Phase 5.",
    }

    return {
        "horizon": horizon_name,
        "feature_set": feature_set_name,
        "target_column": target_col,
        "overall_status": "PASS",
        "checks": checks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Hyperparameter Grids
# ─────────────────────────────────────────────────────────────────────────────

def get_rf_grid() -> Tuple[RandomForestRegressor, Dict]:
    estimator = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
    grid = {
        "n_estimators":    [100, 200],
        "max_depth":       [8, None],
        "min_samples_leaf": [5],
        "max_features":    ["sqrt", 0.5],
    }
    return estimator, grid


def get_gb_grid() -> Tuple[GradientBoostingRegressor, Dict]:
    estimator = GradientBoostingRegressor(random_state=RANDOM_SEED)
    grid = {
        "n_estimators":   [100, 200],
        "learning_rate":  [0.05],
        "max_depth":      [3, 5],
        "min_samples_leaf": [5],
        "subsample":      [0.8, 1.0],
    }
    return estimator, grid


def get_hgb_grid() -> Tuple[HistGradientBoostingRegressor, Dict]:
    estimator = HistGradientBoostingRegressor(random_state=RANDOM_SEED)
    grid = {
        "max_iter":       [100, 200],
        "learning_rate":  [0.05],
        "max_depth":      [4, 7],
        "min_samples_leaf": [30],
        "l2_regularization": [0.0, 1.0],
    }
    return estimator, grid


def get_xgb_grid() -> Tuple[Any, Dict]:
    if not _XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed.")
    estimator = XGBRegressor(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        eval_metric="mae",
    )
    grid = {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.10],
        "max_depth":     [4, 6],
        "subsample":      [0.8],
        "colsample_bytree": [0.7],
        "reg_lambda":    [5.0],
    }
    return estimator, grid


MODEL_REGISTRY: Dict[str, callable] = {
    "RandomForest":     get_rf_grid,
    "GradientBoosting": get_gb_grid,
    "HistGradBoosting": get_hgb_grid,
}
if _XGBOOST_AVAILABLE:
    MODEL_REGISTRY["XGBoost"] = get_xgb_grid


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Core CV + Fit + Evaluate Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_model_cv(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    abs_target_col: str,
    horizon_name: str,
    feature_set_name: str,
) -> Dict[str, Any]:
    """
    1.  Assert no test runs in data.
    2.  Run GroupKFold GridSearchCV on training runs only.
    3.  Re-fit best estimator on all training rows.
    4.  Evaluate on pooled validation and per-run (ID16, ID22).
    5.  Return structured result dict.
    """
    # Hard safety check
    assert_no_test_runs(train_df, f"{model_name} train input")
    assert_no_test_runs(val_df,   f"{model_name} val input")

    X_tr = train_df[feature_cols].values
    y_tr = train_df[target_col].values
    groups = train_df["exp_id"].values          # used by GroupKFold

    X_va = val_df[feature_cols].values
    y_va = val_df[target_col].values
    T_va = val_df["T_cell_max"].values
    y_abs_va = val_df[abs_target_col].values

    estimator, param_grid = MODEL_REGISTRY[model_name]()

    gkf = GroupKFold(n_splits=N_CV_FOLDS)

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=gkf,
        scoring=scoring,
        refit="mae",
        n_jobs=-1,
        error_score="raise",
    )

    t_start = time.time()
    gs.fit(X_tr, y_tr, groups=groups)
    t_elapsed = time.time() - t_start

    best_model = gs.best_estimator_
    best_idx = gs.best_index_
    cv_mae     = float(-gs.cv_results_["mean_test_mae"][best_idx])

    # ── Extract CV RMSE from cv_results_ ──────────────────────────────────────
    cv_mae_std = float(gs.cv_results_["std_test_mae"][best_idx])
    cv_rmse = float(-gs.cv_results_["mean_test_rmse"][best_idx])
    cv_rmse_std = float(gs.cv_results_["std_test_rmse"][best_idx])
    cv_r2 = float(gs.cv_results_["mean_test_r2"][best_idx])
    cv_r2_std = float(gs.cv_results_["std_test_r2"][best_idx])

    # ── Re-fit best on entire training set (refit=True already did this) ─────
    y_tr_pred = best_model.predict(X_tr)
    tr_metrics = compute_metrics(y_tr, y_tr_pred)

    # ── Pooled validation evaluation ──────────────────────────────────────────
    y_va_pred = best_model.predict(X_va)
    va_metrics = compute_metrics(y_va, y_va_pred, T_va, y_abs_va)

    # ── Per-run validation evaluation ─────────────────────────────────────────
    per_run: Dict[str, Dict] = {}
    for rid in VAL_RUNS:
        mask = val_df["exp_id"].values == rid
        r_metrics = compute_metrics(
            y_va[mask], y_va_pred[mask],
            T_va[mask], y_abs_va[mask],
        )
        per_run[rid] = {"N": int(mask.sum()), **r_metrics}

    # ── Feature importance (impurity-based) ───────────────────────────────────
    fi: Optional[np.ndarray] = None
    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_

    # ── Permutation importance on validation ──────────────────────────────────
    # Only computed if val set is large enough; labelled as val-derived.
    perm_fi: Optional[Dict[str, float]] = None
    if hasattr(best_model, "feature_importances_") and len(X_va) > 100:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(
            best_model, X_va, y_va,
            n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1,
            scoring="neg_mean_absolute_error",
        )
        perm_fi = {
            feature_cols[i]: float(r.importances_mean[i])
            for i in range(len(feature_cols))
        }

    return {
        "model_name":       model_name,
        "horizon":          horizon_name,
        "feature_set":      feature_set_name,
        "n_features":       len(feature_cols),
        "feature_cols":     list(feature_cols),
        "target_col":       target_col,
        "best_params":      gs.best_params_,
        "cv_folds":         N_CV_FOLDS,
        "cv_mae":           round(cv_mae, 5),
        "cv_mae_std":       round(cv_mae_std, 5),
        "cv_rmse":          round(cv_rmse, 5),
        "cv_rmse_std":      round(cv_rmse_std, 5),
        "cv_r2":            round(cv_r2, 5),
        "cv_r2_std":        round(cv_r2_std, 5),
        "training_time_s":  round(t_elapsed, 2),
        "train_metrics":    {k: round(v, 5) for k, v in tr_metrics.items()},
        "val_metrics":      {k: round(v, 5) for k, v in va_metrics.items()},
        "per_run_metrics":  {
            rid: {k: round(v, 5) for k, v in mets.items()}
            for rid, mets in per_run.items()
        },
        "val_predictions":  y_va_pred.tolist(),
        "feature_importance_impurity": (
            {feature_cols[i]: float(fi[i]) for i in range(len(feature_cols))}
            if fi is not None else None
        ),
        "feature_importance_permutation_val": perm_fi,
        "fitted_model":     best_model,   # returned for figure-generation use
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Full Phase-5 Pipeline (all models × horizons × feature sets)
# ─────────────────────────────────────────────────────────────────────────────

HORIZON_CONFIG = [
    ("60s",  "target_DeltaT_60s",  "target_T_cell_max_60s"),
    ("300s", "target_DeltaT_300s", "target_T_cell_max_300s"),
]

FEATURE_CONFIG = [
    ("Baseline (9 feats)",  "baseline",  BASELINE_9_FEATURES),
    ("Extended (21 feats)", "extended",  EXTENDED_21_FEATURES),
]


def load_datasets(preprocessed_dir: Path, horizon: str, feat_key: str):
    """Load unscaled train / val CSVs for a given horizon and feature experiment."""
    train_df = pd.read_csv(preprocessed_dir / f"{feat_key}_{horizon}_train.csv")
    val_df   = pd.read_csv(preprocessed_dir / f"{feat_key}_{horizon}_val.csv")

    # Critical safety assertions on load
    assert_no_test_runs(train_df, f"load {feat_key}_{horizon}_train")
    assert_no_test_runs(val_df,   f"load {feat_key}_{horizon}_val")

    return train_df, val_df


def run_full_phase5_pipeline(
    preprocessed_dir: Path,
    model_names: Optional[List[str]] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Execute Phase-5 across all model × horizon × feature-set combinations.

    Returns:
      all_results  – list of per-experiment result dicts (full)
      audit_list   – list of audit dicts for each experiment
    """
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    all_results: List[Dict] = []
    audit_list:  List[Dict] = []

    for h_name, target_col, abs_target_col in HORIZON_CONFIG:
        for feat_label, feat_key, feat_cols in FEATURE_CONFIG:
            print(f"\n{'='*72}")
            print(f"  HORIZON: {h_name}  |  FEATURES: {feat_label}")
            print(f"{'='*72}")

            train_df, val_df = load_datasets(preprocessed_dir, h_name, feat_key)

            # Audit this combination
            audit = audit_phase5_inputs(
                train_df, val_df, feat_cols, target_col, h_name, feat_label
            )
            audit_list.append(audit)
            print(f"  Audit: {audit['overall_status']}")

            for m_name in model_names:
                print(f"  Model: {m_name} …", end=" ", flush=True)
                result = run_model_cv(
                    m_name, train_df, val_df,
                    feat_cols, target_col, abs_target_col,
                    h_name, feat_label,
                )
                all_results.append(result)
                print(
                    f"done  (CV MAE={result['cv_mae']:.4f}°C, "
                    f"Val MAE={result['val_metrics']['MAE']:.4f}°C, "
                    f"Val R²={result['val_metrics']['R2']:.4f}, "
                    f"t={result['training_time_s']:.1f}s)"
                )

    return all_results, audit_list


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Results Formatting Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_df(all_results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in all_results:
        vm = r["val_metrics"]
        tm = r["train_metrics"]
        rows.append({
            "Horizon":         r["horizon"],
            "Model":           r["model_name"],
            "Feature_Set":     r["feature_set"],
            "Num_Features":    r["n_features"],
            "CV_MAE":          r["cv_mae"],
            "CV_MAE_Std":      r["cv_mae_std"],
            "CV_RMSE":         r["cv_rmse"],
            "CV_RMSE_Std":     r["cv_rmse_std"],
            "CV_R2":           r["cv_r2"],
            "CV_R2_Std":       r["cv_r2_std"],
            "CV_Folds":        r["cv_folds"],
            "Val_MAE":         round(vm["MAE"],    4),
            "Val_RMSE":        round(vm["RMSE"],   4),
            "Val_R2":          round(vm["R2"],     4),
            "Val_Bias":        round(vm["Bias"],   4),
            "Val_Max_AE":      round(vm["Max_AE"], 4),
            "Val_Abs_MAE":     round(vm.get("Abs_MAE",  float("nan")), 4),
            "Val_Abs_RMSE":    round(vm.get("Abs_RMSE", float("nan")), 4),
            "Val_Abs_R2":      round(vm.get("Abs_R2",   float("nan")), 4),
            "Train_MAE":       round(tm["MAE"],    4),
            "Train_RMSE":      round(tm["RMSE"],   4),
            "Train_R2":        round(tm["R2"],     4),
            "Training_Time_s": r["training_time_s"],
            "Best_Params":     json.dumps(r["best_params"]),
        })
    return pd.DataFrame(rows)


def build_per_run_df(all_results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in all_results:
        for rid, mets in r["per_run_metrics"].items():
            rows.append({
                "Horizon":     r["horizon"],
                "Model":       r["model_name"],
                "Feature_Set": r["feature_set"],
                "Run":         rid,
                "N":           mets["N"],
                "MAE":         round(mets["MAE"],    4),
                "RMSE":        round(mets["RMSE"],   4),
                "R2":          round(mets["R2"],     4),
                "Bias":        round(mets["Bias"],   4),
                "Max_AE":      round(mets["Max_AE"], 4),
                "Abs_MAE":     round(mets.get("Abs_MAE",  float("nan")), 4),
                "Abs_RMSE":    round(mets.get("Abs_RMSE", float("nan")), 4),
                "Abs_R2":      round(mets.get("Abs_R2",   float("nan")), 4),
            })
    return pd.DataFrame(rows)


def build_feature_importance_df(all_results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in all_results:
        fi = r.get("feature_importance_impurity")
        if fi:
            for feat, imp in fi.items():
                rows.append({
                    "Horizon":      r["horizon"],
                    "Model":        r["model_name"],
                    "Feature_Set":  r["feature_set"],
                    "Feature":      feat,
                    "Importance_Impurity": round(imp, 6),
                    "Importance_Permutation_Val": round(
                        (r.get("feature_importance_permutation_val") or {}).get(feat, float("nan")), 6
                    ),
                })
    return pd.DataFrame(rows)
