"""
freetwinev_baselines.py
=======================
Defines and evaluates baseline predictive models for FreeTwinEV battery
thermal management under the locked Phase 3 experimental protocol.

Baselines evaluated:
1. Naive Persistence Baseline: Predicts DeltaT = 0 (T_future = T_current).
2. Mean Baseline: Predicts training-set mean target DeltaT.
3. Linear Regression: Ordinary Least Squares (OLS) on 9-feature Baseline set.
4. Ridge Regression: L2-regularized linear model (fixed alpha=1.0) on scaled features.

Constraints enforced:
- Training on Train runs ONLY (ID01, ID02, ID11, ID12, ID13, ID14, ID15, ID17, ID20, ID21).
- Evaluation on Validation runs ONLY (ID16, ID22), pooled and per-run.
- Final test runs (ID18, ID19) remain strictly UNTOUCHED.
- No future target leakage or random row splitting.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


# Standard split protocol definitions
TRAIN_RUNS = [
    "ID01", "ID02", "ID11", "ID12", "ID13",
    "ID14", "ID15", "ID17", "ID20", "ID21"
]
VAL_RUNS = ["ID16", "ID22"]
TEST_RUNS = ["ID18", "ID19"]

BASELINE_9_FEATURES = [
    "current",
    "voltage",
    "SOC",
    "liquid_flow",
    "temp_inlet_C",
    "temp_ambient_C",
    "T_cell_max",
    "T_cell_spread",
    "T_plate_mean"
]

EXTENDED_21_FEATURES = BASELINE_9_FEATURES + [
    "current_squared",
    "current_abs",
    "power_gross",
    "current_mean_60s",
    "current_mean_300s",
    "current_sq_mean_60s",
    "current_sq_mean_300s",
    "T_cell_max_diff_30s",
    "T_cell_max_diff_60s",
    "T_cell_to_plate_diff",
    "T_cell_to_inlet_diff",
    "flow_x_inlet_temp"
]


# =============================================================================
# 1. Model Baseline Classes
# =============================================================================

class NaivePersistenceBaseline:
    """
    Predicts DeltaT = 0 for any horizon.
    Reconstructs absolute temperature as T_future = T_current.
    """
    def __init__(self):
        self.name = "Naive Persistence"
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaivePersistenceBaseline":
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class MeanBaseline:
    """
    Predicts constant mean target value learned strictly from training data.
    """
    def __init__(self):
        self.name = "Mean Baseline"
        self.mean_value_: Optional[float] = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanBaseline":
        self.mean_value_ = float(np.mean(y))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return np.full(len(X), self.mean_value_, dtype=float)


class LinearRegressionBaseline:
    """
    Ordinary Least Squares (OLS) Linear Regression benchmark.
    """
    def __init__(self, name: str = "Linear Regression (OLS)"):
        self.name = name
        self.model = LinearRegression()
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionBaseline":
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self.model.predict(X)


class RidgeRegressionBaseline:
    """
    L2-regularized Ridge Regression benchmark with fixed alpha.
    """
    def __init__(self, alpha: float = 1.0, name: str = "Ridge Regression (alpha=1.0)"):
        self.name = name
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, random_state=42)
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeRegressionBaseline":
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self.model.predict(X)


# =============================================================================
# 2. Metric Computation & Absolute Temperature Reconstruction
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    T_cell_max_current: Optional[np.ndarray] = None,
    y_abs_true: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes regression evaluation metrics for delta-temperature targets:
    - MAE (Mean Absolute Error, degC)
    - RMSE (Root Mean Squared Error, degC)
    - R2 (Coefficient of Determination)
    - Bias (Mean Error = mean(y_pred - y_true), degC)
    - Max_AE (Maximum Absolute Error, degC)

    If current T_cell_max and true absolute future temperature are provided,
    computes secondary reconstructed absolute future temperature metrics:
    T_pred_future = T_cell_max_current + y_pred.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(errors))
    max_ae = float(np.max(abs_errors))

    results = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Bias": bias,
        "Max_AE": max_ae
    }

    if T_cell_max_current is not None and y_abs_true is not None:
        T_cell_max_current = np.asarray(T_cell_max_current, dtype=float)
        y_abs_true = np.asarray(y_abs_true, dtype=float)
        y_abs_pred = T_cell_max_current + y_pred

        abs_err = y_abs_pred - y_abs_true
        results["Abs_MAE"] = float(np.mean(np.abs(abs_err)))
        results["Abs_RMSE"] = float(np.sqrt(np.mean(abs_err ** 2)))
        results["Abs_R2"] = float(r2_score(y_abs_true, y_abs_pred))
        results["Abs_Bias"] = float(np.mean(abs_err))
        results["Abs_Max_AE"] = float(np.max(np.abs(abs_err)))

    return results


# =============================================================================
# 3. Scientific Leakage & Protocol Assertions
# =============================================================================

def audit_baseline_protocol(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    horizon_name: str
) -> Dict[str, Any]:
    """
    Rigorous pre-training scientific checks confirming zero leakage and
    strict adherence to the run-level split protocol.
    """
    audit_results = {
        "horizon": horizon_name,
        "target_column": target_col,
        "features_checked": list(feature_cols),
        "checks": {}
    }

    train_runs = set(train_df["exp_id"].unique())
    val_runs = set(val_df["exp_id"].unique())

    # Check 1: Test runs are NEVER loaded into training or validation
    overlap_train_test = train_runs.intersection(set(TEST_RUNS))
    overlap_val_test = val_runs.intersection(set(TEST_RUNS))
    assert len(overlap_train_test) == 0, f"LEAKAGE: Test runs found in train: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"LEAKAGE: Test runs found in val: {overlap_val_test}"
    audit_results["checks"]["no_test_runs_in_train_or_val"] = {
        "status": "PASS",
        "train_runs": sorted(list(train_runs)),
        "val_runs": sorted(list(val_runs)),
        "test_runs_excluded": TEST_RUNS
    }

    # Check 2: No target columns appear in feature set
    forbidden_targets = [
        "target_DeltaT_60s", "target_DeltaT_300s",
        "target_T_cell_max_60s", "target_T_cell_max_300s"
    ]
    target_in_features = [col for col in feature_cols if col in forbidden_targets]
    assert len(target_in_features) == 0, f"LEAKAGE: Target column in features: {target_in_features}"
    audit_results["checks"]["no_target_in_features"] = {
        "status": "PASS",
        "target_in_features": target_in_features
    }

    # Check 3: No future sensor measurements or future leads
    future_keywords = ["future", "lead", "horizon", "t_plus"]
    future_cols = [c for c in feature_cols if any(k in c.lower() for k in future_keywords)]
    assert len(future_cols) == 0, f"LEAKAGE: Future columns detected in features: {future_cols}"
    audit_results["checks"]["no_future_measurements_in_features"] = {
        "status": "PASS",
        "flagged_future_cols": future_cols
    }

    # Check 4: Zero run overlap between train and validation
    run_overlap = train_runs.intersection(val_runs)
    assert len(run_overlap) == 0, f"LEAKAGE: Overlap between train and val runs: {run_overlap}"
    audit_results["checks"]["zero_run_overlap_train_val"] = {
        "status": "PASS",
        "overlap": list(run_overlap)
    }

    # Check 5: Expected run membership
    expected_train = set(TRAIN_RUNS)
    expected_val = set(VAL_RUNS)
    assert train_runs == expected_train, f"MISMATCH: Train runs {train_runs} != {expected_train}"
    assert val_runs == expected_val, f"MISMATCH: Val runs {val_runs} != {expected_val}"
    audit_results["checks"]["exact_expected_runs"] = {
        "status": "PASS",
        "train_count": len(train_runs),
        "val_count": len(val_runs)
    }

    # Check 6: Zero NaNs in features and targets
    train_nans_X = int(train_df[feature_cols].isna().sum().sum())
    train_nans_y = int(train_df[target_col].isna().sum())
    val_nans_X = int(val_df[feature_cols].isna().sum().sum())
    val_nans_y = int(val_df[target_col].isna().sum())
    assert train_nans_X == 0 and train_nans_y == 0, f"NaNs in train: X={train_nans_X}, y={train_nans_y}"
    assert val_nans_X == 0 and val_nans_y == 0, f"NaNs in val: X={val_nans_X}, y={val_nans_y}"
    audit_results["checks"]["zero_missing_values"] = {
        "status": "PASS",
        "train_nan_X": train_nans_X,
        "train_nan_y": train_nans_y,
        "val_nan_X": val_nans_X,
        "val_nan_y": val_nans_y
    }

    audit_results["overall_status"] = "PASS"
    return audit_results


# =============================================================================
# 4. End-to-End Baseline Evaluation Pipeline
# =============================================================================

def run_baseline_pipeline(
    preprocessed_dir: Path,
    output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes the full baseline benchmarking workflow across 60s and 300s horizons.

    Returns:
    - pooled_results_df: Summary of metrics across all baselines on validation & train.
    - per_run_results_df: Per-run breakdown on validation runs (ID16, ID22).
    - predictions_df: Validation predictions table for residual diagnostics.
    - audit_summary: Comprehensive machine-readable audit report.
    """
    pooled_records = []
    per_run_records = []
    val_pred_frames = []
    audit_summary = {}

    horizons = [
        ("60s", "target_DeltaT_60s", "target_T_cell_max_60s"),
        ("300s", "target_DeltaT_300s", "target_T_cell_max_300s")
    ]

    for h_name, target_col, abs_target_col in horizons:
        audit_summary[h_name] = {}

        # ---------------------------------------------------------
        # Load Datasets: Unscaled & Scaled
        # ---------------------------------------------------------
        # Baseline 9-feature sets
        train_base_path = preprocessed_dir / f"baseline_{h_name}_train.csv"
        val_base_path = preprocessed_dir / f"baseline_{h_name}_val.csv"
        train_base_df = pd.read_csv(train_base_path)
        val_base_df = pd.read_csv(val_base_path)

        train_base_sc_path = preprocessed_dir / "scaled" / f"baseline_{h_name}_train_scaled.csv"
        val_base_sc_path = preprocessed_dir / "scaled" / f"baseline_{h_name}_val_scaled.csv"
        train_base_sc_df = pd.read_csv(train_base_sc_path)
        val_base_sc_df = pd.read_csv(val_base_sc_path)

        # Extended 21-feature sets (for direct feature comparison)
        train_ext_sc_path = preprocessed_dir / "scaled" / f"extended_{h_name}_train_scaled.csv"
        val_ext_sc_path = preprocessed_dir / "scaled" / f"extended_{h_name}_val_scaled.csv"
        train_ext_sc_df = pd.read_csv(train_ext_sc_path)
        val_ext_sc_df = pd.read_csv(val_ext_sc_path)

        # ---------------------------------------------------------
        # Run Scientific Protocol Audit
        # ---------------------------------------------------------
        audit_base = audit_baseline_protocol(
            train_base_df, val_base_df, BASELINE_9_FEATURES, target_col, f"{h_name}_baseline"
        )
        audit_summary[h_name]["baseline_audit"] = audit_base

        audit_ext = audit_baseline_protocol(
            train_ext_sc_df, val_ext_sc_df, EXTENDED_21_FEATURES, target_col, f"{h_name}_extended"
        )
        audit_summary[h_name]["extended_audit"] = audit_ext

        # Common ground truth validation vectors
        y_val_true = val_base_df[target_col].values
        T_val_curr = val_base_df["T_cell_max"].values
        y_abs_val_true = val_base_df[abs_target_col].values

        y_train_true = train_base_df[target_col].values
        T_train_curr = train_base_df["T_cell_max"].values
        y_abs_train_true = train_base_df[abs_target_col].values

        # Base prediction frame to hold validation predictions for diagnostics
        pred_frame = val_base_df[["exp_id", "time_s", "time_min", "T_cell_max"]].copy()
        pred_frame["horizon"] = h_name
        pred_frame[target_col] = y_val_true
        pred_frame[abs_target_col] = y_abs_val_true

        # ---------------------------------------------------------
        # Define Models to Evaluate
        # ---------------------------------------------------------
        models = [
            (
                "Naive Persistence",
                "None (DeltaT=0)",
                NaivePersistenceBaseline(),
                train_base_df[BASELINE_9_FEATURES],
                val_base_df[BASELINE_9_FEATURES],
                False
            ),
            (
                "Mean Baseline",
                "None (Train Mean)",
                MeanBaseline(),
                train_base_df[BASELINE_9_FEATURES],
                val_base_df[BASELINE_9_FEATURES],
                False
            ),
            (
                "Linear Regression",
                "Baseline (9 feats)",
                LinearRegressionBaseline("Linear Regression (Baseline 9)"),
                train_base_df[BASELINE_9_FEATURES],
                val_base_df[BASELINE_9_FEATURES],
                False
            ),
            (
                "Ridge Regression (alpha=1.0)",
                "Baseline (9 feats)",
                RidgeRegressionBaseline(alpha=1.0, name="Ridge (Baseline 9, alpha=1.0)"),
                train_base_sc_df[BASELINE_9_FEATURES],
                val_base_sc_df[BASELINE_9_FEATURES],
                True
            ),
            (
                "Linear Regression",
                "Extended (21 feats)",
                LinearRegressionBaseline("Linear Regression (Extended 21)"),
                train_ext_sc_df[EXTENDED_21_FEATURES],
                val_ext_sc_df[EXTENDED_21_FEATURES],
                True
            ),
            (
                "Ridge Regression (alpha=1.0)",
                "Extended (21 feats)",
                RidgeRegressionBaseline(alpha=1.0, name="Ridge (Extended 21, alpha=1.0)"),
                train_ext_sc_df[EXTENDED_21_FEATURES],
                val_ext_sc_df[EXTENDED_21_FEATURES],
                True
            )
        ]

        for model_name, feature_desc, model_obj, X_tr, X_va, is_scaled in models:
            # 1. Fit strictly on train
            model_obj.fit(X_tr, train_base_df[target_col])

            # 2. Predict on train (for bias/underfit check)
            y_tr_pred = model_obj.predict(X_tr)
            tr_metrics = compute_metrics(
                y_train_true, y_tr_pred, T_train_curr, y_abs_train_true
            )

            # 3. Predict on pooled validation
            y_va_pred = model_obj.predict(X_va)
            va_metrics = compute_metrics(
                y_val_true, y_va_pred, T_val_curr, y_abs_val_true
            )

            # Save prediction column in pred_frame
            pred_col_name = f"pred_DeltaT_{model_name}_{feature_desc.split()[0]}"
            pred_frame[pred_col_name] = y_va_pred
            pred_frame[f"pred_T_future_{model_name}_{feature_desc.split()[0]}"] = T_val_curr + y_va_pred
            pred_frame[f"residual_{model_name}_{feature_desc.split()[0]}"] = y_va_pred - y_val_true

            # Record pooled record
            pooled_records.append({
                "Horizon": h_name,
                "Model": model_name,
                "Feature_Set": feature_desc,
                "Num_Features": X_tr.shape[1],
                "Scaled": is_scaled,
                "Val_MAE": round(va_metrics["MAE"], 4),
                "Val_RMSE": round(va_metrics["RMSE"], 4),
                "Val_R2": round(va_metrics["R2"], 4),
                "Val_Bias": round(va_metrics["Bias"], 4),
                "Val_Max_AE": round(va_metrics["Max_AE"], 4),
                "Val_Abs_MAE": round(va_metrics["Abs_MAE"], 4),
                "Val_Abs_RMSE": round(va_metrics["Abs_RMSE"], 4),
                "Val_Abs_R2": round(va_metrics["Abs_R2"], 4),
                "Train_MAE": round(tr_metrics["MAE"], 4),
                "Train_RMSE": round(tr_metrics["RMSE"], 4),
                "Train_R2": round(tr_metrics["R2"], 4),
                "Train_Bias": round(tr_metrics["Bias"], 4)
            })

            # 4. Per-run validation evaluation
            for rid in VAL_RUNS:
                mask = (val_base_df["exp_id"] == rid).values
                y_va_r = y_val_true[mask]
                y_pred_r = y_va_pred[mask]
                T_curr_r = T_val_curr[mask]
                y_abs_r = y_abs_val_true[mask]

                r_metrics = compute_metrics(y_va_r, y_pred_r, T_curr_r, y_abs_r)
                per_run_records.append({
                    "Horizon": h_name,
                    "Model": model_name,
                    "Feature_Set": feature_desc,
                    "Run": rid,
                    "Num_Samples": int(mask.sum()),
                    "MAE": round(r_metrics["MAE"], 4),
                    "RMSE": round(r_metrics["RMSE"], 4),
                    "R2": round(r_metrics["R2"], 4),
                    "Bias": round(r_metrics["Bias"], 4),
                    "Max_AE": round(r_metrics["Max_AE"], 4),
                    "Abs_MAE": round(r_metrics["Abs_MAE"], 4),
                    "Abs_RMSE": round(r_metrics["Abs_RMSE"], 4),
                    "Abs_R2": round(r_metrics["Abs_R2"], 4)
                })

        val_pred_frames.append(pred_frame)

    pooled_df = pd.DataFrame(pooled_records)
    per_run_df = pd.DataFrame(per_run_records)
    predictions_df = pd.concat(val_pred_frames, ignore_index=True)

    return pooled_df, per_run_df, predictions_df, audit_summary
