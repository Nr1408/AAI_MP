"""Metrics and visual evaluation utilities for the thermal prediction models."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return standard regression metrics with explicit, human-readable names."""
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(root_mean_squared_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)),
    }


def plot_actual_vs_predicted(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
    model_name: str,
    output_path: str | Path,
) -> None:
    """Plot final-test actual versus predicted temperatures with an identity line."""
    sns.set_theme(style="whitegrid", context="notebook")
    actual_values = np.asarray(actual)
    predicted_values = np.asarray(predicted)
    lower = min(actual_values.min(), predicted_values.min())
    upper = max(actual_values.max(), predicted_values.max())

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.scatter(actual_values, predicted_values, alpha=0.75, color="#0f766e", edgecolor="white", linewidth=0.4)
    axis.plot([lower, upper], [lower, upper], color="#b45309", linestyle="--", label="Ideal prediction")
    axis.set_xlabel("Actual MaxTemp_C")
    axis.set_ylabel("Predicted MaxTemp_C")
    axis.set_title(f"Actual vs Predicted Temperature: {model_name}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_residuals(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
    timestamps: pd.Series,
    model_name: str,
    output_path: str | Path,
) -> None:
    """Plot chronological test residuals (actual minus predicted)."""
    residuals = np.asarray(actual) - np.asarray(predicted)
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.axhline(0, color="#374151", linewidth=1)
    axis.scatter(timestamps, residuals, color="#7c3aed", alpha=0.8, s=24)
    axis.set_title(f"Chronological Test Residuals: {model_name}")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Residual (actual - predicted, C)")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_tree_feature_importance(
    fitted_models: dict[str, object], output_path: str | Path
) -> pd.DataFrame:
    """Render feature importances for fitted tree models and return their values."""
    tree_models = {
        name: model
        for name, model in fitted_models.items()
        if hasattr(model.named_steps["regressor"], "feature_importances_")
    }
    if not tree_models:
        raise ValueError("No fitted tree model with feature_importances_ was provided.")

    first_model = next(iter(tree_models.values()))
    feature_names = first_model.named_steps["preprocessor"].get_feature_names_out()
    importance_data = pd.DataFrame({"feature": feature_names})

    for name, model in tree_models.items():
        importance_data[name] = model.named_steps["regressor"].feature_importances_

    long_data = importance_data.melt(id_vars="feature", var_name="model", value_name="importance")
    figure, axis = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=long_data.sort_values("importance", ascending=False),
        x="importance",
        y="feature",
        hue="model",
        orient="h",
        ax=axis,
    )
    axis.set_title("Tree-Model Feature Importances")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Preprocessed feature")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return importance_data
