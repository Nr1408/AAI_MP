"""Causal lag construction and temporal exploration for battery temperature.

All functions build rows at time t using information from t or earlier only;
the response is explicitly shifted to a future timestamp.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.data_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN, TIMESTAMP_COLUMN

LAG_WINDOWS = (0, 1, 2, 5, 10)
FORECAST_HORIZONS = (1, 2, 5, 10)


def _validate_temporal_data(data: pd.DataFrame) -> None:
    if TIMESTAMP_COLUMN not in data or TARGET_COLUMN not in data:
        raise ValueError("Timestamp and MaxTemp_C columns are required for temporal analysis.")
    if not data[TIMESTAMP_COLUMN].is_monotonic_increasing:
        raise ValueError("Temporal analysis requires chronologically ordered rows.")


def create_lagged_frame(
    data: pd.DataFrame,
    horizon_minutes: int,
    feature_lags: tuple[int, ...] = LAG_WINDOWS,
) -> pd.DataFrame:
    """Return causally aligned features at t and target temperature at t+h.

    The returned target is ``MaxTemp_C(t+h)``. A feature ending in ``lag_0``
    comes from t; positive lags come from the past. No t+1...t+h observation
    is placed in the feature matrix.
    """
    _validate_temporal_data(data)
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive.")
    if any(lag < 0 for lag in feature_lags):
        raise ValueError("Feature lags must be zero or positive.")

    frame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: data[TIMESTAMP_COLUMN],
            "target_timestamp": data[TIMESTAMP_COLUMN].shift(-horizon_minutes),
            f"{TARGET_COLUMN}_future_{horizon_minutes}m": data[TARGET_COLUMN].shift(-horizon_minutes),
        }
    )
    source_columns = [TARGET_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    for column in source_columns:
        for lag in feature_lags:
            frame[f"{column}_lag_{lag}"] = data[column].shift(lag)

    return frame.dropna().reset_index(drop=True)


def future_temperature_autocorrelation(data: pd.DataFrame, horizons: tuple[int, ...] = FORECAST_HORIZONS) -> pd.DataFrame:
    """Measure correlation between current and future target temperature."""
    _validate_temporal_data(data)
    rows = []
    for horizon in horizons:
        rows.append(
            {
                "horizon_minutes": horizon,
                "max_temp_autocorrelation": data[TARGET_COLUMN].corr(data[TARGET_COLUMN].shift(-horizon)),
            }
        )
    return pd.DataFrame(rows)


def numeric_lag_correlations(
    data: pd.DataFrame,
    horizon_minutes: int,
    lags: tuple[int, ...] = LAG_WINDOWS,
) -> pd.DataFrame:
    """Correlate each allowed numeric past/current value with future temperature."""
    _validate_temporal_data(data)
    future_target = data[TARGET_COLUMN].shift(-horizon_minutes)
    rows = []
    for feature in [TARGET_COLUMN, *NUMERIC_FEATURES]:
        for lag in lags:
            rows.append(
                {
                    "feature": feature,
                    "lag_minutes": lag,
                    "correlation_with_future_max_temp": data[feature].shift(lag).corr(future_target),
                }
            )
    return pd.DataFrame(rows)


def categorical_future_target_summary(
    data: pd.DataFrame,
    horizon_minutes: int,
    lags: tuple[int, ...] = LAG_WINDOWS,
) -> pd.DataFrame:
    """Summarize future temperature by causal categorical states at each lag."""
    _validate_temporal_data(data)
    future_target = data[TARGET_COLUMN].shift(-horizon_minutes)
    rows = []
    for feature in CATEGORICAL_FEATURES:
        for lag in lags:
            aligned = pd.DataFrame(
                {"category": data[feature].shift(lag), "future_target": future_target}
            ).dropna()
            grouped = aligned.groupby("category", dropna=False)["future_target"].agg(["count", "mean"])
            for category, values in grouped.iterrows():
                rows.append(
                    {
                        "feature": feature,
                        "lag_minutes": lag,
                        "category": category,
                        "count": int(values["count"]),
                        "future_max_temp_mean": float(values["mean"]),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_persistence_baseline(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score MaxTemp_C(t) -> MaxTemp_C(t+h) on development-only time folds.

    This model-free reference uses the current measured maximum temperature as
    its prediction and never uses information after the feature timestamp.
    """
    _validate_temporal_data(data)
    fold_rows = []
    for horizon in horizons:
        frame = create_lagged_frame(data, horizon, feature_lags=(0,))
        actual = frame[f"{TARGET_COLUMN}_future_{horizon}m"]
        predicted = frame[f"{TARGET_COLUMN}_lag_0"]
        splitter = TimeSeriesSplit(n_splits=n_splits)
        for fold_number, (_, validation_indices) in enumerate(splitter.split(frame), start=1):
            fold_actual = actual.iloc[validation_indices]
            fold_predicted = predicted.iloc[validation_indices]
            fold_rows.append(
                {
                    "horizon_minutes": horizon,
                    "fold": fold_number,
                    "MAE": mean_absolute_error(fold_actual, fold_predicted),
                    "RMSE": root_mean_squared_error(fold_actual, fold_predicted),
                    "R2": r2_score(fold_actual, fold_predicted),
                }
            )
    fold_results = pd.DataFrame(fold_rows)
    summary = (
        fold_results.groupby("horizon_minutes")[["MAE", "RMSE", "R2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "horizon_minutes" if column[0] == "horizon_minutes" else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    return fold_results, summary


def create_temporal_figures(
    development_data: pd.DataFrame,
    output_directory: str | Path,
    horizon_minutes: int = 5,
) -> list[Path]:
    """Create development-only figures for causal temporal investigation."""
    _validate_temporal_data(development_data)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(development_data[TIMESTAMP_COLUMN], development_data[TARGET_COLUMN], color="#0f766e", linewidth=1.1)
    axis.set_title("Development Period: Maximum Temperature Over Time")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("MaxTemp_C")
    figure.autofmt_xdate()
    paths.append(output_path / "temporal_max_temp_development.png")
    figure.tight_layout()
    figure.savefig(paths[-1], dpi=160)
    plt.close(figure)

    autocorrelation = future_temperature_autocorrelation(development_data, tuple(range(1, 31)))
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.stem(autocorrelation["horizon_minutes"], autocorrelation["max_temp_autocorrelation"], linefmt="#7c3aed", markerfmt="o")
    axis.axhline(0, color="#374151", linewidth=1)
    axis.axvline(horizon_minutes, color="#b45309", linestyle="--", label=f"{horizon_minutes}-minute horizon")
    axis.set_title("MaxTemp_C Persistence: Current vs Future Temperature")
    axis.set_xlabel("Future horizon (minutes)")
    axis.set_ylabel("Pearson correlation")
    axis.legend()
    paths.append(output_path / "max_temp_autocorrelation.png")
    figure.tight_layout()
    figure.savefig(paths[-1], dpi=160)
    plt.close(figure)

    aligned = create_lagged_frame(development_data, horizon_minutes, feature_lags=(0,))
    future_column = f"{TARGET_COLUMN}_future_{horizon_minutes}m"
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    sns.regplot(
        data=aligned,
        x=f"{TARGET_COLUMN}_lag_0",
        y=future_column,
        scatter_kws={"alpha": 0.65, "color": "#0f766e"},
        line_kws={"color": "#b45309"},
        ax=axis,
    )
    axis.set_title(f"Current vs {horizon_minutes}-Minute Future Temperature")
    axis.set_xlabel("MaxTemp_C at time t")
    axis.set_ylabel(f"MaxTemp_C at t+{horizon_minutes}")
    paths.append(output_path / "previous_vs_future_temperature.png")
    figure.tight_layout()
    figure.savefig(paths[-1], dpi=160)
    plt.close(figure)

    correlations = numeric_lag_correlations(development_data, horizon_minutes)
    heatmap_values = correlations.pivot(
        index="feature", columns="lag_minutes", values="correlation_with_future_max_temp"
    )
    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(heatmap_values, cmap="vlag", center=0, annot=True, fmt=".2f", ax=axis)
    axis.set_title(f"Causal Lag Correlations With MaxTemp_C(t+{horizon_minutes})")
    axis.set_xlabel("Feature lag in minutes")
    axis.set_ylabel("Feature available at or before t")
    paths.append(output_path / "lag_correlations_to_future_temperature.png")
    figure.tight_layout()
    figure.savefig(paths[-1], dpi=160)
    plt.close(figure)

    return paths
