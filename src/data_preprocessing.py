"""Leakage-aware dataset preparation for the EV thermal prototype.

This module deliberately does not train a predictive model. It validates the
source data, preserves original measurements, defines a chronological split,
and constructs an unfitted preprocessing transformer for a later model phase.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "MaxTemp_C"
TIMESTAMP_COLUMN = "Timestamp"

# Sensor and operating fields selected for the first, conservative model.
NUMERIC_FEATURES = [
    "PackVoltage_V",
    "CellVoltage_V",
    "ChargeCurrent_A",
    "SOC_%",
    "AmbientTemp_C",
    "InternalResistance_mOhm",
    "StateOfHealth_%",
    "VibrationLevel_mg",
    "Pressure_kPa",
]
CATEGORICAL_FEATURES = ["ChargingStage", "MoistureDetected"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXCLUDED_COLUMNS = {
    "MinTemp_C": "same-interval thermal measurement; direct target leakage",
    "AvgTemp_C": "same-interval thermal measurement; direct target leakage",
    "TR_Probability": "likely thermal-risk-derived score",
    "BMS_Status": "post-measurement risk/status label",
    "EventFlag": "post-measurement event/risk label",
    "Notes": "free text that includes thermal-event language",
    "Timestamp": "used for chronological order, not as a raw feature",
    "ChargerID": "identifier likely to encourage memorization",
    "CellID": "identifier likely to encourage memorization",
    "DemandVoltage_V": "near-duplicate of PackVoltage_V (correlation 0.995)",
    "DemandCurrent_A": "near-duplicate of ChargeCurrent_A (correlation 0.997); preserve raw data for review",
    "ChargePower_kW": "derived/redundant with voltage and current",
}


def load_and_validate_data(data_path: str | Path) -> pd.DataFrame:
    """Load the CSV, parse time, and apply only non-destructive quality checks."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN, TIMESTAMP_COLUMN])
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data[TIMESTAMP_COLUMN] = pd.to_datetime(data[TIMESTAMP_COLUMN], errors="raise")
    # scikit-learn's categorical imputer requires object-compatible values in
    # this environment; this changes no values or source-file data.
    data[CATEGORICAL_FEATURES] = data[CATEGORICAL_FEATURES].astype("object")
    if not data[TIMESTAMP_COLUMN].is_monotonic_increasing:
        raise ValueError("Timestamp values must be monotonically increasing for a chronological split.")
    if data.duplicated().any():
        raise ValueError("Duplicate rows require an explicit review before preprocessing.")

    # Keep the five negative demand-current readings untouched. They are not a
    # feature in the conservative first model and require domain confirmation.
    return data


def chronological_split(data: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return an ordered 80/20 split without shuffling or future-data leakage."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    split_index = int(len(data) * train_fraction)
    if split_index == 0 or split_index == len(data):
        raise ValueError("Dataset is too small for the requested chronological split.")
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Build an unfitted transformer; fit it later on training data only.

    Scaling is appropriate for a later linear-regression pipeline. A later
    tree-model pipeline should call this function with ``scale_numeric=False``.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def create_eda_figures(data: pd.DataFrame, output_directory: str | Path) -> list[Path]:
    """Create read-only EDA figures relevant to temperature prediction."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    figure_paths: list[Path] = []

    fig, axis = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=data, x=TARGET_COLUMN, bins=24, kde=True, color="#0f766e", ax=axis)
    axis.set_title("Distribution of Maximum Battery Temperature")
    axis.set_xlabel("MaxTemp_C")
    figure_paths.append(output_path / "target_distribution.png")
    fig.tight_layout()
    fig.savefig(figure_paths[-1], dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(data[TIMESTAMP_COLUMN], data[TARGET_COLUMN], color="#b45309", linewidth=1.2)
    axis.set_title("Maximum Battery Temperature Over Time")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("MaxTemp_C")
    fig.autofmt_xdate()
    figure_paths.append(output_path / "temperature_over_time.png")
    fig.tight_layout()
    fig.savefig(figure_paths[-1], dpi=160)
    plt.close(fig)

    correlation_columns = NUMERIC_FEATURES + [TARGET_COLUMN]
    fig, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(data[correlation_columns].corr(), cmap="vlag", center=0, annot=True, fmt=".2f", ax=axis)
    axis.set_title("Correlation: Conservative Numeric Features and Target")
    figure_paths.append(output_path / "candidate_feature_correlation.png")
    fig.tight_layout()
    fig.savefig(figure_paths[-1], dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=data, x="ChargingStage", y=TARGET_COLUMN, hue="MoistureDetected", ax=axis)
    axis.set_title("Temperature by Charging Stage and Moisture State")
    axis.set_xlabel("ChargingStage")
    axis.set_ylabel("MaxTemp_C")
    axis.tick_params(axis="x", rotation=15)
    figure_paths.append(output_path / "temperature_by_stage_and_moisture.png")
    fig.tight_layout()
    fig.savefig(figure_paths[-1], dpi=160)
    plt.close(fig)

    return figure_paths
