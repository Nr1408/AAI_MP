"""Development-only sanity checks for the supplied EV thermal CSV.

These are diagnostic calculations, not safety thresholds or predictive models.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_preprocessing import TARGET_COLUMN, TIMESTAMP_COLUMN

TEMPERATURE_COLUMNS = ["MaxTemp_C", "MinTemp_C", "AvgTemp_C", "AmbientTemp_C"]
TEMPERATURE_RELATIONSHIP_COLUMNS = [
    "ChargeCurrent_A",
    "PackVoltage_V",
    "CellVoltage_V",
    "SOC_%",
    "AmbientTemp_C",
    "InternalResistance_mOhm",
    "StateOfHealth_%",
    "ChargePower_kW",
    "Pressure_kPa",
    "VibrationLevel_mg",
]


def temperature_delta_diagnostics(data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return signed one-minute MaxTemp differences and diagnostic-bin counts."""
    delta = data[TARGET_COLUMN].diff().dropna()
    absolute_delta = delta.abs()
    bins = pd.cut(
        absolute_delta,
        bins=[-np.inf, 1, 2, 5, 10, np.inf],
        right=False,
        labels=["<1 C", "1-2 C", "2-5 C", "5-10 C", ">10 C"],
    )
    counts = bins.value_counts(sort=False).rename("count").to_frame()
    counts["percent"] = (counts["count"] / len(delta) * 100).round(2)
    return delta, counts


def thermal_consistency_diagnostics(data: pd.DataFrame) -> dict[str, object]:
    """Check ordering, spread, and whether average temperature is a midpoint."""
    ordering_valid = (data["MinTemp_C"] <= data["AvgTemp_C"]) & (data["AvgTemp_C"] <= data["MaxTemp_C"])
    spread = data["MaxTemp_C"] - data["MinTemp_C"]
    midpoint_error = data["AvgTemp_C"] - (data["MaxTemp_C"] + data["MinTemp_C"]) / 2
    return {
        "ordering_valid_count": int(ordering_valid.sum()),
        "ordering_invalid_count": int((~ordering_valid).sum()),
        "spread": spread,
        "midpoint_error": midpoint_error,
        "maximum_absolute_midpoint_error": float(midpoint_error.abs().max()),
    }


def electrical_consistency_diagnostics(data: pd.DataFrame) -> dict[str, pd.Series | float]:
    """Check arithmetic consistency without assuming undocumented pack topology."""
    calculated_power = data["PackVoltage_V"] * data["ChargeCurrent_A"] / 1000
    power_error = data["ChargePower_kW"] - calculated_power
    relative_power_error = np.where(
        calculated_power.abs() > 0,
        power_error.abs() / calculated_power.abs() * 100,
        np.nan,
    )
    return {
        "calculated_power_kW": calculated_power,
        "power_error_kW": power_error,
        "relative_power_error_percent": pd.Series(relative_power_error, index=data.index),
        "pack_cell_ratio": data["PackVoltage_V"] / data["CellVoltage_V"],
        "demand_voltage_delta": data["DemandVoltage_V"] - data["PackVoltage_V"],
        "demand_current_delta": data["DemandCurrent_A"] - data["ChargeCurrent_A"],
        "power_correlation": float(calculated_power.corr(data["ChargePower_kW"])),
    }


def consecutive_row_diagnostics(data: pd.DataFrame) -> dict[str, object]:
    """Measure whether consecutive rows resemble a continuous single entity."""
    numeric_columns = data.select_dtypes(include=np.number).columns
    transitions = {
        column: int(data[column].ne(data[column].shift()).iloc[1:].sum())
        for column in ["ChargerID", "CellID", "ChargingStage", "MoistureDetected", "BMS_Status", "EventFlag"]
    }
    return {
        "transitions": transitions,
        "unique_charger_cell_pairs": int(data[["ChargerID", "CellID"]].drop_duplicates().shape[0]),
        "lag_one_correlations": data[numeric_columns].corrwith(data[numeric_columns].shift(1)),
    }


def create_sanity_figures(data: pd.DataFrame, output_directory: str | Path) -> list[Path]:
    """Create the four required development-only provenance diagnostic figures."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    paths: list[Path] = []
    thermal = thermal_consistency_diagnostics(data)
    electrical = electrical_consistency_diagnostics(data)
    delta, _ = temperature_delta_diagnostics(data)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(data[TIMESTAMP_COLUMN], data["MaxTemp_C"], label="MaxTemp_C", color="#b45309", linewidth=1)
    axes[0, 0].plot(data[TIMESTAMP_COLUMN], data["AvgTemp_C"], label="AvgTemp_C", color="#0f766e", linewidth=1)
    axes[0, 0].plot(data[TIMESTAMP_COLUMN], data["MinTemp_C"], label="MinTemp_C", color="#2563eb", linewidth=1)
    axes[0, 0].set_title("Development Temperature Fields Over Time")
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 1].scatter(data["MinTemp_C"], data["MaxTemp_C"], alpha=0.6, color="#0f766e")
    limits = [data["MinTemp_C"].min(), data["MaxTemp_C"].max()]
    axes[0, 1].plot(limits, limits, "--", color="#374151")
    axes[0, 1].set_title("MinTemp_C vs MaxTemp_C")
    axes[0, 1].set_xlabel("MinTemp_C")
    axes[0, 1].set_ylabel("MaxTemp_C")
    sns.histplot(thermal["spread"], bins=20, color="#7c3aed", ax=axes[1, 0])
    axes[1, 0].set_title("Thermal Spread: MaxTemp_C - MinTemp_C")
    axes[1, 0].set_xlabel("Spread (C)")
    sns.histplot(thermal["midpoint_error"], bins=20, color="#b45309", ax=axes[1, 1])
    axes[1, 1].set_title("AvgTemp_C Minus Midpoint of Min/Max")
    axes[1, 1].set_xlabel("Difference (C)")
    paths.append(output_path / "thermal_consistency.png")
    figure.tight_layout()
    figure.savefig(paths[-1], dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(data[TIMESTAMP_COLUMN].iloc[1:], delta, color="#7c3aed", linewidth=0.9)
    axes[0].axhline(0, color="#374151", linewidth=1)
    axes[0].set_title("One-Minute Change in MaxTemp_C")
    axes[0].set_xlabel("Timestamp")
    axes[0].set_ylabel("Delta MaxTemp_C (C)")
    axes[0].tick_params(axis="x", rotation=25)
    sns.histplot(delta, bins=32, color="#0f766e", ax=axes[1])
    axes[1].axvline(0, color="#374151", linewidth=1)
    axes[1].set_title("Distribution of One-Minute Temperature Changes")
    axes[1].set_xlabel("Delta MaxTemp_C (C)")
    paths.append(output_path / "temperature_deltas.png")
    figure.tight_layout()
    figure.savefig(paths[-1], dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].scatter(electrical["calculated_power_kW"], data["ChargePower_kW"], alpha=0.65, color="#0f766e")
    power_limits = [0, max(electrical["calculated_power_kW"].max(), data["ChargePower_kW"].max())]
    axes[0, 0].plot(power_limits, power_limits, "--", color="#b45309")
    axes[0, 0].set_title("Reported vs V x I Derived Charge Power")
    axes[0, 0].set_xlabel("PackVoltage_V x ChargeCurrent_A / 1000")
    axes[0, 0].set_ylabel("ChargePower_kW")
    sns.histplot(electrical["power_error_kW"], bins=24, color="#7c3aed", ax=axes[0, 1])
    axes[0, 1].set_title("Reported Minus Derived Charge Power")
    axes[0, 1].set_xlabel("Difference (kW)")
    sns.histplot(electrical["pack_cell_ratio"], bins=24, color="#2563eb", ax=axes[1, 0])
    axes[1, 0].set_title("PackVoltage_V / CellVoltage_V Ratio")
    axes[1, 0].set_xlabel("Ratio")
    axes[1, 1].scatter(data["ChargeCurrent_A"], data["DemandCurrent_A"], alpha=0.65, color="#b45309")
    current_limits = [min(data["ChargeCurrent_A"].min(), data["DemandCurrent_A"].min()), max(data["ChargeCurrent_A"].max(), data["DemandCurrent_A"].max())]
    axes[1, 1].plot(current_limits, current_limits, "--", color="#374151")
    axes[1, 1].set_title("Measured vs Demand Current")
    axes[1, 1].set_xlabel("ChargeCurrent_A")
    axes[1, 1].set_ylabel("DemandCurrent_A")
    paths.append(output_path / "electrical_consistency.png")
    figure.tight_layout()
    figure.savefig(paths[-1], dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(3, 4, figsize=(15, 10))
    for axis, feature in zip(axes.flat, TEMPERATURE_RELATIONSHIP_COLUMNS):
        sns.regplot(
            data=data,
            x=feature,
            y=TARGET_COLUMN,
            scatter_kws={"alpha": 0.45, "s": 16, "color": "#0f766e"},
            line_kws={"color": "#b45309"},
            ax=axis,
        )
        axis.set_title(f"MaxTemp_C vs {feature}", fontsize=9)
    sns.boxplot(data=data, x="ChargingStage", y=TARGET_COLUMN, color="#c4b5fd", ax=axes.flat[10])
    axes.flat[10].set_title("MaxTemp_C by Charging Stage", fontsize=9)
    axes.flat[10].tick_params(axis="x", rotation=20, labelsize=7)
    sns.boxplot(data=data, x="MoistureDetected", y=TARGET_COLUMN, color="#99f6e4", ax=axes.flat[11])
    axes.flat[11].set_title("MaxTemp_C by Moisture State", fontsize=9)
    paths.append(output_path / "temperature_relationships.png")
    figure.tight_layout()
    figure.savefig(paths[-1], dpi=160)
    plt.close(figure)
    return paths
