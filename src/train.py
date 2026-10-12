"""Chronological baseline and initial ML comparison for EV battery temperature."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data_preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor, chronological_split, load_and_validate_data
from src.evaluate import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_tree_feature_importance,
    regression_metrics,
)

RANDOM_STATE = 42
N_SPLITS = 5


def build_model_pipelines() -> dict[str, Pipeline]:
    """Build fresh, unfitted pipelines for the approved feature set."""
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                ("regressor", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "regressor",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def validate_models(model_pipelines: dict[str, Pipeline], features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Evaluate pipelines on growing time-series folds, without final-test access."""
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    rows = []
    for model_name, pipeline in model_pipelines.items():
        scores = cross_validate(
            pipeline,
            features,
            target,
            cv=splitter,
            scoring=scoring,
            return_train_score=True,
            n_jobs=1,
        )
        rows.append(
            {
                "model": model_name,
                "cv_train_MAE_mean": -scores["train_mae"].mean(),
                "cv_validation_MAE_mean": -scores["test_mae"].mean(),
                "cv_validation_MAE_std": scores["test_mae"].std(),
                "cv_train_RMSE_mean": -scores["train_rmse"].mean(),
                "cv_validation_RMSE_mean": -scores["test_rmse"].mean(),
                "cv_validation_RMSE_std": scores["test_rmse"].std(),
                "cv_train_R2_mean": scores["train_r2"].mean(),
                "cv_validation_R2_mean": scores["test_r2"].mean(),
                "cv_validation_R2_std": scores["test_r2"].std(),
            }
        )
    return pd.DataFrame(rows).sort_values("cv_validation_MAE_mean", ignore_index=True)


def fit_and_test_models(
    model_pipelines: dict[str, Pipeline],
    development_features: pd.DataFrame,
    development_target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> tuple[dict[str, Pipeline], pd.DataFrame, dict[str, pd.Series]]:
    """Fit exactly once on all development data, then score the untouched period."""
    fitted_models: dict[str, Pipeline] = {}
    test_predictions: dict[str, pd.Series] = {}
    rows = []
    for model_name, pipeline in model_pipelines.items():
        fitted_pipeline = pipeline.fit(development_features, development_target)
        predictions = pd.Series(fitted_pipeline.predict(test_features), index=test_target.index)
        metrics = regression_metrics(test_target, predictions)
        fitted_models[model_name] = fitted_pipeline
        test_predictions[model_name] = predictions
        rows.append({"model": model_name, **{f"final_test_{key}": value for key, value in metrics.items()}})
    return fitted_models, pd.DataFrame(rows), test_predictions


def run_experiment(project_root: Path) -> tuple[pd.DataFrame, str]:
    """Run validation, select solely by CV MAE, and create required result artifacts."""
    data_path = project_root / "EV_Battery_Charging_TR_Dataset_with_Notes (1).csv"
    results_directory = project_root / "results"
    figures_directory = results_directory / "figures"
    results_directory.mkdir(exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)

    data = load_and_validate_data(data_path)
    development_data, final_test_data = chronological_split(data, train_fraction=0.8)
    development_features = development_data[FEATURE_COLUMNS]
    development_target = development_data[TARGET_COLUMN]
    test_features = final_test_data[FEATURE_COLUMNS]
    test_target = final_test_data[TARGET_COLUMN]

    pipelines = build_model_pipelines()
    comparison = validate_models(pipelines, development_features, development_target)
    selected_model = comparison.iloc[0]["model"]

    fitted_models, final_test_scores, predictions = fit_and_test_models(
        pipelines, development_features, development_target, test_features, test_target
    )
    comparison = comparison.merge(final_test_scores, on="model", how="left")
    comparison["selected_by_validation"] = comparison["model"].eq(selected_model)
    comparison.to_csv(results_directory / "model_comparison.csv", index=False, float_format="%.6f")

    selected_predictions = predictions[selected_model]
    plot_actual_vs_predicted(
        test_target,
        selected_predictions,
        selected_model,
        figures_directory / "actual_vs_predicted.png",
    )
    plot_residuals(
        test_target,
        selected_predictions,
        final_test_data["Timestamp"],
        selected_model,
        figures_directory / "residuals.png",
    )
    plot_tree_feature_importance(fitted_models, figures_directory / "feature_importance.png")

    return comparison, selected_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chronological EV thermal-model comparison.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project directory containing the source CSV.",
    )
    args = parser.parse_args()
    comparison, selected_model = run_experiment(args.project_root)
    print(f"Selected using validation MAE only: {selected_model}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
