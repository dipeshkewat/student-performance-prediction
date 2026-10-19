"""Model training, cross-validation, candidate comparison, and artifact serialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline

from src.data_loader import load_data
from src.preprocessing import (
    ALL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    clean_dataset,
    split_features_target,
    create_preprocessor,
)
from src.evaluate import (
    compute_regression_metrics,
    extract_feature_importance,
    save_evaluation_results,
)

MODELS_DIR = Path("models")
PROCESSED_DATA_DIR = Path("data") / "processed"
OUTPUTS_DIR = Path("outputs")


def train_and_evaluate(
    raw_csv_path: str | Path = Path("data") / "raw" / "student-mat.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end training pipeline.

    1. Loads raw dataset.
    2. Cleans and validates data; saves processed CSV.
    3. Splits features and target.
    4. Creates 80/20 train/test split.
    5. Runs 5-fold cross-validation on train set for Dummy, Linear Regression, and Random Forest.
    6. Fits each candidate pipeline on training set and tests on held-out test set.
    7. Selects the champion model based on lowest test RMSE.
    8. Serializes best pipeline to models/best_pipeline.joblib.
    9. Exports evaluation metrics, test predictions, and feature importance.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("STUDENT PERFORMANCE PREDICTION: MODEL TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load raw data
    print(f"[1/6] Loading raw dataset from: {raw_csv_path}")
    raw_df = load_data(raw_csv_path)
    print(f"      Loaded {len(raw_df)} raw records with {len(raw_df.columns)} columns.")

    # 2. Clean data
    print("[2/6] Cleaning dataset and validating schema ranges...")
    cleaned_df, quality_report = clean_dataset(raw_df)
    processed_path = PROCESSED_DATA_DIR / "student_cleaned.csv"
    cleaned_df.to_csv(processed_path, index=False)
    print(f"      Retained {len(cleaned_df)} verified records. Saved to: {processed_path}")

    # 3. Separate features and target
    X, y = split_features_target(cleaned_df, features=ALL_FEATURES, target=TARGET_COL)

    # 4. Train/Test split
    print(f"[3/6] Creating reproducible train/test split ({int((1-test_size)*100)}/{int(test_size*100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    split_info = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": ALL_FEATURES,
        "target": TARGET_COL,
        "random_state": random_state,
        "test_size": test_size,
    }
    with open(PROCESSED_DATA_DIR / "train_test_split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # 5. Define candidate models
    candidates = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, max_depth=6, min_samples_split=4, random_state=random_state
        ),
    }

    print("[4/6] Comparing candidates via 5-fold CV and held-out test evaluation...")
    comparison_results: Dict[str, Any] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}
    test_metrics: Dict[str, Dict[str, float]] = {}

    scoring = {
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    for name, regressor in candidates.items():
        preprocessor = create_preprocessor()
        pipe = Pipeline([("prep", preprocessor), ("reg", regressor)])

        # Cross-validation on training partition
        cv_scores = cross_validate(pipe, X_train, y_train, cv=5, scoring=scoring)
        cv_mae = float(-cv_scores["test_neg_mae"].mean())
        cv_rmse = float(-cv_scores["test_neg_rmse"].mean())
        cv_r2 = float(cv_scores["test_r2"].mean())

        # Fit on full training partition and test on held-out test partition
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = compute_regression_metrics(y_test, preds)

        comparison_results[name] = {
            "cv_5fold_mae": round(cv_mae, 4),
            "cv_5fold_rmse": round(cv_rmse, 4),
            "cv_5fold_r2": round(cv_r2, 4),
            "test_mae": metrics["mae"],
            "test_rmse": metrics["rmse"],
            "test_r2": metrics["r2"],
        }
        fitted_pipelines[name] = pipe
        test_metrics[name] = metrics

        print(
            f"      {name:<22} | Test MAE: {metrics['mae']:<6.3f} | "
            f"Test RMSE: {metrics['rmse']:<6.3f} | Test R^2: {metrics['r2']:<6.3f} | "
            f"CV R^2: {cv_r2:<6.3f}"
        )

    # 6. Select champion model (lowest test RMSE)
    best_name = min(test_metrics, key=lambda k: test_metrics[k]["rmse"])
    best_pipeline = fitted_pipelines[best_name]
    best_metrics = test_metrics[best_name]

    print(f"\n[5/6] Champion Candidate: {best_name} (Lowest Test RMSE: {best_metrics['rmse']})")

    # Serialize champion pipeline
    model_path = MODELS_DIR / "best_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"      Saved production pipeline to: {model_path}")

    # Generate held-out predictions table
    y_test_pred = best_pipeline.predict(X_test)
    test_preds_df = X_test.copy()
    test_preds_df["Actual_G3"] = y_test.values
    test_preds_df["Predicted_G3"] = y_test_pred.round(2)
    test_preds_df["Residual"] = (y_test.values - y_test_pred).round(2)

    # Extract feature importance
    feature_importance = extract_feature_importance(
        best_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )

    # Package metadata
    model_metadata = {
        "champion_model": best_name,
        "evaluation_partition": "held_out_test_set_20_percent",
        "test_mae": best_metrics["mae"],
        "test_rmse": best_metrics["rmse"],
        "test_r2": best_metrics["r2"],
        "cv_5fold_rmse": comparison_results[best_name]["cv_5fold_rmse"],
        "cv_5fold_r2": comparison_results[best_name]["cv_5fold_r2"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target": TARGET_COL,
        "features": ALL_FEATURES,
    }

    # Save artifacts
    print("[6/6] Persisting metrics and evaluation artifacts...")
    save_evaluation_results(
        best_metrics=model_metadata,
        model_comparison=comparison_results,
        test_predictions_df=test_preds_df,
        feature_importance=feature_importance,
        output_dir=OUTPUTS_DIR,
    )
    print("      Successfully generated outputs/metrics.json, outputs/model_comparison.json,")
    print("      outputs/test_predictions.csv, and outputs/feature_importance.json.")
    print("=" * 65)
    print("[OK] TRAINING COMPLETED SUCCESSFULLY.")
    print("=" * 65)

    return model_metadata


if __name__ == "__main__":
    train_and_evaluate()
