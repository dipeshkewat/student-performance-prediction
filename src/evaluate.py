"""Evaluation metrics computation, artifact persistence, and result loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

DEFAULT_OUTPUT_DIR = Path("outputs")


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate core regression evaluation metrics: MAE, RMSE, R-squared.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted values.

    Returns:
        Dictionary containing MAE, RMSE, and R2.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def extract_feature_importance(
    pipeline: Any,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[Dict[str, Any]]:
    """Extract feature importance or regression coefficients from the fitted pipeline.

    Args:
        pipeline: Fitted Scikit-learn Pipeline with ('prep', ColumnTransformer) and ('reg', Model).
        numeric_features: List of numeric feature names.
        categorical_features: List of categorical feature names.

    Returns:
        List of dicts with feature name and importance/weight, sorted by magnitude descending.
    """
    if "reg" not in pipeline.named_steps or "prep" not in pipeline.named_steps:
        return []

    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["reg"]

    # Construct transformed feature names
    cat_step = preprocessor.named_transformers_["cat"]
    onehot = cat_step.named_steps["onehot"]
    cat_feature_names = list(onehot.get_feature_names_out(categorical_features))
    transformed_features = numeric_features + cat_feature_names

    importance_list: List[Dict[str, Any]] = []

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for name, imp in zip(transformed_features, importances):
            importance_list.append({
                "feature": name,
                "importance": round(float(imp), 4),
                "type": "Gini Importance"
            })
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        for name, coef in zip(transformed_features, coefs):
            importance_list.append({
                "feature": name,
                "importance": round(float(abs(coef)), 4),
                "coefficient": round(float(coef), 4),
                "type": "Absolute Coefficient"
            })

    importance_list.sort(key=lambda x: x["importance"], reverse=True)
    return importance_list


def save_evaluation_results(
    best_metrics: Dict[str, Any],
    model_comparison: Dict[str, Any],
    test_predictions_df: pd.DataFrame,
    feature_importance: List[Dict[str, Any]],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Save all evaluation artifacts as JSON and CSV files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    with open(out_path / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(model_comparison, f, indent=2)

    with open(out_path / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(feature_importance, f, indent=2)

    test_predictions_df.to_csv(out_path / "test_predictions.csv", index=False)


def load_metrics(filepath: str | Path = DEFAULT_OUTPUT_DIR / "metrics.json") -> Optional[Dict[str, Any]]:
    """Safely load saved metrics JSON file."""
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_model_comparison(filepath: str | Path = DEFAULT_OUTPUT_DIR / "model_comparison.json") -> Optional[Dict[str, Any]]:
    """Safely load saved model comparison JSON file."""
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_feature_importance(filepath: str | Path = DEFAULT_OUTPUT_DIR / "feature_importance.json") -> Optional[List[Dict[str, Any]]]:
    """Safely load saved feature importance JSON file."""
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_test_predictions(filepath: str | Path = DEFAULT_OUTPUT_DIR / "test_predictions.csv") -> Optional[pd.DataFrame]:
    """Safely load test set predictions CSV."""
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None
