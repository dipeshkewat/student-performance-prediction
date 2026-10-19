"""Prediction service supporting single-student inference and batch CSV processing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import validate_prediction_input, ALL_FEATURES

DEFAULT_MODEL_PATH = Path("models") / "best_pipeline.joblib"


def load_production_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load the serialized champion model pipeline from disk.

    Raises:
        FileNotFoundError: If the model pipeline has not been trained yet.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{path.resolve()}'. "
            "Please run 'python -m src.train' first to train and persist the champion model pipeline."
        )
    return joblib.load(path)


def classify_performance_tier(grade: float) -> Tuple[str, str, str]:
    """Classify 0-20 grade into an academic performance band, color badge, and guidance."""
    g = float(np.clip(grade, 0.0, 20.0))
    if g >= 16.0:
        return (
            "Excellent (16-20)",
            "success",
            "Demonstrates outstanding academic mastery. Well-positioned for academic honors.",
        )
    elif g >= 14.0:
        return (
            "Good (14-15.9)",
            "info",
            "Demonstrates solid comprehension and steady academic progress. Minor review suggested.",
        )
    elif g >= 10.0:
        return (
            "Satisfactory (10-13.9)",
            "warning",
            "Meets passing threshold. Targeted tutoring or study-time reinforcement is recommended.",
        )
    else:
        return (
            "Academic Warning (<10)",
            "danger",
            "Currently falling below passing grade. Immediate academic intervention and advisory recommended.",
        )


def analyze_risk_factors(data: Dict[str, Any]) -> List[str]:
    """Identify behavioral or past academic risk indicators for proactive guidance."""
    factors: List[str] = []
    g1 = float(data.get("G1", 10))
    g2 = float(data.get("G2", 10))
    failures = int(data.get("failures", 0))
    absences = int(data.get("absences", 0))
    studytime = int(data.get("studytime", 2))

    if g2 < 10.0:
        factors.append(f"Period 2 grade ({g2:.1f}) is currently below passing threshold.")
    elif g2 < g1:
        factors.append(f"Performance declined from Period 1 ({g1:.1f}) to Period 2 ({g2:.1f}).")

    if failures > 0:
        factors.append(f"Student has a history of {failures} previous course failure(s).")

    if absences > 10:
        factors.append(f"Elevated absence rate ({absences} days missed) may impede curriculum continuity.")

    if studytime == 1:
        factors.append("Low reported weekly study time (< 2 hours/week).")

    return factors


def predict_single(
    student_data: Dict[str, Any],
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """Execute single-student grade prediction with validation and diagnostic feedback.

    Args:
        student_data: Dictionary containing all required features.
        model_path: Path to serialized Scikit-learn Pipeline.

    Returns:
        Dictionary with predicted grade, performance tier, risk factors, and advisory note.
    """
    validated_df, errors = validate_prediction_input(student_data)
    if errors:
        return {
            "success": False,
            "errors": errors,
            "predicted_grade": None,
        }

    pipeline = load_production_model(model_path)
    raw_pred = float(pipeline.predict(validated_df)[0])
    clipped_pred = round(float(np.clip(raw_pred, 0.0, 20.0)), 2)

    band, badge, guidance = classify_performance_tier(clipped_pred)
    risks = analyze_risk_factors(student_data)

    return {
        "success": True,
        "predicted_grade": clipped_pred,
        "performance_band": band,
        "badge": badge,
        "guidance": guidance,
        "risk_factors": risks,
        "disclaimer": (
            "Educational Analytics Notice: This prediction is a probabilistic estimate derived from "
            "historical data patterns. It is intended strictly as an early-warning diagnostic advisory "
            "and does not represent a fixed or guaranteed outcome."
        ),
    }


def predict_batch(
    df: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Execute batch grade prediction on a DataFrame.

    Args:
        df: Input DataFrame with required feature columns.
        model_path: Path to serialized Scikit-learn Pipeline.

    Returns:
        Tuple of (results_df, list_of_validation_errors).
    """
    validated_df, errors = validate_prediction_input(df)
    if errors:
        return None, errors

    pipeline = load_production_model(model_path)
    raw_preds = pipeline.predict(validated_df)
    clipped_preds = np.clip(raw_preds, 0.0, 20.0).round(2)

    result_df = df.copy()
    result_df["Predicted_G3"] = clipped_preds
    tiers = [classify_performance_tier(p)[0] for p in clipped_preds]
    result_df["Performance_Band"] = tiers

    return result_df, []
