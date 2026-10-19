"""Unit and integration tests for model inference, schema contracts, and batch operations."""

from __future__ import annotations

import io
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.predict import (
    load_production_model,
    predict_single,
    predict_batch,
    classify_performance_tier,
)
from src.preprocessing import ALL_FEATURES
from src.evaluate import load_metrics, load_model_comparison, load_test_predictions


@pytest.fixture
def valid_student_dict() -> dict:
    return {
        "G1": 13,
        "G2": 14,
        "studytime": 3,
        "failures": 0,
        "absences": 4,
        "schoolsup": "no",
        "famsup": "yes",
        "activities": "yes",
        "higher": "yes",
        "internet": "yes",
    }


def test_production_model_loads_successfully():
    """Verify production pipeline is serializable and loadable."""
    pipeline = load_production_model("models/best_pipeline.joblib")
    assert pipeline is not None
    assert hasattr(pipeline, "predict")


def test_predict_single_returns_valid_grade(valid_student_dict: dict):
    """Verify single prediction output is within [0, 20] and contains diagnostic fields."""
    result = predict_single(valid_student_dict)
    assert result["success"] is True
    pred_grade = result["predicted_grade"]
    assert isinstance(pred_grade, float)
    assert 0.0 <= pred_grade <= 20.0
    assert "performance_band" in result
    assert "guidance" in result
    assert "disclaimer" in result


def test_predict_single_detects_out_of_range(valid_student_dict: dict):
    """Verify input validation flags impossible grades or study times."""
    invalid_student = valid_student_dict.copy()
    invalid_student["G1"] = 28  # Above 20 max
    result = predict_single(invalid_student)
    assert result["success"] is False
    assert any("G1" in err for err in result["errors"])


def test_predict_single_detects_invalid_category(valid_student_dict: dict):
    """Verify categorical validator flags unsupported choices."""
    invalid_student = valid_student_dict.copy()
    invalid_student["schoolsup"] = "maybe"  # Allowed: yes, no
    result = predict_single(invalid_student)
    assert result["success"] is False
    assert any("schoolsup" in err for err in result["errors"])


def test_predict_batch_csv_flow(valid_student_dict: dict):
    """Verify batch prediction processes DataFrames and exports valid CSV data."""
    student_a = valid_student_dict.copy()
    student_b = valid_student_dict.copy()
    student_b.update({"G1": 5, "G2": 6, "failures": 2, "absences": 18})

    df = pd.DataFrame([student_a, student_b])
    results_df, errors = predict_batch(df)

    assert errors == []
    assert results_df is not None
    assert len(results_df) == 2
    assert "Predicted_G3" in results_df.columns
    assert "Performance_Band" in results_df.columns
    assert results_df["Predicted_G3"].between(0.0, 20.0).all()

    # Verify CSV export and reload
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    reloaded_df = pd.read_csv(csv_buffer)
    assert len(reloaded_df) == 2
    assert "Predicted_G3" in reloaded_df.columns


def test_performance_tier_classification():
    """Verify boundary grade classifications."""
    assert "Excellent" in classify_performance_tier(18.5)[0]
    assert "Good" in classify_performance_tier(14.5)[0]
    assert "Satisfactory" in classify_performance_tier(11.0)[0]
    assert "Warning" in classify_performance_tier(7.5)[0]


def test_missing_model_raises_clear_error():
    """Verify requesting a nonexistent model yields a clear, informative error."""
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        load_production_model("models/non_existent_pipeline.joblib")


def test_evaluation_loaders_fallback_gracefully():
    """Verify loaders return None instead of crashing on nonexistent files."""
    assert load_metrics("outputs/non_existent_metrics.json") is None
    assert load_model_comparison("outputs/non_existent_comp.json") is None
    assert load_test_predictions("outputs/non_existent_preds.csv") is None
