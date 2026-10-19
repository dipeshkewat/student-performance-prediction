"""Unit tests for data loading, schema validation, and preprocessing pipelines."""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_data, validate_raw_schema, detect_delimiter
from src.preprocessing import (
    ALL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    clean_dataset,
    split_features_target,
    create_preprocessor,
    validate_prediction_input,
)


@pytest.fixture
def raw_data_path() -> Path:
    return Path("data") / "raw" / "student-mat.csv"


@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "G1": [10, 15, 8],
        "G2": [11, 14, 7],
        "studytime": [2, 3, 1],
        "failures": [0, 0, 1],
        "absences": [4, 2, 10],
        "schoolsup": ["no", "no", "yes"],
        "famsup": ["yes", "yes", "no"],
        "activities": ["yes", "no", "yes"],
        "higher": ["yes", "yes", "no"],
        "internet": ["yes", "no", "yes"],
        "G3": [11, 15, 8],
    })


def test_load_data_valid(raw_data_path: Path):
    """Verify that the raw UCI dataset loads successfully with expected record counts."""
    assert raw_data_path.exists(), "Raw dataset file missing"
    df = load_data(raw_data_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 390
    assert "G3" in df.columns
    assert "G1" in df.columns
    assert "G2" in df.columns


def test_load_data_missing_file():
    """Verify FileNotFoundError is raised when an invalid path is supplied."""
    with pytest.raises(FileNotFoundError):
        load_data("data/raw/non_existent_file.csv")


def test_detect_delimiter(tmp_path: Path):
    """Verify delimiter detection for comma and semicolon separated files."""
    semicolon_file = tmp_path / "semi.csv"
    semicolon_file.write_text("col1;col2;col3\n1;2;3\n", encoding="utf-8")
    assert detect_delimiter(semicolon_file) == ";"

    comma_file = tmp_path / "comma.csv"
    comma_file.write_text("col1,col2,col3\n1,2,3\n", encoding="utf-8")
    assert detect_delimiter(comma_file) == ","


def test_validate_raw_schema_detects_missing():
    """Verify schema validator identifies missing required columns."""
    df = pd.DataFrame({"G1": [10], "G2": [12]})
    is_valid, missing = validate_raw_schema(df, required_cols=["G1", "G2", "G3", "absences"])
    assert not is_valid
    assert "G3" in missing
    assert "absences" in missing


def test_clean_dataset_removes_duplicates_and_out_of_range():
    """Verify cleaning rejects duplicate rows and grades outside [0, 20]."""
    dirty_df = pd.DataFrame({
        "G1": [10, 10, 25, -2],  # 25 and -2 are out of range
        "G2": [12, 12, 14, 10],
        "studytime": [2, 2, 2, 2],
        "failures": [0, 0, 0, 0],
        "absences": [4, 4, 2, 0],
        "schoolsup": ["no", "no", "no", "no"],
        "famsup": ["yes", "yes", "yes", "yes"],
        "activities": ["no", "no", "no", "no"],
        "higher": ["yes", "yes", "yes", "yes"],
        "internet": ["yes", "yes", "yes", "yes"],
        "G3": [11, 11, 15, 10],
    })

    cleaned, report = clean_dataset(dirty_df)
    # Row 0 and 1 are duplicates -> 1 removed
    # Row 2 (G1=25) and Row 3 (G1=-2) -> dropped
    assert len(cleaned) == 1
    assert report["duplicate_rows_removed"] == 1
    assert report["out_of_range_dropped"] == 2


def test_split_features_target_prevents_leakage(sample_valid_df: pd.DataFrame):
    """Verify split_features_target guarantees G3 is never present in X."""
    X, y = split_features_target(sample_valid_df, features=ALL_FEATURES, target="G3")
    assert "G3" not in X.columns
    assert list(X.columns) == ALL_FEATURES
    assert len(y) == len(sample_valid_df)
    assert y.name == "G3"

    # Attempting to include G3 in features must raise ValueError
    with pytest.raises(ValueError, match="Data leakage detected"):
        split_features_target(sample_valid_df, features=ALL_FEATURES + ["G3"], target="G3")


def test_create_preprocessor_transforms_data(sample_valid_df: pd.DataFrame):
    """Verify ColumnTransformer transforms both numeric and categorical features."""
    preprocessor = create_preprocessor()
    X = sample_valid_df[ALL_FEATURES]
    transformed = preprocessor.fit_transform(X)

    # 5 numeric features + 5 onehot encoded binary columns = 10 columns
    assert transformed.shape[0] == len(sample_valid_df)
    assert transformed.shape[1] == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    assert not np.isnan(transformed).any()
