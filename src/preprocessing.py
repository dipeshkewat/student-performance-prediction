"""Preprocessing, feature engineering, and data quality validation pipelines."""

from __future__ import annotations

from typing import Tuple, List, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Core modeling feature contract
NUMERIC_FEATURES: List[str] = ["G1", "G2", "studytime", "failures", "absences"]
CATEGORICAL_FEATURES: List[str] = [
    "schoolsup",
    "famsup",
    "activities",
    "higher",
    "internet",
]
ALL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL: str = "G3"

FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "G1": (0.0, 20.0),
    "G2": (0.0, 20.0),
    "G3": (0.0, 20.0),
    "studytime": (1.0, 4.0),
    "failures": (0.0, 4.0),
    "absences": (0.0, 93.0),
}

ALLOWED_CATEGORIES: Dict[str, List[str]] = {
    "schoolsup": ["yes", "no"],
    "famsup": ["yes", "no"],
    "activities": ["yes", "no"],
    "higher": ["yes", "no"],
    "internet": ["yes", "no"],
}


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean the raw student performance dataset and compile a data quality report.

    Rules:
    - Drops duplicate rows.
    - Ensures numeric fields are converted to appropriate float/int types.
    - Validates grade ranges (0-20), studytime (1-4), failures (>=0), absences (>=0).
    - Normalizes categorical string values (lowercase, stripped).
    - Filters out rows with invalid or corrupted values instead of arbitrary imputation.

    Args:
        df: Raw pandas DataFrame.

    Returns:
        Tuple of (cleaned_df, quality_report).
    """
    report: Dict[str, Any] = {
        "initial_rows": len(df),
        "duplicate_rows_removed": 0,
        "missing_values_detected": {},
        "out_of_range_dropped": 0,
        "final_rows": 0,
        "warnings": [],
    }

    cleaned = df.copy()

    # 1. Detect and drop duplicate rows
    initial_count = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    report["duplicate_rows_removed"] = initial_count - len(cleaned)

    # 2. Check missing values
    missing_counts = cleaned[ALL_FEATURES + ([TARGET_COL] if TARGET_COL in cleaned.columns else [])].isnull().sum()
    report["missing_values_detected"] = missing_counts[missing_counts > 0].to_dict()

    # Drop rows with missing values in critical modeling columns
    cleaned = cleaned.dropna(subset=[col for col in ALL_FEATURES if col in cleaned.columns])

    # 3. Numeric conversions & range validations
    valid_mask = pd.Series(True, index=cleaned.index)

    for col in NUMERIC_FEATURES:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            valid_mask &= cleaned[col].notnull()
            min_val, max_val = FEATURE_RANGES[col]
            valid_mask &= (cleaned[col] >= min_val) & (cleaned[col] <= max_val)

    if TARGET_COL in cleaned.columns:
        cleaned[TARGET_COL] = pd.to_numeric(cleaned[TARGET_COL], errors="coerce")
        valid_mask &= cleaned[TARGET_COL].notnull()
        min_g3, max_g3 = FEATURE_RANGES[TARGET_COL]
        valid_mask &= (cleaned[TARGET_COL] >= min_g3) & (cleaned[TARGET_COL] <= max_g3)

    # 4. Categorical normalizations
    for col in CATEGORICAL_FEATURES:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip().str.lower()
            valid_mask &= cleaned[col].isin(ALLOWED_CATEGORIES[col])

    invalid_count = int((~valid_mask).sum())
    report["out_of_range_dropped"] = invalid_count
    if invalid_count > 0:
        report["warnings"].append(
            f"Filtered out {invalid_count} row(s) exceeding academic value thresholds or categorical rules."
        )

    cleaned = cleaned[valid_mask].reset_index(drop=True)
    report["final_rows"] = len(cleaned)

    return cleaned, report


def split_features_target(
    df: pd.DataFrame,
    features: List[str] = ALL_FEATURES,
    target: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate input features from target variable, strictly enforcing no target leakage.

    Args:
        df: Input DataFrame.
        features: List of feature column names.
        target: Target column name.

    Returns:
        Tuple of (X: DataFrame, y: Series).

    Raises:
        ValueError: If target is in features list or target column is missing from df.
    """
    if target in features:
        raise ValueError(
            f"Data leakage detected! Target column '{target}' cannot be included in features list."
        )

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing from the dataset.")

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    X = df[features].copy()
    y = df[target].copy()
    return X, y


def create_preprocessor() -> ColumnTransformer:
    """Construct a Scikit-learn ColumnTransformer for preprocessing.

    - Numeric features: Median imputation followed by StandardScaler.
    - Categorical features: Most-frequent imputation followed by OneHotEncoder (drop='first', handle_unknown='ignore').

    Returns:
        Configured ColumnTransformer instance.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def validate_prediction_input(
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
) -> Tuple[pd.DataFrame, List[str]]:
    """Validate user input or batch CSV payload against the expected model feature schema.

    Args:
        data: A dict, list of dicts, or pandas DataFrame.

    Returns:
        Tuple of (valid_dataframe, list_of_error_messages).
    """
    errors: List[str] = []

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        return pd.DataFrame(), ["Invalid input data type. Expected dict, list of dicts, or DataFrame."]

    if df.empty:
        return pd.DataFrame(), ["Input dataset contains 0 records."]

    # Check required columns
    missing_cols = [c for c in ALL_FEATURES if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required feature columns: {', '.join(missing_cols)}")
        return pd.DataFrame(), errors

    # Check numeric columns and ranges
    for col in NUMERIC_FEATURES:
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        nan_indices = numeric_series[numeric_series.isna()].index.tolist()
        if nan_indices:
            errors.append(
                f"Column '{col}' contains non-numeric or missing values in row(s): {nan_indices[:5]}"
            )
            continue

        min_val, max_val = FEATURE_RANGES[col]
        out_of_range = (numeric_series < min_val) | (numeric_series > max_val)
        if out_of_range.any():
            bad_indices = df.index[out_of_range].tolist()
            errors.append(
                f"Column '{col}' has values outside valid range [{min_val}, {max_val}] in row(s): {bad_indices[:5]}"
            )

        df[col] = numeric_series

    # Check categorical columns
    for col in CATEGORICAL_FEATURES:
        str_series = df[col].astype(str).str.strip().str.lower()
        allowed = set(ALLOWED_CATEGORIES[col])
        invalid_mask = ~str_series.isin(allowed)
        if invalid_mask.any():
            bad_indices = df.index[invalid_mask].tolist()
            errors.append(
                f"Column '{col}' has invalid categories (must be one of {allowed}) in row(s): {bad_indices[:5]}"
            )
        df[col] = str_series

    # Reorder columns to match strict training feature order
    df = df[ALL_FEATURES]

    return df, errors
