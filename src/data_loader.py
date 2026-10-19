"""Data loading and schema validation utilities for Student Performance dataset."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd


DEFAULT_RAW_PATH = Path("data") / "raw" / "student-mat.csv"


def detect_delimiter(filepath: str | Path, num_lines: int = 5) -> str:
    """Detect whether a CSV file uses semicolons, commas, or tabs.

    Args:
        filepath: Path to the CSV file.
        num_lines: Number of header lines to inspect.

    Returns:
        Detected delimiter string (defaulting to ';' if uncertain).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample_lines = [f.readline() for _ in range(num_lines)]

    sample_text = "".join(sample_lines)
    if not sample_text.strip():
        raise ValueError(f"File at {path} is empty.")

    semicolon_count = sample_text.count(";")
    comma_count = sample_text.count(",")
    tab_count = sample_text.count("\t")

    counts = {";": semicolon_count, ",": comma_count, "\t": tab_count}
    best_delim = max(counts, key=counts.get)
    return best_delim if counts[best_delim] > 0 else ";"


def load_data(
    filepath: str | Path = DEFAULT_RAW_PATH,
    sep: Optional[str] = None
) -> pd.DataFrame:
    """Load the student dataset into a Pandas DataFrame with automatic delimiter detection.

    Args:
        filepath: Path to the CSV file.
        sep: Optional custom delimiter. If None, auto-detected.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If file is empty or cannot be parsed.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at path: {path.resolve()}")

    delimiter = sep or detect_delimiter(path)
    try:
        df = pd.read_csv(path, sep=delimiter)
    except Exception as e:
        raise ValueError(f"Failed to read CSV at {path}: {str(e)}") from e

    if df.empty:
        raise ValueError(f"Loaded DataFrame from {path} contains no records.")

    return df


def validate_raw_schema(
    df: pd.DataFrame,
    required_cols: List[str]
) -> Tuple[bool, List[str]]:
    """Validate that all required columns exist in the DataFrame.

    Args:
        df: Input DataFrame.
        required_cols: List of expected column names.

    Returns:
        Tuple of (is_valid: bool, missing_columns: list[str]).
    """
    if df is None or df.empty:
        return False, required_cols

    existing_cols = set(df.columns)
    missing = [col for col in required_cols if col not in existing_cols]
    return (len(missing) == 0), missing
