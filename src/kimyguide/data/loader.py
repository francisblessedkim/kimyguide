"""Data loading helpers for the KimyGuide prototype.

This module contains a tiny convenience wrapper around reading a CSV
of course-level metadata and building a combined `text` field used by
the feature pipeline.
"""

import pandas as pd
from pathlib import Path

# Base directory = project root (two levels above `src/kimyguide`)
BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def load_courses(
    # csv_name: str = "courses.csv",
    csv_name: str = "courses_mooclike.csv",
    id_col: str = "course_id",
    title_col: str = "title",
    desc_col: str = "description",
) -> pd.DataFrame:
    """Load course-level data and prepare a `text` column.

    The returned DataFrame will contain at minimum the following
    columns:
      - item_id: standardized id column (renamed from `id_col`)
      - title: the title column
      - description: the description column
      - text: title + description (used by TF-IDF features)

    Raises FileNotFoundError when the expected CSV is missing and
    KeyError when required columns are absent. Callers can override
    `csv_name` and column names to match a different input file.
    """
    path = RAW_DATA_DIR / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Place your CSV in data/raw/.")

    # Read CSV using pandas
    df = pd.read_csv(path)

    # Check expected columns exist; fail early with clear message
    for col in [id_col, title_col, desc_col]:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in {csv_name}, found {df.columns.tolist()}")

    # Keep only the fields we need and drop rows with missing description
    df = df[[id_col, title_col, desc_col]].copy()
    df = df.dropna(subset=[desc_col])

    # Build a combined text field used by the feature extractor
    df["text"] = df[title_col].fillna("") + " " + df[desc_col].fillna("")

    # Standardise id column name for downstream code
    df = df.rename(columns={id_col: "item_id"})
    return df
