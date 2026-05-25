from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path)


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from dataset.")

    missing_features = [feature for feature in REQUIRED_FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing required feature columns: {missing_features}")

    features = df[REQUIRED_FEATURES].copy()
    target = df[target_column].copy()
    return features, target


def coerce_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing_features = [feature for feature in REQUIRED_FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Input data is missing required feature columns: {missing_features}")
    return df[REQUIRED_FEATURES].copy()
