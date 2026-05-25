import pandas as pd
import pytest

from heart_disease.data import REQUIRED_FEATURES, coerce_prediction_frame, split_features_target


def test_split_features_target_returns_expected_columns() -> None:
    df = pd.DataFrame([{feature: 1 for feature in REQUIRED_FEATURES}])
    df["target"] = 0

    features, target = split_features_target(df, "target")

    assert list(features.columns) == REQUIRED_FEATURES
    assert target.tolist() == [0]


def test_coerce_prediction_frame_rejects_missing_columns() -> None:
    df = pd.DataFrame({"age": [55]})

    with pytest.raises(ValueError, match="missing required feature"):
        coerce_prediction_frame(df)
