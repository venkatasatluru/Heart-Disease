from pathlib import Path

from heart_disease.config import TrainingConfig
from heart_disease.inference import predict_record
from heart_disease.training import train_from_config


def test_train_and_predict_round_trip(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    config = TrainingConfig(
        data_path=Path("data/raw/heart_sample.csv"),
        target_column="target",
        test_size=0.25,
        random_state=42,
        model_output_path=model_path,
        metrics_output_path=metrics_path,
        log_level="WARNING",
    )

    metrics = train_from_config(config)
    result = predict_record(
        model_path,
        {
            "age": 54,
            "sex": 1,
            "cp": 2,
            "trestbps": 130,
            "chol": 246,
            "fbs": 0,
            "restecg": 1,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 2,
        },
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert "f1" in metrics
    assert result["prediction"] in {0, 1}
    assert 0.0 <= result["heart_disease_probability"] <= 1.0
