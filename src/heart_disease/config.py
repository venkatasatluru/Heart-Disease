from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    data_path: Path
    target_column: str
    test_size: float
    random_state: int
    model_output_path: Path
    metrics_output_path: Path
    log_level: str = "INFO"


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file) or {}

    return TrainingConfig(
        data_path=Path(raw_config["data_path"]),
        target_column=str(raw_config.get("target_column", "target")),
        test_size=float(raw_config.get("test_size", 0.2)),
        random_state=int(raw_config.get("random_state", 42)),
        model_output_path=Path(raw_config.get("model_output_path", "models/heart_disease_pipeline.joblib")),
        metrics_output_path=Path(raw_config.get("metrics_output_path", "reports/metrics.json")),
        log_level=str(raw_config.get("log_level", "INFO")),
    )
