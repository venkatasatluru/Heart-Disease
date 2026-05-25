from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from heart_disease.data import REQUIRED_FEATURES


@dataclass(frozen=True)
class ModelArtifact:
    pipeline: Any
    feature_columns: list[str]
    target_column: str
    created_at: str
    metrics: dict[str, float] | None = None


def create_artifact(pipeline: Any, target_column: str, metrics: dict[str, float] | None = None) -> ModelArtifact:
    return ModelArtifact(
        pipeline=pipeline,
        feature_columns=REQUIRED_FEATURES,
        target_column=target_column,
        created_at=datetime.now(UTC).isoformat(),
        metrics=metrics,
    )


def save_artifact(artifact: ModelArtifact, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)


def load_artifact(path: str | Path) -> ModelArtifact:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, ModelArtifact):
        raise TypeError("Loaded object is not a ModelArtifact.")
    return artifact
