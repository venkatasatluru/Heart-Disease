from __future__ import annotations

import argparse
import logging

from sklearn.model_selection import train_test_split

from heart_disease.artifacts import create_artifact, save_artifact
from heart_disease.config import TrainingConfig, load_training_config
from heart_disease.data import load_dataset, split_features_target
from heart_disease.logging_utils import configure_logging
from heart_disease.metrics import classification_metrics, write_metrics
from heart_disease.pipeline import build_model_pipeline


logger = logging.getLogger(__name__)


def train_from_config(config: TrainingConfig) -> dict[str, float]:
    configure_logging(config.log_level)
    logger.info("Loading dataset from %s", config.data_path)
    df = load_dataset(config.data_path)
    features, target = split_features_target(df, config.target_column)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target if target.nunique() > 1 else None,
    )

    logger.info("Training model pipeline")
    pipeline = build_model_pipeline(random_state=config.random_state)
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_probability = pipeline.predict_proba(x_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
    metrics = classification_metrics(y_test, y_pred, y_probability)

    logger.info("Writing metrics to %s", config.metrics_output_path)
    write_metrics(metrics, config.metrics_output_path)

    logger.info("Saving model artifact to %s", config.model_output_path)
    artifact = create_artifact(pipeline, target_column=config.target_column, metrics=metrics)
    save_artifact(artifact, config.model_output_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the heart disease classifier.")
    parser.add_argument("--config", default="configs/training.yaml", help="Path to YAML training config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    metrics = train_from_config(config)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
