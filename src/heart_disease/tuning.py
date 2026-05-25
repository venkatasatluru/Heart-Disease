from __future__ import annotations

import argparse
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

from heart_disease.artifacts import create_artifact, save_artifact
from heart_disease.config import load_training_config
from heart_disease.data import load_dataset, split_features_target
from heart_disease.logging_utils import configure_logging
from heart_disease.metrics import classification_metrics, write_metrics
from heart_disease.pipeline import build_model_pipeline


logger = logging.getLogger(__name__)


def tune_from_config(config_path: str) -> dict[str, float]:
    config = load_training_config(config_path)
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

    pipeline = build_model_pipeline(random_state=config.random_state)
    param_grid = [
        {
            "classifier": [LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state)],
            "classifier__C": [0.1, 1.0, 10.0],
        },
        {
            "classifier": [RandomForestClassifier(class_weight="balanced", random_state=config.random_state)],
            "classifier__n_estimators": [100, 300],
            "classifier__max_depth": [None, 4, 8],
        },
    ]

    logger.info("Running grid search")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(x_train, y_train)

    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(x_test)
    y_probability = best_pipeline.predict_proba(x_test)[:, 1] if hasattr(best_pipeline, "predict_proba") else None
    metrics = classification_metrics(y_test, y_pred, y_probability)
    metrics["best_cv_f1"] = float(grid_search.best_score_)

    write_metrics(metrics, config.metrics_output_path)
    artifact = create_artifact(best_pipeline, target_column=config.target_column, metrics=metrics)
    save_artifact(artifact, config.model_output_path)

    logger.info("Best parameters: %s", grid_search.best_params_)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the heart disease classifier.")
    parser.add_argument("--config", default="configs/training.yaml", help="Path to YAML training config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = tune_from_config(args.config)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
