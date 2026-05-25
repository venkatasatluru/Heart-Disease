from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="heart_disease_training_pipeline",
    description="Train and evaluate the heart disease classifier.",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "heart-disease"],
) as dag:
    install_project = BashOperator(
        task_id="install_project",
        bash_command="pip install -e .",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="python scripts/train.py --config configs/training.yaml",
    )

    install_project >> train_model
