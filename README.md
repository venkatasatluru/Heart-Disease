# Heart Disease Classification & Inference System

An end-to-end Python project for heart disease prediction using scikit-learn. It covers data loading, preprocessing, model training, hyperparameter tuning, evaluation, model persistence, command-line inference, and a browser-based Streamlit interface for safe stakeholder exploration.

## Project Structure

```text
.
├── configs/                 # Training configuration
├── data/raw/                # Input CSV data
├── models/                  # Persisted model artifacts
├── orchestration/           # Airflow integration example
├── reports/                 # Metrics and evaluation outputs
├── scripts/                 # Developer-friendly CLI wrappers
├── src/heart_disease/       # Reusable package code
└── tests/                   # Unit tests
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
heart-disease train --config configs/training.yaml
heart-disease predict --model models/heart_disease_pipeline.joblib --input data/raw/heart_sample.csv
heart-disease app
```

You can also run the project directly from a checkout before installing the package:

```bash
./bin/heart-disease train --config configs/training.yaml
./bin/heart-disease predict --model models/heart_disease_pipeline.joblib --input data/raw/heart_sample.csv
make app
```

The legacy developer wrappers remain available too:

```bash
python3 scripts/train.py --config configs/training.yaml
python3 scripts/predict.py --model models/heart_disease_pipeline.joblib --input data/raw/heart_sample.csv
streamlit run src/heart_disease/web_app.py
```

The included `data/raw/heart_sample.csv` is intentionally small and only meant to prove the pipeline works. Replace it with a real dataset before drawing conclusions.

## Expected Dataset

The default configuration expects the common UCI-style heart disease schema:

```text
age, sex, cp, trestbps, chol, fbs, restecg, thalach,
exang, oldpeak, slope, ca, thal, target
```

`target` should be binary, where `1` indicates heart disease risk and `0` indicates no detected disease.

## Train

```bash
heart-disease train --config configs/training.yaml
```

This creates:

- `models/heart_disease_pipeline.joblib`
- `reports/metrics.json`

## Tune Hyperparameters

```bash
heart-disease tune --config configs/training.yaml
```

The tuning script performs cross-validated grid search over logistic regression and random forest classifiers, then writes the best persisted pipeline.

## Predict

Batch prediction:

```bash
heart-disease predict \
  --model models/heart_disease_pipeline.joblib \
  --input data/raw/heart_sample.csv \
  --output reports/predictions.csv
```

Single-patient prediction:

```bash
heart-disease predict \
  --model models/heart_disease_pipeline.joblib \
  --record '{"age": 54, "sex": 1, "cp": 2, "trestbps": 130, "chol": 246, "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2}'
```

## Browser Interface

```bash
heart-disease app
```

The app loads the persisted model artifact, lets users enter patient feature values, and returns a class prediction plus probability. It is designed for exploration, not clinical decision-making.

## Production Integration

`orchestration/airflow_dag_example.py` shows how the training and evaluation steps can be scheduled from Airflow or adapted to another orchestrator.

## Important Disclaimer

This project is for software engineering and machine learning workflow demonstration only. It is not a medical device, diagnostic tool, or substitute for clinician judgment.
