from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from heart_disease.artifacts import load_artifact
from heart_disease.data import coerce_prediction_frame


def predict_frame(model_path: str | Path, input_frame: pd.DataFrame) -> pd.DataFrame:
    artifact = load_artifact(model_path)
    features = coerce_prediction_frame(input_frame)
    predictions = artifact.pipeline.predict(features)
    probabilities = artifact.pipeline.predict_proba(features)[:, 1]

    output = input_frame.copy()
    output["prediction"] = predictions
    output["heart_disease_probability"] = probabilities
    return output


def predict_record(model_path: str | Path, record: dict) -> dict:
    frame = pd.DataFrame([record])
    prediction = predict_frame(model_path, frame).iloc[0]
    return {
        "prediction": int(prediction["prediction"]),
        "heart_disease_probability": float(prediction["heart_disease_probability"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heart disease inference.")
    parser.add_argument("--model", default="models/heart_disease_pipeline.joblib", help="Path to model artifact.")
    parser.add_argument("--input", help="CSV file with patient records for batch inference.")
    parser.add_argument("--record", help="Single patient record as a JSON object.")
    parser.add_argument("--output", help="Optional path to write batch predictions as CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input and not args.record:
        raise SystemExit("Provide either --input for batch prediction or --record for single-row prediction.")

    if args.record:
        result = predict_record(args.model, json.loads(args.record))
        print(json.dumps(result, indent=2))
        return

    input_frame = pd.read_csv(args.input)
    predictions = predict_frame(args.model, input_frame)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"Wrote predictions to {output_path}")
    else:
        print(predictions.to_csv(index=False))


if __name__ == "__main__":
    main()
