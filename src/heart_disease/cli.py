from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heart-disease", description="Heart disease classifier command line app.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and persist the classifier.")
    train_parser.add_argument("--config", default="configs/training.yaml", help="Path to YAML training config.")

    tune_parser = subparsers.add_parser("tune", help="Tune hyperparameters and persist the best classifier.")
    tune_parser.add_argument("--config", default="configs/training.yaml", help="Path to YAML training config.")

    predict_parser = subparsers.add_parser("predict", help="Run batch or single-record inference.")
    predict_parser.add_argument("--model", default="models/heart_disease_pipeline.joblib", help="Path to model artifact.")
    predict_parser.add_argument("--input", help="CSV file with patient records for batch inference.")
    predict_parser.add_argument("--record", help="Single patient record as a JSON object.")
    predict_parser.add_argument("--output", help="Optional path to write batch predictions as CSV.")

    app_parser = subparsers.add_parser("app", help="Launch the Streamlit browser interface.")
    app_parser.add_argument("--server-port", type=int, help="Optional Streamlit server port.")
    app_parser.add_argument("--server-address", help="Optional Streamlit server address.")
    return parser


def _print_metrics(metrics: dict[str, float]) -> None:
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


def _run_train(config_path: str) -> None:
    from heart_disease.config import load_training_config
    from heart_disease.training import train_from_config

    config = load_training_config(config_path)
    _print_metrics(train_from_config(config))


def _run_tune(config_path: str) -> None:
    from heart_disease.tuning import tune_from_config

    _print_metrics(tune_from_config(config_path))


def _run_predict(args: argparse.Namespace) -> None:
    import pandas as pd

    from heart_disease.inference import predict_frame, predict_record

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


def _run_app(args: argparse.Namespace) -> None:
    app_path = Path(__file__).with_name("web_app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--browser.gatherUsageStats=false",
    ]
    if args.server_port is not None:
        command.extend(["--server.port", str(args.server_port)])
    if args.server_address:
        command.extend(["--server.address", args.server_address])
    raise SystemExit(subprocess.call(command))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        _run_train(args.config)
    elif args.command == "tune":
        _run_tune(args.config)
    elif args.command == "predict":
        _run_predict(args)
    elif args.command == "app":
        _run_app(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
