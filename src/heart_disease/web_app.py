from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from heart_disease.artifacts import load_artifact
from heart_disease.data import REQUIRED_FEATURES


DEFAULT_MODEL_PATH = Path("models/heart_disease_pipeline.joblib")


def patient_form() -> dict[str, float | int]:
    st.sidebar.header("Patient Features")
    return {
        "age": st.sidebar.slider("Age", 18, 100, 54),
        "sex": st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda value: "Female" if value == 0 else "Male"),
        "cp": st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3]),
        "trestbps": st.sidebar.slider("Resting Blood Pressure", 80, 220, 130),
        "chol": st.sidebar.slider("Serum Cholesterol", 100, 650, 240),
        "fbs": st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1]),
        "restecg": st.sidebar.selectbox("Resting ECG Result", options=[0, 1, 2]),
        "thalach": st.sidebar.slider("Max Heart Rate", 60, 230, 150),
        "exang": st.sidebar.selectbox("Exercise-Induced Angina", options=[0, 1]),
        "oldpeak": st.sidebar.slider("ST Depression", 0.0, 7.0, 1.0, step=0.1),
        "slope": st.sidebar.selectbox("ST Slope", options=[0, 1, 2]),
        "ca": st.sidebar.selectbox("Major Vessels Colored", options=[0, 1, 2, 3, 4]),
        "thal": st.sidebar.selectbox("Thalassemia", options=[0, 1, 2, 3]),
    }


def render_prediction(record: dict[str, float | int], model_path: Path) -> None:
    artifact = load_artifact(model_path)
    frame = pd.DataFrame([record], columns=REQUIRED_FEATURES)
    prediction = int(artifact.pipeline.predict(frame)[0])
    probability = float(artifact.pipeline.predict_proba(frame)[:, 1][0])

    st.subheader("Prediction")
    st.metric("Predicted class", "Heart disease risk" if prediction == 1 else "No detected risk")
    st.metric("Estimated probability", f"{probability:.1%}")
    st.progress(max(0.0, min(1.0, probability)))

    with st.expander("Model input record"):
        st.dataframe(frame)


def main() -> None:
    st.set_page_config(page_title="Heart Disease Classifier", page_icon="🫀", layout="centered")
    st.title("Heart Disease Classification")
    st.write(
        "Explore a trained scikit-learn model using patient feature values. "
        "This interface is for demonstration and stakeholder exploration only."
    )

    model_path = Path(st.text_input("Model artifact path", value=str(DEFAULT_MODEL_PATH)))
    record = patient_form()

    if not model_path.exists():
        st.warning("Train a model first with `heart-disease train --config configs/training.yaml`.")
        return

    if st.button("Run Prediction", type="primary"):
        render_prediction(record, model_path)

    st.caption("Not a medical device. Do not use for diagnosis or treatment decisions.")


if __name__ == "__main__":
    main()
