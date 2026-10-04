"""
Heart Disease Prediction App
Streamlit app for the Machine Learning (3170724) CIPAT project.

Loads the pre-trained pipeline (Model/heart_disease_pipeline.pkl), produced by
running train_model.py, and uses it directly for predictions.

Run with:  streamlit run app.py
"""

import json

import joblib
import pandas as pd
import sklearn
import streamlit as st

from custom_transformers import IQRCapper  # noqa: F401  (needed so joblib can load the pipeline)

st.set_page_config(page_title="Heart Disease Predictor",
                   page_icon="❤️", layout="centered")

PIPELINE_PATHS = [
    "Model/heart_disease_pipeline.pkl",
    "model/heart_disease_pipeline.pkl",
    "../Model/heart_disease_pipeline.pkl",
]
METRICS_PATHS = [
    "Model/metrics.json",
    "model/metrics.json",
    "../Model/metrics.json",
]

FEATURE_ORDER = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                 "restecg", "thalach", "exang", "oldpeak", "slope"]

FEATURE_CONFIG = {
    "age": {"label": "Age (years)", "kind": "int"},
    "sex": {"label": "Sex", "kind": "select",
            "options": {1: "Male", 0: "Female"}},
    "cp": {"label": "Chest Pain Type", "kind": "select",
           "options": {1: "Typical angina", 2: "Atypical angina",
                       3: "Non-anginal pain", 4: "Asymptomatic"}},
    "trestbps": {"label": "Resting Blood Pressure (mm Hg)", "kind": "int"},
    "chol": {"label": "Serum Cholesterol (mg/dl)", "kind": "int"},
    "fbs": {"label": "Fasting Blood Sugar > 120 mg/dl", "kind": "select",
            "options": {0: "No", 1: "Yes"}},
    "restecg": {"label": "Resting ECG Result", "kind": "select",
                "options": {0: "Normal", 1: "ST-T wave abnormality",
                            2: "Left ventricular hypertrophy"}},
    "thalach": {"label": "Maximum Heart Rate Achieved (bpm)", "kind": "int"},
    "exang": {"label": "Exercise-Induced Angina", "kind": "select",
              "options": {0: "No", 1: "Yes"}},
    "oldpeak": {"label": "ST Depression Induced by Exercise", "kind": "float"},
    "slope": {"label": "Slope of Peak Exercise ST Segment", "kind": "select",
              "options": {1: "Upsloping", 2: "Flat", 3: "Downsloping"}},
}


@st.cache_resource
def load_pipeline():
    for path in PIPELINE_PATHS:
        try:
            return joblib.load(path)
        except FileNotFoundError:
            continue
    return None


@st.cache_data
def load_metrics():
    for path in METRICS_PATHS:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            continue
    return None


def render_input(col, name, info):
    cfg = FEATURE_CONFIG[name]
    lo, hi, default = info["min"], info["max"], info["default"]

    if cfg["kind"] == "select":
        keys = list(cfg["options"].keys())
        value = col.selectbox(
            cfg["label"],
            options=keys,
            index=keys.index(int(default)),
            format_func=lambda k, opts=cfg["options"]: f"{k} - {opts[k]}",
            key=name,
        )
        col.caption(f"Range: {min(keys)} – {max(keys)}")
    elif cfg["kind"] == "int":
        value = col.number_input(
            cfg["label"], min_value=int(lo), max_value=int(hi),
            value=int(round(default)), step=1, key=name)
        col.caption(f"Range: {int(lo)} – {int(hi)}")
    else:
        value = col.number_input(
            cfg["label"], min_value=round(lo, 1), max_value=round(hi, 1),
            value=round(default, 1), step=0.1, key=name)
        col.caption(f"Range: {round(lo, 1)} – {round(hi, 1)}")
    return float(value)


st.title("❤️ Heart Disease Predictor")
st.caption("Machine Learning CIPAT Project")

pipeline = load_pipeline()
metrics = load_metrics()

if pipeline is None or metrics is None:
    st.error(
        "Model files not found. Run **train_model.py** first to generate "
        "`Model/heart_disease_pipeline.pkl` and `Model/metrics.json`, then reload this app."
    )
    st.stop()

if metrics.get("sklearn_version") != sklearn.__version__:
    st.warning(
        f"The model was trained with scikit-learn {metrics.get('sklearn_version')} but this app "
        f"is running {sklearn.__version__}. Pin `scikit-learn=={metrics.get('sklearn_version')}` "
        "in requirements.txt."
    )

st.info("Educational project. This is not a medical diagnosis.")

st.divider()
st.subheader("Enter Patient Details")

c1, c2 = st.columns(2)
values = {}
for i, name in enumerate(FEATURE_ORDER):
    column = c1 if i % 2 == 0 else c2
    values[name] = render_input(column, name, metrics["feature_info"][name])

predict_btn = st.button("Predict", type="primary", use_container_width=True)

if predict_btn:
    input_df = pd.DataFrame([values])[FEATURE_ORDER]

    prediction = int(pipeline.predict(input_df)[0])

    if prediction == 1:
        st.error("### Heart disease is likely present")
        st.caption("Please consult a doctor for proper tests.")
    else:
        st.success("### No heart disease predicted")

    st.caption(
        f"Estimate based on SVM (RBF kernel). Test accuracy of the model: "
        f"{metrics['Accuracy'] * 100:.1f}%."
    )

st.divider()
with st.expander("About this model"):
    details = [
        "**Model:** Support Vector Machine, RBF kernel",
        f"**Features used:** {', '.join(FEATURE_ORDER)}",
        "**Preprocessing:** missing values filled (median for numeric, most frequent for "
        "categorical), outliers capped with the IQR rule, and all features standardized, "
        "all inside one scikit-learn `Pipeline` so the same steps run on every prediction.",
        f"**Performance (held-out test set, {metrics['test_rows']} patients):** "
        f"Accuracy = {metrics['Accuracy']:.3f}, Precision = {metrics['Precision']:.3f}, "
        f"Recall = {metrics['Recall']:.3f}, F1 = {metrics['F1']:.3f}",
        "**Data:** UCI Heart Disease dataset (Cleveland, Hungarian, Switzerland, VA Long Beach), "
        "collected in the late 1980s.",
    ]
    st.markdown("\n".join(f"- {d}" for d in details))
