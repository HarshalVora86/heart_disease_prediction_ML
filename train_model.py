"""
Train and export the complete preprocessing + classification pipeline
for the Heart Disease Prediction project.

Run:
    python train_model.py

Output:
    Model/heart_disease_pipeline.pkl
    Model/metrics.json
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from custom_transformers import IQRCapper

CONTINUOUS_FEATURES = ["trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["fbs", "restecg", "exang", "slope"]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope"]
TARGET = "target"


def main():
    df = pd.read_csv("dataset/heart_disease_combined.csv")

    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan

    df = df.drop(columns=["ca", "thal"])
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = (df["num"] > 0).astype(int)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    continuous_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("capper", IQRCapper()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipeline, CONTINUOUS_FEATURES),
            ("categorical", SimpleImputer(strategy="most_frequent"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    feature_info = {}
    for col in FEATURES:
        if col in CATEGORICAL_FEATURES + ["sex", "cp"]:
            default = float(X[col].mode()[0])
        else:
            default = float(X[col].median())
        feature_info[col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "default": default,
        }

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "sklearn_version": sklearn.__version__,
        "feature_info": feature_info,
    }

    os.makedirs("Model", exist_ok=True)

    joblib.dump(pipeline, "Model/heart_disease_pipeline.pkl")

    with open("Model/metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel trained successfully.")
    print("Pipeline: Model/heart_disease_pipeline.pkl")
    print("\nEvaluation:")
    for key in ["Accuracy", "Precision", "Recall", "F1", "train_rows", "test_rows"]:
        print(f"{key}: {metrics[key]}")
    print(f"\nscikit-learn version used: {sklearn.__version__}")
    print(f"Add this line to requirements.txt:  scikit-learn=={sklearn.__version__}")


if __name__ == "__main__":
    main()
