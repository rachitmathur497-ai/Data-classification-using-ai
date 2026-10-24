"""
DecodeLabs AI Project 2: Data Classification Using AI
Train and evaluate a supervised machine-learning classifier on the Iris dataset.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "iris.csv"
OUTPUT_DIR = ROOT / "outputs"
TARGET = "species"
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load and understand the dataset
    data = pd.read_csv(DATA_PATH)
    print("\\n=== Dataset preview ===")
    print(data.head())
    print("\\n=== Dataset information ===")
    print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
    print("\\nClass counts:")
    print(data[TARGET].value_counts())
    print("\\nMissing values:")
    print(data.isnull().sum())

    if data.isnull().any().any():
        raise ValueError("Dataset contains missing values; clean it before training.")

    X = data[FEATURES]
    y = data[TARGET]

    # 2. Split into training and testing sets (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Train a simple classification algorithm
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_train, y_train)

    # 4. Evaluate on held-out test data
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    print("\\n=== Model evaluation ===")
    print(f"Test accuracy: {accuracy:.3f} ({accuracy:.1%})")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "Iris",
            "algorithm": "StandardScaler + LogisticRegression",
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "accuracy": float(accuracy),
            "classification_report": report,
        }, f, indent=2)

    # Save confusion matrix image
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=model.classes_, cmap="Blues"
    )
    plt.title("Iris Classifier — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # 5. Predict a sample flower (measurements in cm)
    sample = pd.DataFrame([{
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2,
    }], columns=FEATURES)
    predicted_species = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    print("\\n=== Example prediction ===")
    print(f"Measurements: {sample.iloc[0].to_dict()}")
    print(f"Predicted species: {predicted_species}")
    print("Class probabilities:")
    for name, probability in zip(model.classes_, probabilities):
        print(f"  {name}: {probability:.3f}")

    with open(OUTPUT_DIR / "sample_prediction.json", "w", encoding="utf-8") as f:
        json.dump({
            "measurements_cm": sample.iloc[0].to_dict(),
            "predicted_species": str(predicted_species),
            "probabilities": {
                str(name): float(probability)
                for name, probability in zip(model.classes_, probabilities)
            },
        }, f, indent=2)

    print(f"\\nSaved evaluation files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
