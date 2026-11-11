"""Logistic Regression Titanic survival assignment."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "titanic" / "train.csv"
OUTPUT_DIR = ROOT / "outputs"


def build_model(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)),
    ])


def main():
    parser = argparse.ArgumentParser(description="Train Logistic Regression on Titanic data.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to Titanic train.csv")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}\n"
            "Download Kaggle Titanic train.csv and place it in data/titanic/."
        )

    data = pd.read_csv(args.data)
    if "Survived" not in data.columns:
        raise ValueError("Expected target column 'Survived'. Use the Kaggle Titanic training CSV.")

    y = data["Survived"]
    X = data.drop(columns=["Survived"])
    for identifier in ("PassengerId",):
        if identifier in X.columns:
            X = X.drop(columns=[identifier])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = build_model(X_train)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probabilities)
    except ValueError:
        auc = float("nan")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_text = (
        "TITANIC SURVIVAL PREDICTION — LOGISTIC REGRESSION\n"
        f"Rows in dataset: {len(data)}\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1-score: {f1:.4f}\n"
        f"ROC-AUC: {auc:.4f}\n\n"
        "Classification report:\n"
        f"{classification_report(y_test, predictions, zero_division=0)}"
    )
    (OUTPUT_DIR / "titanic_metrics.txt").write_text(metrics_text, encoding="utf-8")

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Did not survive", "Survived"],
                yticklabels=["Did not survive", "Survived"])
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Titanic Survival: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "titanic_confusion_matrix.png", dpi=160)
    plt.close()

    joblib.dump(model, OUTPUT_DIR / "titanic_logistic_regression.joblib")
    print(metrics_text)
    print(f"Confusion matrix and trained model saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
