
"""Linear Regression house-price assignment and mini project."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project folders
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "house_prices" / "Housing.csv"
OUTPUT_DIR = ROOT / "outputs"


def build_model(X: pd.DataFrame) -> Pipeline:
    """Build preprocessing and Linear Regression pipeline."""

    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

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

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", LinearRegression()),
    ])

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a Linear Regression house-price model."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to Kaggle Housing.csv",
    )

    args = parser.parse_args()

    # Check dataset
    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}\n"
            "Download Housing.csv from the Kaggle Housing Prices Dataset "
            "and place it in data/house_prices/."
        )

    # Load dataset
    data = pd.read_csv(args.data)

    # Check target column
    target_column = "price"

    if target_column not in data.columns:
        raise ValueError(
            f"Expected target column '{target_column}'.\n"
            f"Available columns: {data.columns.tolist()}"
        )

    # Separate input features and target
    y = data[target_column]
    X = data.drop(columns=[target_column])

    # Remove ID column if present
    if "Id" in X.columns:
        X = X.drop(columns=["Id"])

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # Train Linear Regression model
    model = build_model(X_train)
    model.fit(X_train, y_train)

    # Predict house prices
    predictions = model.predict(X_test)

    # Evaluate model
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Create output folder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save evaluation results
    metrics_text = (
        "HOUSE PRICE PREDICTION - LINEAR REGRESSION\n"
        "==========================================\n"
        f"Dataset: {args.data.name}\n"
        f"Target column: {target_column}\n"
        f"Total dataset rows: {len(data)}\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n\n"
        f"R2 Score: {r2:.4f}\n"
        f"Mean Absolute Error (MAE): {mae:,.2f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:,.2f}\n"
    )

    metrics_file = OUTPUT_DIR / "house_price_metrics.txt"
    metrics_file.write_text(metrics_text, encoding="utf-8")

    # Plot actual vs predicted prices
    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.65,
        edgecolors="none",
    )

    low = min(
        float(y_test.min()),
        float(np.min(predictions)),
    )

    high = max(
        float(y_test.max()),
        float(np.max(predictions)),
    )

    plt.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=2,
        label="Perfect prediction",
    )

    plt.xlabel("Actual House Price")
    plt.ylabel("Predicted House Price")
    plt.title("House Prices: Actual vs Predicted")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plot_file = OUTPUT_DIR / "house_price_actual_vs_predicted.png"
    plt.savefig(plot_file, dpi=160)
    plt.close()

    # Save trained model
    model_file = OUTPUT_DIR / "house_price_linear_regression.joblib"
    joblib.dump(model, model_file)

    # Display results
    print(metrics_text)
    print("Plot saved at:", plot_file)
    print("Trained model saved at:", model_file)


if __name__ == "__main__":
    main()
