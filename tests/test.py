"""
Unit tests for IRIS ML Pipeline
--------------------------------
This script validates:
1. Data structure and quality
2. Model performance
"""

import os
import pytest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

DATA_PATH = "data/iris.csv"
MODEL_PATH = "model/model.pkl"
TEST_DATA_PATH = "data/test.csv"


def test_data_file_exists():
    assert os.path.exists(DATA_PATH), f"{DATA_PATH} not found."


def test_data_validity():
    df = pd.read_csv(DATA_PATH)
    expected_columns = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert set(df.columns) == expected_columns, "Columns mismatch."
    assert not df.isnull().values.any(), "Dataset contains missing values."


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"{MODEL_PATH} not found."


def test_model_accuracy():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not trained yet. Run train.py first.")
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=["species"])
    y_test = test_df["species"]
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test data: {acc:.2f}")
    assert acc > 0.7, f"Model accuracy too low: {acc}"


def test_model_prediction_shape():
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=["species"])
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(X_test), "Prediction length mismatch."
