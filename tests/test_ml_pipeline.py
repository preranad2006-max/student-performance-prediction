"""
Automated Test Suite for Student Performance Prediction System.
Tests data generation, loading, preprocessing, model training, evaluation, and risk assessment.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.data_loader import (
    generate_sample_dataset,
    load_dataset,
    REQUIRED_FEATURES,
    TARGET_COLUMN,
)
from src.preprocessor import StudentDataPreprocessor
from src.models import (
    initialize_models,
    train_all_models,
    save_artifacts,
    load_artifacts,
)
from src.evaluation import compare_models, evaluate_single_model
from src.risk_analyzer import assess_student_risk, analyze_cohort_risk


def test_generate_sample_dataset():
    """Verify synthetic dataset generation schema, ranges, and target balance."""
    df = generate_sample_dataset(n_samples=250, random_state=42)
    assert len(df) == 250
    for feat in REQUIRED_FEATURES:
        assert feat in df.columns
    assert TARGET_COLUMN in df.columns
    assert "Grade" in df.columns
    assert set(df["Final_Result"].unique()).issubset({"Pass", "Fail"})
    assert (df["Attendance"] >= 40.0).all() and (df["Attendance"] <= 100.0).all()
    assert (df["Study_Hours"] >= 0.5).all() and (df["Study_Hours"] <= 12.0).all()


def test_data_loader_validation(tmp_path):
    """Verify data loader checks required columns and formats."""
    df = generate_sample_dataset(n_samples=100, random_state=42)
    csv_file = tmp_path / "test_data.csv"
    df.to_csv(csv_file, index=False)

    loaded_df, report = load_dataset(str(csv_file))
    assert report["valid"] is True
    assert len(loaded_df) == 100
    assert report["target_present"] is True

    # Test invalid data (missing required feature)
    bad_df = df.drop(columns=["Study_Hours"])
    bad_file = tmp_path / "bad_data.csv"
    bad_df.to_csv(bad_file, index=False)
    _, bad_report = load_dataset(str(bad_file))
    assert bad_report["valid"] is False
    assert len(bad_report["errors"]) > 0


def test_preprocessor_pipeline():
    """Verify preprocessing, scaling, and 80/20 train-test split."""
    df = generate_sample_dataset(n_samples=200, random_state=42)
    preprocessor = StudentDataPreprocessor(scaler_type="standard")
    X_train, X_test, y_train, y_test, summary = preprocessor.fit_transform(
        df, test_size=0.20, random_state=42
    )

    assert len(X_train) == 160
    assert len(X_test) == 40
    assert X_train.shape[1] == len(REQUIRED_FEATURES)
    assert summary["train_samples"] == 160
    assert summary["test_samples"] == 40

    # Test single transformation for live inference
    sample_input = {
        "Study_Hours": 5.0,
        "Attendance": 80.0,
        "Previous_Marks": 70.0,
        "Assignments": 75.0,
        "Internal_Marks": 72.0,
    }
    scaled_single = preprocessor.transform_single(sample_input)
    assert scaled_single.shape == (1, len(REQUIRED_FEATURES))


def test_all_5_models_training_and_evaluation():
    """Verify all 5 ML models train properly and output required metrics."""
    df = generate_sample_dataset(n_samples=200, random_state=42)
    preprocessor = StudentDataPreprocessor()
    X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(df)

    trained_results = train_all_models(X_train, y_train, X_test)
    assert len(trained_results) == 5

    expected_models = {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine (SVM)",
        "Naive Bayes",
    }
    assert set(trained_results.keys()) == expected_models

    comp_df, best_model_name, detailed_evals = compare_models(trained_results, y_test)
    assert len(comp_df) == 5
    assert best_model_name in expected_models

    # Metrics check
    for m in expected_models:
        eval_m = detailed_evals[m]
        assert 0.0 <= eval_m["accuracy"] <= 1.0
        assert 0.0 <= eval_m["f1_score"] <= 1.0
        assert len(eval_m["confusion_matrix"]) == 2


def test_risk_assessment():
    """Verify individual student risk categorization and recommendations."""
    # Test High Risk student
    high_risk = assess_student_risk(
        prob_pass=0.20,
        study_hours=1.5,
        attendance=50.0,
        previous_marks=35.0,
        assignments=40.0,
        internal_marks=42.0,
    )
    assert high_risk["risk_level"] == "High Risk"
    assert high_risk["priority"] == 1
    assert len(high_risk["recommendations"]) > 0

    # Test Low Risk student
    low_risk = assess_student_risk(
        prob_pass=0.95,
        study_hours=8.0,
        attendance=95.0,
        previous_marks=90.0,
        assignments=92.0,
        internal_marks=94.0,
    )
    assert low_risk["risk_level"] == "Low Risk"
    assert low_risk["priority"] == 3


def test_model_serialization(tmp_path):
    """Verify artifact serialization and reloading."""
    df = generate_sample_dataset(n_samples=100, random_state=42)
    preprocessor = StudentDataPreprocessor()
    X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(df)

    trained_results = train_all_models(X_train, y_train, X_test)
    best_clf = trained_results["Logistic Regression"]["model"]

    save_path = save_artifacts(
        best_clf,
        preprocessor,
        "Logistic Regression",
        output_dir=str(tmp_path),
        metadata={"version": "1.0"},
    )
    assert os.path.exists(save_path)

    loaded = load_artifacts(save_path)
    assert loaded["model_name"] == "Logistic Regression"
    assert loaded["preprocessor"].is_fitted is True
