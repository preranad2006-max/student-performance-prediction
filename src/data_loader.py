"""
Data Input and Generation Module for Student Performance Prediction System.
Handles dataset loading, validation, and realistic synthetic benchmark data generation.
"""

import os
from io import BytesIO
from typing import Tuple, Dict, Any, Union
import numpy as np
import pandas as pd

REQUIRED_FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignments",
    "Internal_Marks",
]

TARGET_COLUMN = "Final_Result"


def generate_sample_dataset(
    n_samples: int = 1000, random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic, statistically grounded synthetic student academic dataset.
    
    Features:
        - Student_ID: Unique identifier (e.g. STU1001)
        - Study_Hours: Daily study hours (1.0 to 10.0 hours)
        - Attendance: Attendance percentage (45.0% to 100.0%)
        - Previous_Marks: Previous exam score (20.0 to 100.0)
        - Assignments: Assignment average score (20.0 to 100.0)
        - Internal_Marks: Internal assessment score (20.0 to 100.0)
        - Final_Result: Pass / Fail binary target
        - Grade: Academic grade classification (A+, A, B, C, Fail)
    """
    np.random.seed(random_state)

    student_ids = [f"STU{1000 + i + 1}" for i in range(n_samples)]

    # Generate correlated base academic capability factor ~ N(60, 15)
    base_ability = np.random.normal(loc=60, scale=15, size=n_samples)
    base_ability = np.clip(base_ability, 20, 95)

    # Study Hours: 1.0 to 9.5 hours, positively correlated with base ability
    study_hours = np.clip(
        0.08 * base_ability + np.random.normal(1.2, 1.0, size=n_samples),
        1.0,
        10.0,
    ).round(1)

    # Attendance: 45% to 100%, positively correlated with base ability
    attendance = np.clip(
        0.55 * base_ability + np.random.normal(35, 9, size=n_samples),
        45.0,
        100.0,
    ).round(1)

    # Previous Marks: 20 to 100
    prev_marks = np.clip(
        base_ability + np.random.normal(0, 7, size=n_samples),
        20.0,
        100.0,
    ).round(1)

    # Assignments: 20 to 100
    assignments = np.clip(
        0.7 * base_ability + 0.3 * (study_hours * 10) + np.random.normal(5, 8, size=n_samples),
        20.0,
        100.0,
    ).round(1)

    # Internal Marks: 20 to 100
    internal_marks = np.clip(
        0.6 * base_ability + 0.25 * attendance + np.random.normal(5, 7, size=n_samples),
        20.0,
        100.0,
    ).round(1)

    # Academic Composite Performance Calculation
    # Weights: Previous 25%, Internals 25%, Assignments 20%, Attendance 15%, Study Hours 15%
    composite_score = (
        0.25 * prev_marks
        + 0.25 * internal_marks
        + 0.20 * assignments
        + 0.15 * attendance
        + 0.15 * (study_hours * 10)
        + np.random.normal(0, 4, size=n_samples)
    )
    composite_score = np.clip(composite_score, 15, 100).round(1)

    # Final Result: Pass if composite >= 50.0 else Fail
    # Standard passing threshold is 50%
    final_result = np.where(composite_score >= 50.0, "Pass", "Fail")

    # Grade Classification
    def assign_grade(score: float) -> str:
        if score >= 85:
            return "A+"
        elif score >= 75:
            return "A"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C"
        else:
            return "Fail"

    grades = [assign_grade(s) for s in composite_score]

    df = pd.DataFrame(
        {
            "Student_ID": student_ids,
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "Previous_Marks": prev_marks,
            "Assignments": assignments,
            "Internal_Marks": internal_marks,
            "Final_Result": final_result,
            "Grade": grades,
        }
    )

    return df


def save_default_datasets(data_dir: str = "data") -> Tuple[str, str]:
    """Generates and saves standard CSV and Excel benchmark datasets."""
    os.makedirs(data_dir, exist_ok=True)
    df = generate_sample_dataset(n_samples=1000, random_state=42)

    csv_path = os.path.join(data_dir, "student_data.csv")
    xlsx_path = os.path.join(data_dir, "student_data.xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    return csv_path, xlsx_path


def load_dataset(
    file_or_path: Union[str, BytesIO], filename: str = ""
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads and validates a CSV or Excel dataset.
    
    Returns:
        df: pd.DataFrame
        validation_report: dict containing validation status and diagnostics
    """
    report: Dict[str, Any] = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "shape": (0, 0),
        "missing_values": {},
        "target_present": False,
    }

    try:
        if isinstance(file_or_path, str):
            if file_or_path.endswith(".csv"):
                df = pd.read_csv(file_or_path)
            elif file_or_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_or_path)
            else:
                try:
                    df = pd.read_csv(file_or_path)
                except Exception:
                    df = pd.read_excel(file_or_path)
        else:
            # File buffer / BytesIO
            if filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_or_path)
            else:
                df = pd.read_csv(file_or_path)

        report["shape"] = df.shape

        # Normalize column names: strip whitespace
        df.columns = [str(c).strip() for c in df.columns]

        # Check required features
        missing_features = [f for f in REQUIRED_FEATURES if f not in df.columns]
        if missing_features:
            report["errors"].append(
                f"Missing required feature columns: {missing_features}. Required are: {REQUIRED_FEATURES}"
            )
            return df, report

        # Check target column
        if TARGET_COLUMN in df.columns:
            report["target_present"] = True
        else:
            report["warnings"].append(
                f"Target column '{TARGET_COLUMN}' not found. Dataset can be used for batch prediction but not for model training."
            )

        # Missing values check
        missing_counts = df.isnull().sum().to_dict()
        report["missing_values"] = {k: v for k, v in missing_counts.items() if v > 0}

        report["valid"] = True
        return df, report

    except Exception as e:
        report["errors"].append(f"Failed to read file: {str(e)}")
        return pd.DataFrame(), report
