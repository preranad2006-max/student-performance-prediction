"""
Data Preprocessing Module for Student Performance Prediction System.
Handles data cleaning, missing value imputation, duplicate removal, feature scaling,
and stratified train-test splitting.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

FEATURE_COLUMNS = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignments",
    "Internal_Marks",
]

TARGET_COLUMN = "Final_Result"


class StudentDataPreprocessor:
    """
    Encapsulates preprocessing operations for the Student Performance Prediction system.
    Supports both training transformation and inference transformation.
    """

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        self.feature_columns = FEATURE_COLUMNS
        self.target_column = TARGET_COLUMN
        self.is_fitted = False
        self.summary_: Dict[str, Any] = {}

    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans data: removes duplicates, imputes missing values using median.
        """
        df_clean = df.copy()
        initial_rows = len(df_clean)

        # 1. Deduplication
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_rows - len(df_clean)

        # 2. Imputation for features
        imputed_counts = {}
        for col in self.feature_columns:
            if col in df_clean.columns:
                null_count = int(df_clean[col].isnull().sum())
                if null_count > 0:
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    imputed_counts[col] = null_count

        # 3. Target cleaning if present
        if self.target_column in df_clean.columns:
            # Drop rows with null target if any
            target_nulls = int(df_clean[self.target_column].isnull().sum())
            if target_nulls > 0:
                df_clean = df_clean.dropna(subset=[self.target_column])
                imputed_counts[self.target_column] = target_nulls

        summary = {
            "initial_rows": initial_rows,
            "final_rows": len(df_clean),
            "duplicates_removed": duplicates_removed,
            "imputed_features": imputed_counts,
        }
        return df_clean, summary

    def fit_transform(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fits scaler on training data, transforms features, and splits into 80/20 train/test.
        
        Returns:
            X_train, X_test, y_train, y_test, preprocessing_summary
        """
        df_clean, clean_summary = self.clean_data(df)

        if self.target_column not in df_clean.columns:
            raise ValueError(
                f"Target column '{self.target_column}' is required for training."
            )

        X_raw = df_clean[self.feature_columns].values
        # Encode Target: Pass -> 1, Fail -> 0
        y_raw = df_clean[self.target_column].apply(
            lambda x: 1 if str(x).strip().lower() == "pass" else 0
        ).values

        # Stratified 80/20 train-test split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw,
            y_raw,
            test_size=test_size,
            random_state=random_state,
            stratify=y_raw,
        )

        # Fit scaler on training set only to prevent data leakage
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)
        self.is_fitted = True

        pass_count = int(np.sum(y_raw == 1))
        fail_count = int(np.sum(y_raw == 0))

        self.summary_ = {
            **clean_summary,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "total_samples": int(len(df_clean)),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_ratio": round(pass_count / len(y_raw), 3),
            "feature_names": self.feature_columns,
            "scaler_type": self.scaler_type,
        }

        return X_train, X_test, y_train, y_test, self.summary_

    def transform_single(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """
        Transforms a single student feature dictionary for live inference.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming features.")

        values = [float(feature_dict[col]) for col in self.feature_columns]
        arr = np.array(values).reshape(1, -1)
        return self.scaler.transform(arr)

    def transform_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Cleans and transforms a batch dataframe for prediction.
        """
        df_clean, _ = self.clean_data(df)
        X_raw = df_clean[self.feature_columns].values
        if not self.is_fitted:
            # Fallback fit if scaler not fitted
            X_scaled = self.scaler.fit_transform(X_raw)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X_raw)
        return X_scaled, df_clean
