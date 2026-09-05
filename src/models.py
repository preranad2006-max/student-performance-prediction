"""
Machine Learning Module for Student Performance Prediction System.
Implements, trains, compares, and serializes the 5 required Machine Learning algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. Gaussian Naive Bayes
"""

import os
from typing import Dict, Any, Tuple
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine (SVM)",
    "Naive Bayes",
]


def initialize_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Initializes standard classifiers with balanced hyperparameters.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=500, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=5, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=random_state
        ),
        "Support Vector Machine (SVM)": CalibratedClassifierCV(
            SVC(C=1.0, kernel="rbf", random_state=random_state),
            ensemble=False,
        ),
        "Naive Bayes": GaussianNB(),
    }


def train_single_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Trains a single classifier and generates predictions and probabilities on test set.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # Fallback decision function converted to min-max pseudo-probability
        df = model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-df))
    else:
        y_prob = y_pred.astype(float)

    return model, y_pred, y_prob


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Trains all 5 classifiers and collects their trained instances and predictions.
    """
    raw_models = initialize_models(random_state=random_state)
    trained_results: Dict[str, Dict[str, Any]] = {}

    for name, clf in raw_models.items():
        trained_clf, y_pred, y_prob = train_single_model(clf, X_train, y_train, X_test)
        trained_results[name] = {
            "model": trained_clf,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    return trained_results


def save_artifacts(
    model_obj: Any,
    preprocessor_obj: Any,
    model_name: str,
    output_dir: str = "models",
    metadata: Dict[str, Any] = None,
) -> str:
    """
    Serializes the model, preprocessor, and metadata into a joblib bundle.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle_path = os.path.join(output_dir, "best_student_model.joblib")
    payload = {
        "model": model_obj,
        "preprocessor": preprocessor_obj,
        "model_name": model_name,
        "metadata": metadata or {},
    }
    joblib.dump(payload, bundle_path)
    return bundle_path


def load_artifacts(
    bundle_path: str = "models/best_student_model.joblib",
) -> Dict[str, Any]:
    """
    Loads saved model bundle containing model, preprocessor, and metadata.
    """
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model artifact not found at {bundle_path}")
    return joblib.load(bundle_path)
