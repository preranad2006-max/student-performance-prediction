"""
Student Performance Prediction System - Core ML Package
"""

from .data_loader import generate_sample_dataset, load_dataset, save_default_datasets
from .preprocessor import StudentDataPreprocessor
from .models import initialize_models, train_all_models, save_artifacts, load_artifacts
from .evaluation import evaluate_single_model, compare_models
from .risk_analyzer import assess_student_risk, analyze_cohort_risk
from .visualizer import (
    plot_correlation_heatmap,
    plot_feature_vs_result,
    plot_target_distribution,
    plot_model_comparison,
    plot_confusion_matrix_heatmap,
    plot_feature_importance,
)

__all__ = [
    "generate_sample_dataset",
    "load_dataset",
    "save_default_datasets",
    "StudentDataPreprocessor",
    "initialize_models",
    "train_all_models",
    "save_artifacts",
    "load_artifacts",
    "evaluate_single_model",
    "compare_models",
    "assess_student_risk",
    "analyze_cohort_risk",
    "plot_correlation_heatmap",
    "plot_feature_vs_result",
    "plot_target_distribution",
    "plot_model_comparison",
    "plot_confusion_matrix_heatmap",
    "plot_feature_importance",
]
