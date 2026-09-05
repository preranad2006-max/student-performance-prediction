"""
Visualization Module for Student Performance Prediction System.
Generates publication-quality charts for Exploratory Data Analysis (EDA),
model evaluation metrics, confusion matrices, and feature importance.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set global modern aesthetic
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Plots correlation heatmap of numerical academic features."""
    num_cols = ["Study_Hours", "Attendance", "Previous_Marks", "Assignments", "Internal_Marks"]
    valid_cols = [c for c in num_cols if c in df.columns]

    corr = df[valid_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmax=1.0,
        vmin=-0.2,
        square=True,
        linewidths=1.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: Academic Performance Factors", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_feature_vs_result(
    df: pd.DataFrame, feature: str, target: str = "Final_Result"
) -> plt.Figure:
    """Plots feature distribution compared across Final Result (Pass vs Fail)."""
    fig, (ax_box, ax_kde) = plt.subplots(
        nrows=1, ncols=2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.4]}
    )

    palette = {"Pass": "#2ecc71", "Fail": "#e74c3c"}

    # Boxplot
    sns.boxplot(
        data=df,
        x=target,
        y=feature,
        hue=target,
        palette=palette,
        ax=ax_box,
        legend=False,
    )
    ax_box.set_title(f"{feature} Distribution by Result", fontweight="bold", fontsize=12)
    ax_box.set_xlabel("Final Result", fontweight="bold")
    ax_box.set_ylabel(feature, fontweight="bold")

    # KDE Distribution
    sns.kdeplot(
        data=df,
        x=feature,
        hue=target,
        palette=palette,
        fill=True,
        common_norm=False,
        alpha=0.4,
        ax=ax_kde,
    )
    ax_kde.set_title(f"{feature} Density Curve", fontweight="bold", fontsize=12)
    ax_kde.set_xlabel(feature, fontweight="bold")
    ax_kde.set_ylabel("Density", fontweight="bold")

    plt.tight_layout()
    return fig


def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plots distribution of Pass/Fail target and Grade breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Pass / Fail Count
    if "Final_Result" in df.columns:
        result_counts = df["Final_Result"].value_counts()
        colors = ["#2ecc71" if k == "Pass" else "#e74c3c" for k in result_counts.index]
        axes[0].pie(
            result_counts.values,
            labels=result_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            explode=[0.05] * len(result_counts),
            shadow=True,
            textprops={"fontsize": 11, "fontweight": "bold"},
        )
        axes[0].set_title("Overall Pass vs Fail Ratio", fontsize=13, fontweight="bold")

    # Grade Breakdown
    if "Grade" in df.columns:
        grade_order = ["A+", "A", "B", "C", "Fail"]
        existing_grades = [g for g in grade_order if g in df["Grade"].values]
        palette = {
            "A+": "#27ae60",
            "A": "#2ecc71",
            "B": "#3498db",
            "C": "#f39c12",
            "Fail": "#e74c3c",
        }
        sns.countplot(
            data=df,
            x="Grade",
            order=existing_grades,
            hue="Grade",
            palette=palette,
            ax=axes[1],
            legend=False,
        )
        axes[1].set_title("Grade Distribution Breakdown", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Student Grade", fontweight="bold")
        axes[1].set_ylabel("Number of Students", fontweight="bold")
        for p in axes[1].patches:
            height = p.get_height()
            if height > 0:
                axes[1].annotate(
                    f"{int(height)}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    plt.tight_layout()
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """Plots comparative bar chart across models for Accuracy, Precision, Recall, F1."""
    metrics_to_plot = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"]
    plot_df = comparison_df.melt(
        id_vars=["Model"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=plot_df,
        x="Model",
        y="Score",
        hue="Metric",
        palette="viridis",
        ax=ax,
    )
    ax.set_ylim(50, 105)
    ax.set_title("Machine Learning Algorithms Benchmark Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Model Architecture", fontweight="bold")
    ax.set_ylabel("Score (%)", fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    return fig


def plot_confusion_matrix_heatmap(cm: List[List[int]], model_name: str) -> plt.Figure:
    """Plots annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm_arr = np.array(cm)

    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Fail", "Predicted Pass"],
        yticklabels=["Actual Fail", "Actual Pass"],
        cbar=False,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str], importances: np.ndarray, title: str = "Feature Importance"
) -> plt.Figure:
    """Plots horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(7, 4))
    indices = np.argsort(importances)

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    ax.barh(range(len(sorted_features)), sorted_importances, color="#3498db", edgecolor="#2980b9")
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontweight="bold")
    ax.set_xlabel("Relative Importance Score", fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig
