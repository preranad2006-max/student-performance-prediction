"""
At-Risk Student Identification and Intervention Module.
Analyzes student performance risks, assigns Risk Levels (Low, Medium, High),
and produces prescriptive, actionable academic intervention recommendations.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def assess_student_risk(
    prob_pass: float,
    study_hours: float,
    attendance: float,
    previous_marks: float,
    assignments: float,
    internal_marks: float,
) -> Dict[str, Any]:
    """
    Evaluates individual student risk based on ML failure probability and academic indicators.
    
    Risk Levels:
        - Low Risk: Pass Probability >= 0.70 AND Attendance >= 75%
        - Medium Risk: Pass Probability between 0.45 and 0.69 OR Attendance 65-74%
        - High Risk: Pass Probability < 0.45 OR Attendance < 65%
    """
    prob_fail = max(0.0, min(1.0, 1.0 - prob_pass))

    # Determine risk category
    if prob_fail >= 0.50 or attendance < 65.0:
        risk_level = "High Risk"
        risk_badge = "🔴 High Risk"
        priority = 1
    elif prob_fail >= 0.28 or attendance < 75.0 or previous_marks < 50.0:
        risk_level = "Medium Risk"
        risk_badge = "🟡 Medium Risk"
        priority = 2
    else:
        risk_level = "Low Risk"
        risk_badge = "🟢 Low Risk"
        priority = 3

    # Generate prescriptive recommendations
    recommendations: List[str] = []

    if attendance < 75.0:
        recommendations.append(
            f"Attendance Alert ({attendance}%): Below mandatory 75% threshold. Require attendance counseling."
        )
    if study_hours < 3.0:
        recommendations.append(
            f"Study Habit Alert ({study_hours} hrs/day): Recommend structured study timetable and peer study group."
        )
    if assignments < 50.0:
        recommendations.append(
            f"Assignment Deficit ({assignments}/100): Schedule assignment remedial workshops and re-submission opportunity."
        )
    if internal_marks < 50.0:
        recommendations.append(
            f"Internal Marks Alert ({internal_marks}/100): Conduct targeted concept review before semester examination."
        )
    if previous_marks < 50.0:
        recommendations.append(
            f"Historical Performance ({previous_marks}/100): Pair with student mentor for foundation strengthening."
        )

    if not recommendations:
        recommendations.append(
            "Performance is well on track. Encourage student to maintain current consistency and aim for honors/distinction."
        )

    # Estimate Grade Class
    composite_est = (
        0.25 * previous_marks
        + 0.25 * internal_marks
        + 0.20 * assignments
        + 0.15 * attendance
        + 0.15 * (study_hours * 10)
    )

    if prob_pass < 0.50:
        estimated_grade = "Fail"
    elif composite_est >= 85:
        estimated_grade = "A+ (Distinction)"
    elif composite_est >= 75:
        estimated_grade = "A (First Class with Distinction)"
    elif composite_est >= 60:
        estimated_grade = "B (First Class)"
    else:
        estimated_grade = "C (Second Class / Pass)"

    return {
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "priority": priority,
        "failure_probability": round(prob_fail * 100, 1),
        "pass_probability": round(prob_pass * 100, 1),
        "estimated_grade": estimated_grade,
        "composite_score": round(composite_est, 1),
        "recommendations": recommendations,
    }


def analyze_cohort_risk(
    df: pd.DataFrame,
    y_preds: np.ndarray,
    y_probs: np.ndarray,
) -> pd.DataFrame:
    """
    Performs batch risk analysis across an entire student cohort.
    
    Returns:
        df_risk: DataFrame enriched with Predicted_Result, Pass_Prob, Risk_Level, and Action
    """
    df_risk = df.copy()

    df_risk["Predicted_Result"] = np.where(y_preds == 1, "Pass", "Fail")
    df_risk["Pass_Probability (%)"] = (y_probs * 100).round(1)
    df_risk["Fail_Probability (%)"] = ((1.0 - y_probs) * 100).round(1)

    risk_levels = []
    badges = []
    actions = []

    for i in range(len(df_risk)):
        row = df_risk.iloc[i]
        prob_pass = float(y_probs[i])
        risk_info = assess_student_risk(
            prob_pass=prob_pass,
            study_hours=float(row.get("Study_Hours", 4.0)),
            attendance=float(row.get("Attendance", 75.0)),
            previous_marks=float(row.get("Previous_Marks", 60.0)),
            assignments=float(row.get("Assignments", 60.0)),
            internal_marks=float(row.get("Internal_Marks", 60.0)),
        )
        risk_levels.append(risk_info["risk_level"])
        badges.append(risk_info["risk_badge"])
        actions.append(risk_info["recommendations"][0])

    df_risk["Risk_Level"] = risk_levels
    df_risk["Risk_Badge"] = badges
    df_risk["Primary_Intervention"] = actions

    return df_risk
