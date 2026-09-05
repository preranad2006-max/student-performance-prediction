"""
Student Performance Prediction System using Machine Learning
Modern, Interactive Streamlit Web Application
"""

import os
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import (
    load_dataset,
    generate_sample_dataset,
    REQUIRED_FEATURES,
    TARGET_COLUMN,
)
from src.preprocessor import StudentDataPreprocessor
from src.models import (
    train_all_models,
    save_artifacts,
    load_artifacts,
    MODEL_NAMES,
)
from src.evaluation import compare_models
from src.risk_analyzer import assess_student_risk, analyze_cohort_risk
from src.visualizer import (
    plot_correlation_heatmap,
    plot_feature_vs_result,
    plot_target_distribution,
    plot_model_comparison,
    plot_confusion_matrix_heatmap,
    plot_feature_importance,
)

# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Global Styles */
    .main {
        background-color: #f8fafc;
    }
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Card */
    .hero-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 2.2rem;
        border-radius: 14px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
        margin-bottom: 2rem;
    }
    .hero-title {
        color: #ffffff !important;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #e0e7ff;
        font-size: 1.1rem;
        max-width: 850px;
        line-height: 1.6;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.3rem;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    /* Result Cards */
    .pass-card {
        background: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        color: #065f46;
        text-align: center;
    }
    .fail-card {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        color: #991b1b;
        text-align: center;
    }
    
    /* Badge styling */
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-med {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
DATA_PATH = "data/student_data.csv"
MODEL_PATH = "models/best_student_model.joblib"

if "df" not in st.session_state:
    if os.path.exists(DATA_PATH):
        df_init, _ = load_dataset(DATA_PATH)
        st.session_state.df = df_init
    else:
        st.session_state.df = generate_sample_dataset(1000)

if "trained_results" not in st.session_state:
    st.session_state.trained_results = None

if "comparison_df" not in st.session_state:
    st.session_state.comparison_df = None

if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None

if "active_bundle" not in st.session_state:
    if os.path.exists(MODEL_PATH):
        try:
            st.session_state.active_bundle = load_artifacts(MODEL_PATH)
        except Exception:
            st.session_state.active_bundle = None
    else:
        st.session_state.active_bundle = None

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 EduPredict ML")
    st.markdown("**Student Performance Prediction**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home Page",
            "📁 Dataset Page",
            "📊 Data Analysis (EDA)",
            "⚙️ Model Training & Evaluation",
            "🔮 Student Prediction",
            "🎯 Dashboard & At-Risk Students",
        ],
    )

    st.markdown("---")
    st.markdown("### 📌 System Status")
    if st.session_state.df is not None:
        st.success(f"Dataset Loaded: **{len(st.session_state.df)} records**")
    else:
        st.warning("No dataset loaded.")

    if st.session_state.active_bundle is not None:
        model_name = st.session_state.active_bundle.get("model_name", "Unknown")
        st.info(f"Active Model: **{model_name}**")
    else:
        st.warning("No trained model saved.")

    st.markdown("---")
    st.markdown(
        "<small style='color: #64748b;'>Student Performance Prediction System<br>College Minor Project Demonstration</small>",
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 1. HOME PAGE
# -----------------------------------------------------------------------------
if page == "🏠 Home Page":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🎓 Student Performance Prediction System</div>
            <div class="hero-subtitle">
                An intelligent Machine Learning platform that forecasts student academic outcomes,
                enables early identification of at-risk students, and empowers educators with actionable, data-driven academic insights.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### 🚨 Problem Statement")
        st.markdown(
            """
            In conventional academic environments:
            - **Lagging Indicator Dilemma**: Teachers often discover a student's academic difficulty only after final examination failures.
            - **Manual Burden**: Tracking multi-dimensional trends across attendance, homework, and internal tests is labor-intensive.
            - **Subjective Decision-Making**: Remedial interventions are often triggered by intuition rather than empirical evidence.
            
            This system transforms historical student records into predictive intelligence, allowing proactive intervention weeks before final examinations.
            """
        )

        st.markdown("### 🎯 Project Objectives")
        st.markdown(
            """
            1. **Predict Academic Outcomes**: Accurately classify students into **Pass/Fail** and grade classes.
            2. **Identify At-Risk Students Early**: Automatically categorize students into **Low**, **Medium**, and **High Risk** cohorts.
            3. **Multi-Model Benchmark**: Train and compare 5 distinct Machine Learning classifiers (Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes).
            4. **Prescriptive Guidance**: Provide concrete academic recommendations tailored to each student's weaknesses.
            5. **Interactive Teacher Dashboard**: Deliver executive KPI summaries, visual charts, and exportable alert lists.
            """
        )

    with col2:
        st.markdown("### ⚙️ Machine Learning Pipeline")
        st.markdown(
            """
            ```
            1. Data Ingestion (CSV / Excel / Generator)
                     ↓
            2. Data Preprocessing (Cleaning, Imputation, Scaling)
                     ↓
            3. Exploratory Data Analysis (Heatmaps & Distributions)
                     ↓
            4. 80/20 Stratified Train-Test Split
                     ↓
            5. Model Training (5 Algorithms)
                     ↓
            6. Performance Evaluation (Accuracy, F1, ROC-AUC)
                     ↓
            7. Real-Time Inference & At-Risk Scoring
                     ↓
            8. Prescriptive Academic Interventions
            ```
            """
        )

        st.markdown("### 📊 Key Academic Indicators")
        st.markdown(
            """
            - ⏱️ **Study Hours**: Daily hours invested in self-study.
            - 📅 **Attendance**: Class attendance percentage.
            - 📝 **Previous Marks**: Past semester/term examination percentage.
            - 📑 **Assignments**: Average score across coursework assignments.
            - 🧪 **Internal Marks**: Continuous assessment and internal test marks.
            """
        )

# -----------------------------------------------------------------------------
# 2. DATASET PAGE
# -----------------------------------------------------------------------------
elif page == "📁 Dataset Page":
    st.title("📁 Dataset Management & Inspection")
    st.markdown(
        "Upload your student dataset (CSV or Excel) or generate a standard academic benchmark dataset."
    )

    col_upload, col_actions = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Student Records (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="Upload a file containing: Study_Hours, Attendance, Previous_Marks, Assignments, Internal_Marks, Final_Result",
        )

    with col_actions:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Generate 1,000 Benchmark Records", use_container_width=True):
            st.session_state.df = generate_sample_dataset(1000)
            st.success("Generated 1,000 realistic student records!")

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        df_uploaded, report = load_dataset(BytesIO(file_bytes), uploaded_file.name)
        if report["valid"]:
            st.session_state.df = df_uploaded
            st.success(
                f"Successfully loaded '{uploaded_file.name}' ({df_uploaded.shape[0]} rows, {df_uploaded.shape[1]} columns)"
            )
        else:
            st.error("Dataset validation failed:")
            for err in report["errors"]:
                st.error(f"- {err}")

    st.markdown("---")

    if st.session_state.df is not None:
        df = st.session_state.df

        # Top Summary KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Records", len(df))
        kpi2.metric("Total Features", len(df.columns))
        pass_ratio = (
            round((df["Final_Result"] == "Pass").mean() * 100, 1)
            if "Final_Result" in df.columns
            else "N/A"
        )
        kpi3.metric("Pass Percentage", f"{pass_ratio}%")
        missing_total = int(df.isnull().sum().sum())
        kpi4.metric("Missing Values", missing_total)

        # Tabbed dataset views
        tab_preview, tab_stats, tab_download = st.tabs(
            ["📋 Dataset Preview", "📈 Statistical Summary", "💾 Download Data"]
        )

        with tab_preview:
            st.markdown("#### Sample Records")
            st.dataframe(df.head(25), use_container_width=True)

        with tab_stats:
            st.markdown("#### Descriptive Statistics")
            st.dataframe(df.describe().round(2), use_container_width=True)

            st.markdown("#### Data Types & Missing Counts")
            info_df = pd.DataFrame(
                {
                    "Column": df.columns,
                    "Data Type": [str(t) for t in df.dtypes],
                    "Missing Count": df.isnull().sum().values,
                    "Unique Values": df.nunique().values,
                }
            )
            st.dataframe(info_df, use_container_width=True)

        with tab_download:
            st.markdown("#### Export Benchmark Dataset")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Dataset as CSV",
                    data=csv_data,
                    file_name="student_academic_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_d2:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="StudentData")
                st.download_button(
                    label="⬇️ Download Dataset as Excel (.xlsx)",
                    data=buffer.getvalue(),
                    file_name="student_academic_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

# -----------------------------------------------------------------------------
# 3. DATA ANALYSIS (EDA) PAGE
# -----------------------------------------------------------------------------
elif page == "📊 Data Analysis (EDA)":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown(
        "Interactive visualizations exploring correlations, student performance patterns, and key academic determinants."
    )

    if st.session_state.df is None:
        st.warning("Please load or generate a dataset in the Dataset page first.")
    else:
        df = st.session_state.df

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🔥 Correlation Heatmap",
                "📈 Academic Factors vs Result",
                "🎯 Performance Distributions",
                "🔍 Interactive Feature Inspector",
            ]
        )

        with tab1:
            st.markdown("### Correlation Between Academic Factors")
            st.markdown(
                "Examine how study hours, attendance, assignments, and prior marks correlate with each other."
            )
            fig_corr = plot_correlation_heatmap(df)
            st.pyplot(fig_corr)
            st.info(
                "💡 **Key Insight**: Study Hours and Attendance show strong positive linear relationships with internal exam scores and overall student success."
            )

        with tab2:
            st.markdown("### Academic Determinants vs Final Examination Result")
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.markdown("#### Study Hours vs Final Result")
                fig_sh = plot_feature_vs_result(df, "Study_Hours")
                st.pyplot(fig_sh)

            with sub_col2:
                st.markdown("#### Attendance vs Final Result")
                fig_att = plot_feature_vs_result(df, "Attendance")
                st.pyplot(fig_att)

            st.markdown("#### Previous Marks vs Final Result")
            fig_pm = plot_feature_vs_result(df, "Previous_Marks")
            st.pyplot(fig_pm)

        with tab3:
            st.markdown("### Overall Performance & Grade Breakdown")
            fig_target = plot_target_distribution(df)
            st.pyplot(fig_target)

        with tab4:
            st.markdown("### Interactive Feature Inspector")
            selected_feature = st.selectbox(
                "Select Academic Feature to Inspect",
                [
                    "Study_Hours",
                    "Attendance",
                    "Previous_Marks",
                    "Assignments",
                    "Internal_Marks",
                ],
            )
            fig_custom = plot_feature_vs_result(df, selected_feature)
            st.pyplot(fig_custom)

# -----------------------------------------------------------------------------
# 4. MODEL TRAINING & EVALUATION PAGE
# -----------------------------------------------------------------------------
elif page == "⚙️ Model Training & Evaluation":
    st.title("⚙️ Machine Learning Model Training & Benchmark")
    st.markdown(
        """
        Train and evaluate all **5 Machine Learning algorithms** on an **80% Training / 20% Testing** stratified split:
        `Logistic Regression`, `Decision Tree`, `Random Forest`, `Support Vector Machine (SVM)`, and `Naive Bayes`.
        """
    )

    if st.session_state.df is None:
        st.warning("Please load or generate a dataset in the Dataset page first.")
    else:
        df = st.session_state.df

        col_train_cfg, col_train_btn = st.columns([2, 1])
        with col_train_cfg:
            scaler_choice = st.selectbox(
                "Feature Scaling Method",
                ["StandardScaler (Standard Normalization)", "MinMaxScaler (0 to 1)"],
            )
            scaler_type = "minmax" if "MinMax" in scaler_choice else "standard"

        with col_train_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            train_clicked = st.button(
                "🚀 Train & Benchmark All 5 Models",
                type="primary",
                use_container_width=True,
            )

        if train_clicked:
            with st.spinner("Preprocessing dataset and training all 5 models..."):
                preprocessor = StudentDataPreprocessor(scaler_type=scaler_type)
                X_train, X_test, y_train, y_test, summary = preprocessor.fit_transform(
                    df, test_size=0.20, random_state=42
                )

                trained_results = train_all_models(
                    X_train, y_train, X_test, random_state=42
                )
                comp_df, best_name, detailed_evals = compare_models(
                    trained_results, y_test
                )

                st.session_state.trained_results = trained_results
                st.session_state.comparison_df = comp_df
                st.session_state.best_model_name = best_name
                st.session_state.detailed_evals = detailed_evals
                st.session_state.preprocessor = preprocessor

                # Save best model to disk
                best_clf = trained_results[best_name]["model"]
                save_artifacts(
                    best_clf,
                    preprocessor,
                    best_name,
                    metadata={
                        "comparison": comp_df.to_dict(orient="records"),
                        "summary": summary,
                    },
                )
                st.session_state.active_bundle = load_artifacts(MODEL_PATH)

            st.success(
                f"Training complete! Best performing algorithm: **{best_name}**"
            )

        # Display training results if available
        if st.session_state.comparison_df is not None:
            comp_df = st.session_state.comparison_df
            best_name = st.session_state.best_model_name
            detailed_evals = st.session_state.detailed_evals

            st.markdown("---")
            st.markdown(
                f"""
                <div style="background: #e0f2fe; border-left: 6px solid #0284c7; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.5rem;">
                    <h3 style="color: #0369a1; margin: 0;">🏆 Top Performing Model: {best_name}</h3>
                    <p style="color: #075985; margin: 0.4rem 0 0 0;">
                        Selected as the active production model based on highest composite F1-Score & Accuracy on the 20% unseen test set.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### 📊 Model Benchmark Comparison Table")
            st.dataframe(
                comp_df.style.highlight_max(
                    subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"],
                    color="#dcfce7",
                ),
                use_container_width=True,
            )

            # Benchmark Chart
            st.markdown("### 📈 Comparative Metric Visualization")
            fig_comp = plot_model_comparison(comp_df)
            st.pyplot(fig_comp)

            # Detailed Inspection per Model
            st.markdown("---")
            st.markdown("### 🔬 In-Depth Model Diagnostics & Confusion Matrix")
            selected_model = st.selectbox(
                "Select Model to Inspect Confusion Matrix & Classification Report",
                MODEL_NAMES,
                index=MODEL_NAMES.index(best_name) if best_name in MODEL_NAMES else 0,
            )

            if selected_model in detailed_evals:
                model_eval = detailed_evals[selected_model]
                col_cm, col_cr = st.columns([1, 1.2])

                with col_cm:
                    fig_cm = plot_confusion_matrix_heatmap(
                        model_eval["confusion_matrix"], selected_model
                    )
                    st.pyplot(fig_cm)

                with col_cr:
                    st.markdown("#### Classification Report Details")
                    cr_dict = model_eval["classification_report"]
                    cr_df = pd.DataFrame(cr_dict).transpose().round(3)
                    st.dataframe(cr_df, use_container_width=True)

                    st.markdown(
                        f"""
                        - **True Negatives (Correct Failures)**: `{model_eval['tn']}`
                        - **False Positives (Incorrectly Passed)**: `{model_eval['fp']}`
                        - **False Negatives (Incorrectly Failed)**: `{model_eval['fn']}`
                        - **True Positives (Correctly Passed)**: `{model_eval['tp']}`
                        """
                    )

            # Feature Importance if Random Forest or Decision Tree
            if "Random Forest" in st.session_state.trained_results:
                rf_model = st.session_state.trained_results["Random Forest"]["model"]
                if hasattr(rf_model, "feature_importances_"):
                    st.markdown("---")
                    st.markdown("### 🌳 Feature Importance (Random Forest)")
                    fig_fi = plot_feature_importance(
                        REQUIRED_FEATURES,
                        rf_model.feature_importances_,
                        "Feature Importance: Academic Factors",
                    )
                    st.pyplot(fig_fi)

# -----------------------------------------------------------------------------
# 5. STUDENT PREDICTION PAGE
# -----------------------------------------------------------------------------
elif page == "🔮 Student Prediction":
    st.title("🔮 Student Performance Prediction Engine")
    st.markdown(
        "Predict whether a student will Pass or Fail, identify their estimated Grade, and view tailored prescriptive recommendations."
    )

    if st.session_state.active_bundle is None:
        st.warning(
            "No active trained model found. Please go to the **Model Training & Evaluation** page and train the models."
        )
    else:
        active_bundle = st.session_state.active_bundle
        model = active_bundle["model"]
        preprocessor: StudentDataPreprocessor = active_bundle["preprocessor"]
        active_model_name = active_bundle.get("model_name", "Classifier")

        tab_single, tab_batch = st.tabs(
            ["👤 Single Student Prediction Form", "👥 Batch Student Prediction (CSV)"]
        )

        with tab_single:
            st.markdown(f"**Active Prediction Engine**: `{active_model_name}`")

            # Viva Demo Presets
            st.markdown("##### ⚡ Quick Presets (Ideal for College Viva Demonstration):")
            preset_col1, preset_col2, preset_col3 = st.columns(3)

            preset_study = 4.0
            preset_att = 75.0
            preset_prev = 65.0
            preset_assign = 70.0
            preset_int = 68.0

            if preset_col1.button("🌟 Preset: High Achiever"):
                preset_study = 8.5
                preset_att = 95.0
                preset_prev = 88.0
                preset_assign = 92.0
                preset_int = 90.0

            if preset_col2.button("⚖️ Preset: Average Student"):
                preset_study = 4.5
                preset_att = 78.0
                preset_prev = 62.0
                preset_assign = 65.0
                preset_int = 64.0

            if preset_col3.button("⚠️ Preset: At-Risk Student"):
                preset_study = 1.5
                preset_att = 52.0
                preset_prev = 38.0
                preset_assign = 42.0
                preset_int = 40.0

            st.markdown("---")

            # Input Form
            with st.form("student_prediction_form"):
                col_in1, col_in2 = st.columns(2)

                with col_in1:
                    study_hours = st.slider(
                        "Daily Study Hours (Hours/Day)",
                        min_value=0.5,
                        max_value=12.0,
                        value=float(preset_study),
                        step=0.5,
                        help="Number of hours the student devotes to independent study per day.",
                    )
                    attendance = st.slider(
                        "Class Attendance Percentage (%)",
                        min_value=30.0,
                        max_value=100.0,
                        value=float(preset_att),
                        step=1.0,
                        help="Semester attendance percentage.",
                    )
                    previous_marks = st.slider(
                        "Previous Semester / Exam Marks (%)",
                        min_value=10.0,
                        max_value=100.0,
                        value=float(preset_prev),
                        step=1.0,
                        help="Historical examination percentage.",
                    )

                with col_in2:
                    assignments = st.slider(
                        "Assignments Average Score (0-100)",
                        min_value=10.0,
                        max_value=100.0,
                        value=float(preset_assign),
                        step=1.0,
                        help="Cumulative score on coursework assignments.",
                    )
                    internal_marks = st.slider(
                        "Internal Assessment Marks (0-100)",
                        min_value=10.0,
                        max_value=100.0,
                        value=float(preset_int),
                        step=1.0,
                        help="Mid-term and continuous evaluation score.",
                    )

                predict_btn = st.form_submit_button(
                    "🔮 Predict Student Academic Performance",
                    type="primary",
                    use_container_width=True,
                )

            # Live prediction display (either upon submit or initial load)
            student_features = {
                "Study_Hours": study_hours,
                "Attendance": attendance,
                "Previous_Marks": previous_marks,
                "Assignments": assignments,
                "Internal_Marks": internal_marks,
            }

            X_student = preprocessor.transform_single(student_features)
            pred_class = int(model.predict(X_student)[0])

            if hasattr(model, "predict_proba"):
                prob_pass = float(model.predict_proba(X_student)[0][1])
            else:
                prob_pass = 1.0 if pred_class == 1 else 0.0

            risk_eval = assess_student_risk(
                prob_pass=prob_pass,
                study_hours=study_hours,
                attendance=attendance,
                previous_marks=previous_marks,
                assignments=assignments,
                internal_marks=internal_marks,
            )

            st.markdown("### 🎯 Academic Prediction Outcome")

            res_col1, res_col2 = st.columns([1, 1.2])

            with res_col1:
                if pred_class == 1:
                    st.markdown(
                        f"""
                        <div class="pass-card">
                            <h2 style="color: #065f46; margin: 0;">🎉 PREDICTED TO PASS</h2>
                            <p style="font-size: 1.1rem; margin: 0.5rem 0 0 0;">
                                Confidence: <strong>{risk_eval['pass_probability']}%</strong>
                            </p>
                            <p style="font-size: 1rem; color: #047857;">Estimated Grade: <strong>{risk_eval['estimated_grade']}</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="fail-card">
                            <h2 style="color: #991b1b; margin: 0;">⚠️ PREDICTED TO FAIL</h2>
                            <p style="font-size: 1.1rem; margin: 0.5rem 0 0 0;">
                                Failure Probability: <strong>{risk_eval['failure_probability']}%</strong>
                            </p>
                            <p style="font-size: 1rem; color: #b91c1c;">Estimated Grade: <strong>Fail</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**Risk Level Evaluation**: {risk_eval['risk_badge']}")
                st.progress(
                    int(risk_eval["pass_probability"]),
                    text=f"Pass Probability: {risk_eval['pass_probability']}%",
                )

            with res_col2:
                st.markdown("#### 📋 Prescriptive Academic Interventions")
                for rec in risk_eval["recommendations"]:
                    st.markdown(f"- 💡 {rec}")

                st.markdown(
                    f"""
                    <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-top: 1rem;">
                        <strong>Academic Composite Score:</strong> {risk_eval['composite_score']}/100<br>
                        <strong>Risk Category:</strong> {risk_eval['risk_level']}<br>
                        <strong>Monitoring Priority:</strong> Priority {risk_eval['priority']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab_batch:
            st.markdown("### 👥 Batch Student Prediction from File")
            st.markdown(
                "Upload a CSV or Excel file of students containing academic factors to generate predictions and risk levels for the entire cohort."
            )

            batch_file = st.file_uploader(
                "Upload Cohort File (CSV or Excel)",
                type=["csv", "xlsx"],
                key="batch_uploader",
            )

            if batch_file is not None:
                b_df, b_report = load_dataset(BytesIO(batch_file.read()), batch_file.name)
                if b_report["valid"]:
                    st.write(f"Loaded **{len(b_df)} students** for batch scoring.")
                    X_batch_scaled, df_clean = preprocessor.transform_dataframe(b_df)
                    b_preds = model.predict(X_batch_scaled)
                    b_probs = (
                        model.predict_proba(X_batch_scaled)[:, 1]
                        if hasattr(model, "predict_proba")
                        else b_preds.astype(float)
                    )

                    df_scored = analyze_cohort_risk(df_clean, b_preds, b_probs)

                    st.dataframe(
                        df_scored[
                            [
                                "Student_ID",
                                "Study_Hours",
                                "Attendance",
                                "Previous_Marks",
                                "Predicted_Result",
                                "Risk_Badge",
                                "Pass_Probability (%)",
                                "Primary_Intervention",
                            ]
                        ],
                        use_container_width=True,
                    )

                    csv_scored = df_scored.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Export Scored Student Roster (CSV)",
                        data=csv_scored,
                        file_name="scored_student_cohort.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.error("File validation failed: " + ", ".join(b_report["errors"]))

# -----------------------------------------------------------------------------
# 6. DASHBOARD & AT-RISK STUDENTS PAGE
# -----------------------------------------------------------------------------
elif page == "🎯 Dashboard & At-Risk Students":
    st.title("🎯 Results Dashboard & At-Risk Student Identification")
    st.markdown(
        "Real-time overview of student academic standing, cohort health metrics, and high-risk student alerts."
    )

    if st.session_state.df is None or st.session_state.active_bundle is None:
        st.warning(
            "Please ensure a dataset is loaded and a model is trained to view the live dashboard."
        )
    else:
        df = st.session_state.df
        active_bundle = st.session_state.active_bundle
        model = active_bundle["model"]
        preprocessor: StudentDataPreprocessor = active_bundle["preprocessor"]
        active_model_name = active_bundle.get("model_name", "Classifier")

        # Perform cohort scoring
        X_scaled, df_clean = preprocessor.transform_dataframe(df)
        y_preds = model.predict(X_scaled)
        y_probs = (
            model.predict_proba(X_scaled)[:, 1]
            if hasattr(model, "predict_proba")
            else y_preds.astype(float)
        )

        df_cohort = analyze_cohort_risk(df_clean, y_preds, y_probs)

        total_students = len(df_cohort)
        pass_count = int(np.sum(y_preds == 1))
        fail_count = int(np.sum(y_preds == 0))
        high_risk_count = int(np.sum(df_cohort["Risk_Level"] == "High Risk"))
        med_risk_count = int(np.sum(df_cohort["Risk_Level"] == "Medium Risk"))
        low_risk_count = int(np.sum(df_cohort["Risk_Level"] == "Low Risk"))

        # Executive KPI Cards
        kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
        with kpi_c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Students</div>
                    <div class="metric-value">{total_students}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c2:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left-color: #10b981;">
                    <div class="metric-label">Predicted Pass</div>
                    <div class="metric-value" style="color: #10b981;">{pass_count} ({round(pass_count/total_students*100, 1)}%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c3:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left-color: #ef4444;">
                    <div class="metric-label">Predicted Fail</div>
                    <div class="metric-value" style="color: #ef4444;">{fail_count} ({round(fail_count/total_students*100, 1)}%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c4:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left-color: #dc2626;">
                    <div class="metric-label">At-Risk (High)</div>
                    <div class="metric-value" style="color: #dc2626;">{high_risk_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c5:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left-color: #8b5cf6;">
                    <div class="metric-label">Active Model</div>
                    <div class="metric-value" style="font-size: 1.25rem; color: #8b5cf6;">{active_model_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("### 📊 Cohort Risk Distribution")
            risk_counts = pd.Series(
                {
                    "High Risk": high_risk_count,
                    "Medium Risk": med_risk_count,
                    "Low Risk": low_risk_count,
                }
            )
            fig_risk, ax_r = plt.subplots(figsize=(6, 4))
            ax_r.pie(
                risk_counts.values,
                labels=risk_counts.index,
                autopct="%1.1f%%",
                colors=["#ef4444", "#f59e0b", "#10b981"],
                startangle=140,
                explode=[0.08, 0.04, 0.02],
                textprops={"fontweight": "bold"},
            )
            ax_r.set_title("Student Risk Distribution", fontweight="bold", fontsize=12)
            st.pyplot(fig_risk)

        with chart_col2:
            st.markdown("### 🔍 Attendance vs Study Hours (Risk Mapping)")
            fig_scat, ax_s = plt.subplots(figsize=(7, 4.2))
            risk_palette = {
                "Low Risk": "#10b981",
                "Medium Risk": "#f59e0b",
                "High Risk": "#ef4444",
            }
            sns.scatterplot(
                data=df_cohort,
                x="Study_Hours",
                y="Attendance",
                hue="Risk_Level",
                palette=risk_palette,
                alpha=0.8,
                ax=ax_s,
            )
            ax_s.set_title("Risk Concentration Map", fontweight="bold", fontsize=12)
            ax_s.set_xlabel("Daily Study Hours", fontweight="bold")
            ax_s.set_ylabel("Attendance Percentage (%)", fontweight="bold")
            ax_s.legend(title="Risk Level", frameon=True)
            st.pyplot(fig_scat)

        st.markdown("---")

        # At-Risk Students Action Table
        st.markdown("### 🚨 At-Risk Student Action Center")
        st.markdown(
            "Filter, investigate, and export student records requiring academic counseling or remedial support."
        )

        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            risk_filter = st.selectbox(
                "Filter by Risk Level",
                ["All Students", "High Risk Only", "Medium Risk Only", "Low Risk Only"],
            )

        with filter_col2:
            search_id = st.text_input("Search by Student ID (e.g. STU1024)")

        filtered_df = df_cohort.copy()
        if risk_filter == "High Risk Only":
            filtered_df = filtered_df[filtered_df["Risk_Level"] == "High Risk"]
        elif risk_filter == "Medium Risk Only":
            filtered_df = filtered_df[filtered_df["Risk_Level"] == "Medium Risk"]
        elif risk_filter == "Low Risk Only":
            filtered_df = filtered_df[filtered_df["Risk_Level"] == "Low Risk"]

        if search_id.strip():
            filtered_df = filtered_df[
                filtered_df["Student_ID"].str.contains(search_id.strip(), case=False, na=False)
            ]

        display_cols = [
            "Student_ID",
            "Study_Hours",
            "Attendance",
            "Previous_Marks",
            "Assignments",
            "Internal_Marks",
            "Predicted_Result",
            "Risk_Badge",
            "Pass_Probability (%)",
            "Primary_Intervention",
        ]
        valid_display = [c for c in display_cols if c in filtered_df.columns]

        st.dataframe(filtered_df[valid_display], use_container_width=True)

        st.download_button(
            label="⬇️ Export At-Risk Action List (CSV)",
            data=filtered_df[valid_display].to_csv(index=False).encode("utf-8"),
            file_name="at_risk_students_action_plan.csv",
            mime="text/csv",
            use_container_width=True,
        )
