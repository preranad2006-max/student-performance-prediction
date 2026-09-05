# 🎓 Student Performance Prediction System using Machine Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit--Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License-MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status-Completed](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL_HERE)
A complete, end-to-end Machine Learning web application and data analysis system engineered to forecast student academic performance (Pass/Fail and Grades), identify at-risk students early, benchmark multiple predictive algorithms, and empower faculty and academic advisors with an interactive, data-driven decision dashboard.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Proposed Solution](#-proposed-solution)
4. [Dataset Features & Schema](#-dataset-features--schema)
5. [System Architecture & Workflow](#-system-architecture--workflow)
6. [Machine Learning Algorithms](#-machine-learning-algorithms)
7. [System Modules](#-system-modules)
8. [Project Structure](#-project-structure)
9. [Installation & Setup](#-installation--setup)
10. [How to Run](#-how-to-run)
11. [College Viva & Minor Project Q&A](#-college-viva--minor-project-qa)
12. [Future Enhancements](#-future-enhancements)

---

## 📖 Project Overview

Academic underperformance is often discovered too late in a semester—usually after midterm or final examinations have already concluded. The **Student Performance Prediction System** utilizes historical student behavioral and academic indicators—such as daily self-study hours, class attendance, prior examination marks, coursework assignment scores, and internal assessments—to accurately forecast whether a student will pass or fail and assign their probable letter grade.

By converting raw academic records into early warning signals, the platform highlights vulnerable students requiring immediate academic intervention, counseling, and remedial support.

---

## 🚨 Problem Statement

In traditional educational institutions:
- **No Early Warning Mechanism**: Teachers cannot easily detect lagging trends before final exams.
- **Manual Overhead**: Tracking disparate parameters (attendance sheets, assignment rubrics, and internal marks) across hundreds of students is cumbersome.
- **Subjective Decision-Making**: Remedial sessions are often arranged reactively based on intuition rather than empirical indicators.

---

## 💡 Proposed Solution

An automated, intelligent Machine Learning platform that:
- Ingests student academic data via CSV or Excel or through a built-in benchmark generator.
- Cleans, scales, and prepares the dataset using robust preprocessing techniques.
- Trains and benchmarks **5 distinct Machine Learning classifiers** on an **80% Training / 20% Testing** stratified split.
- Automatically selects the best-performing model based on F1-Score and Accuracy.
- Categorizes students into **Low**, **Medium**, and **High Risk** tiers.
- Formulates tailored, prescriptive academic recommendations (e.g. attendance counseling, remedial workshops).
- Provides an attractive, responsive, multi-page **Streamlit Web Dashboard**.

---

## 📊 Dataset Features & Schema

The dataset includes both behavioral and academic performance attributes:

| Feature Name | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| `Student_ID` | String | e.g. `STU1001` | Unique student identification code |
| `Study_Hours` | Float | 0.5 – 10.0 hrs | Daily hours dedicated to independent study |
| `Attendance` | Float | 40.0% – 100.0% | Percentage of lecture and laboratory attendance |
| `Previous_Marks` | Float | 20.0 – 100.0 | Marks obtained in previous semester / examination |
| `Assignments` | Float | 20.0 – 100.0 | Cumulative continuous assignment score |
| `Internal_Marks` | Float | 20.0 – 100.0 | Midterm and internal assessment test marks |
| `Final_Result` | Categorical | `Pass` / `Fail` | Target variable indicating semester outcome |
| `Grade` | Categorical | `A+`, `A`, `B`, `C`, `Fail` | Academic grade classification |

---

## 🔄 System Architecture & Workflow

```
                   +------------------------------------+
                   |     Student Data Collection        |
                   |  (CSV, Excel, or Benchmark Gen)    |
                   +-----------------+------------------+
                                     |
                                     v
                   +------------------------------------+
                   |         Data Preprocessing         |
                   |  - Missing Value Imputation        |
                   |  - Duplicate Elimination           |
                   |  - Standard / MinMax Scaling       |
                   +-----------------+------------------+
                                     |
                                     v
                   +------------------------------------+
                   |    Exploratory Data Analysis (EDA) |
                   |  - Correlation Heatmap             |
                   |  - Academic Factors vs Result      |
                   |  - Grade & Outcome Distributions   |
                   +-----------------+------------------+
                                     |
                                     v
                   +------------------------------------+
                   |    Stratified 80/20 Train-Test     |
                   +-----------------+------------------+
                                     |
                                     v
                   +------------------------------------+
                   |      Multi-Model Benchmark         |
                   |  1. Logistic Regression            |
                   |  2. Decision Tree                  |
                   |  3. Random Forest                  |
                   |  4. Support Vector Machine (SVM)   |
                   |  5. Gaussian Naive Bayes           |
                   +-----------------+------------------+
                                     |
                                     v
                   +------------------------------------+
                   |     Model Performance Evaluation   |
                   |  - Accuracy, Precision, Recall, F1 |
                   |  - Confusion Matrix & ROC-AUC      |
                   +-----------------+------------------+
                                     |
                                     v
                   +------------------------------------+
                   |    At-Risk Student Identification  |
                   |  - Failure Probability Gauge       |
                   |  - Low / Medium / High Risk Tiers  |
                   |  - Prescriptive Intervention Plan  |
                   +-----------------+------------------+
                                     |
                                     v
                   +------------------------------------+
                   |     Interactive Results Dashboard  |
                   |  - Executive KPI Summary Cards     |
                   |  - Action Center Table & CSV Export|
                   +------------------------------------+
```

---

## 🤖 Machine Learning Algorithms

The project trains and benchmarks **5 core classification algorithms**:

1. **Logistic Regression**: Linear decision boundary model optimizing the log-likelihood function. Provides direct, calibrated probabilities.
2. **Decision Tree Classifier**: Non-parametric tree structure using Gini impurity / Information Gain splits; highly interpretable.
3. **Random Forest Classifier**: Ensemble of 100 bagged decision trees with feature subsampling, reducing variance and offering robust feature importance scores.
4. **Support Vector Machine (SVM)**: Maximum margin hyperplane classifier utilizing Radial Basis Function (RBF) kernel with Platt probability calibration.
5. **Gaussian Naive Bayes**: Probabilistic classifier applying Bayes' Theorem under feature conditional independence assumptions.

---

## 📦 System Modules

### 1. Data Input Module (`src/data_loader.py`)
- Upload custom CSV or Excel (`.xlsx`, `.xls`) datasets.
- Schema verification and missing column validation.
- Built-in generator creating 1,000+ realistic correlated benchmark records.
- Export sample datasets directly from the UI.

### 2. Data Preprocessing Module (`src/preprocessor.py`)
- Automated deduplication and median imputation.
- Feature scaling using `StandardScaler` (or `MinMaxScaler`).
- Prevents data leakage by fitting scalers strictly on the 80% training split.

### 3. Exploratory Data Analysis Module (`src/visualizer.py`)
- Publication-quality seaborn / matplotlib charts:
  - Feature correlation heatmaps.
  - Boxplots & KDE density curves across Pass/Fail classes.
  - Grade distribution bar charts and pie charts.
  - Interactive feature inspector.

### 4. Feature Selection Module
- Feature importance visualization via Random Forest Gini importance.
- Correlation analysis with target variable.

### 5. Machine Learning & Evaluation Module (`src/models.py`, `src/evaluation.py`)
- Trains all 5 models concurrently.
- Formats a comprehensive comparative table ranking models by F1-Score.
- Generates annotated confusion matrix heatmaps and full classification reports.
- Serializes the winning model and scaler bundle to `models/best_student_model.joblib`.

### 6. Student Prediction Module (`app.py`)
- **Single Student Prediction**: Interactive sliders for study hours, attendance, and exam scores.
- **Viva Demo Presets**: One-click quick presets: "High Achiever", "Average Student", and "At-Risk Student".
- **Real-Time Output**: Pass/Fail badge, failure probability gauge, estimated grade class (A+, A, B, C, Fail).
- **Batch Cohort Prediction**: Upload an unlabelled cohort file to score hundreds of students in seconds.

### 7. At-Risk Student Identification Module (`src/risk_analyzer.py`)
- Categorizes each student into risk tiers:
  - 🟢 **Low Risk**: Pass Probability $\ge 70\%$ and Attendance $\ge 75\%$.
  - 🟡 **Medium Risk**: Pass Probability $45\% - 69\%$ or Attendance $65\% - 74\%$.
  - 🔴 **High Risk**: Pass Probability $< 45\%$ or Attendance $< 65\%$.
- Generates tailored academic recommendations (e.g. attendance counseling, foundational review, assignment remediation).

### 8. Results Dashboard (`app.py`)
- Executive KPI Cards: Total Students, Predicted Pass Count, Predicted Fail Count, High Risk Count, and Top Model Accuracy.
- Risk concentration scatter maps (Study Hours vs Attendance).
- Searchable, filterable student registry table with instant CSV export.

---

## 📂 Project Structure

```
Student-Performance-Prediction/
├── .venv/                      # Python virtual environment
├── data/                       # Datasets
│   ├── student_data.csv        # Benchmark CSV dataset (1,000 records)
│   └── student_data.xlsx       # Benchmark Excel dataset
├── models/                     # Serialized artifacts
│   └── best_student_model.joblib # Active trained model & preprocessor bundle
├── notebooks/                  # Interactive Notebooks
│   └── student_performance_prediction.ipynb # Step-by-step ML pipeline
├── src/                        # Modular Core Python Package
│   ├── __init__.py             # Package initializer
│   ├── data_loader.py          # Data ingestion, validation, and generation
│   ├── preprocessor.py         # Data cleaning, scaling, and 80/20 splitting
│   ├── models.py               # 5 ML model configurations & training logic
│   ├── evaluation.py           # Metric calculation and model benchmarking
│   ├── risk_analyzer.py        # Risk scoring and prescriptive advice
│   ├── visualizer.py           # Reusable matplotlib/seaborn plotting functions
│   └── create_notebook.py      # Notebook generator utility
├── tests/                      # Automated Unit Test Suite
│   └── test_ml_pipeline.py     # Pytest unit tests for all components
├── app.py                      # Modern Streamlit Web Application
├── requirements.txt            # Pinned dependencies
├── run_app.bat                 # Windows one-click launcher
└── README.md                   # Complete documentation & Viva Guide
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Git (optional)

### Setup Instructions

1. **Clone or Open the Repository Directory**:
   ```powershell
   cd "c:\Users\prerana d\Downloads\Project"
   ```

2. **Activate the Virtual Environment**:
   ```powershell
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   
   # Windows Command Prompt:
   .venv\Scripts\activate.bat
   ```

3. **Install Dependencies** (if creating a new environment):
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### 1. Launch the Interactive Web Dashboard
Double-click `run_app.bat` OR run:
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

### 2. Run Automated Verification Tests
```powershell
.venv\Scripts\python.exe -m pytest tests/test_ml_pipeline.py -v
```

### 3. Open the Jupyter Notebook
```powershell
.venv\Scripts\python.exe -m jupyter notebook notebooks/student_performance_prediction.ipynb
```
*(Or upload `notebooks/student_performance_prediction.ipynb` directly to Google Colab).*

---

## 🎓 College Viva & Minor Project Q&A

This section prepares students for questions commonly asked during academic project evaluations:

### Q1: Why is this treated as a Classification problem rather than a Regression problem?
**Answer**: While student grades can be numerical percentages (regression), academic institutions operate on categorical intervention boundaries: **Pass vs. Fail**, and grade bands (**A, B, C, Fail**). Classification allows us to compute actionable **failure probabilities**, set sensitivity thresholds, and trigger intervention protocols for high-risk cohorts.

### Q2: Why is the dataset split into 80% Training and 20% Testing?
**Answer**: An 80/20 split is an established empirical standard that provides sufficient training volume (800 samples) for the algorithms to discover robust patterns while preserving an adequate hold-out set (200 samples) to evaluate generalization performance without data leakage.

### Q3: Why is Stratified Splitting used?
**Answer**: Stratification guarantees that the proportion of Pass and Fail instances in the training set matches the proportion in the testing set, preventing sampling bias.

### Q4: Why did we fit the Scaler only on the Training Data?
**Answer**: Fitting the scaler on the entire dataset before splitting causes **data leakage**—information from the test distribution leaks into the training pipeline. In our system, `preprocessor.fit_transform()` strictly fits parameters ($\mu, \sigma$) on `X_train` and merely transforms `X_test` and future inference samples.

### Q5: How is the Risk Level calculated?
**Answer**: The system combines the ML model's calibrated failure probability ($P(\text{Fail}) = 1 - P(\text{Pass})$) with key empirical thresholds. If $P(\text{Fail}) \ge 50\%$ or attendance is below the mandatory $65\%$, the student is designated **High Risk**.

### Q6: Which model performed best and why?
**Answer**: On our benchmark dataset, **Logistic Regression** and **Random Forest** achieved the highest accuracy ($>93\%$) and F1-score ($>95\%$). Logistic Regression performs exceptionally well because academic indicators (study hours, attendance, continuous assessment) have a strong, monotonic relationship with the likelihood of passing.

---

## 🔮 Future Enhancements

1. **Learning Management System (LMS) Integration**: Direct plugins for Canvas, Moodle, and Google Classroom.
2. **Time-Series Deep Learning**: Incorporating LSTM / GRU networks to track day-by-day quiz trajectories over multiple semesters.
3. **Automated SMS & Email Alerts**: Automated dispatches to students and academic tutors when a student drops into the High-Risk category.
4. **Explainable AI (XAI)**: Incorporating SHAP (SHapley Additive exPlanations) values to provide individual visual feature attribution waterfall charts.
5. **Mobile Application**: Native mobile interface for faculty on Android / iOS.

---

## 📄 License & Attribution
Developed for academic research and educational demonstration. Distributed under the MIT License.
