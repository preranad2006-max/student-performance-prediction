@echo off
title Student Performance Prediction System
echo =====================================================================
echo    Student Performance Prediction System using Machine Learning
echo =====================================================================
echo.
echo Launching Streamlit Dashboard...
echo.

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -m streamlit run app.py
) else (
    echo Python virtual environment not found in .venv.
    echo Running with default python...
    python -m streamlit run app.py
)

pause
