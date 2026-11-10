@echo off
title Nexvia Dual Platform Launcher

echo ======================================================================
echo             DATA SCIENCE PROJECTS PORTFOLIO LAUNCHER
echo ======================================================================
echo.

set ROOT=%~dp0

echo [1/3] Preparing and Launching Churn Intelligence Dashboard (Port 8501)...
start "DS-02 Churn Intelligence (Streamlit :8501)" cmd /k "cd /d ""%ROOT%DS-02-Churn-Intelligence"" && if not exist .venv (echo Creating .venv... && python -m venv .venv && call .venv\Scripts\activate.bat && python -m pip install --upgrade pip && pip install -r requirements.txt) else (call .venv\Scripts\activate.bat) && streamlit run app.py"

echo [2/3] Preparing and Launching Demand Forecasting FastAPI Gateway (Port 8000)...
start "DS-01 Forecasting API (FastAPI :8000)" cmd /k "cd /d ""%ROOT%DS-01-Nationwide-Demand-Forecasting"" && if not exist venv (echo Creating venv... && python -m venv venv && call venv\Scripts\activate.bat && python -m pip install --upgrade pip && pip install -r requirements.txt) else (call venv\Scripts\activate.bat) && python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000"

echo [3/3] Preparing and Launching Demand Forecasting Control Tower (Port 8050)...
start "DS-01 Forecasting Control Tower (Dash :8050)" cmd /k "cd /d ""%ROOT%DS-01-Nationwide-Demand-Forecasting"" && if exist venv (call venv\Scripts\activate.bat) && python -m src.dashboard.app"

echo.
echo ======================================================================
echo All 3 services are launching in their own dedicated console windows:
echo.
echo   1. DS-02 Churn Intelligence Dashboard:
echo      URL: http://localhost:8501
echo.
echo   2. DS-01 Forecasting FastAPI REST Gateway (Swagger Docs):
echo      URL: http://localhost:8000/docs
echo.
echo   3. DS-01 Forecasting Plotly Dash Control Tower:
echo      URL: http://localhost:8050
echo ======================================================================
echo.
pause
