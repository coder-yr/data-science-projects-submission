# Nexvia Data Science Platforms Dual Launcher (PowerShell)
$root = $PSScriptRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "             DATA SCIENCE PROJECTS PORTFOLIO LAUNCHER                 " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Churn Intelligence
Write-Host "[1/3] Launching DS-02 Churn Intelligence Dashboard (Port 8501)..." -ForegroundColor Yellow
$churnPath = Join-Path $root "DS-02-Churn-Intelligence"
Start-Process cmd.exe -ArgumentList "/k `"cd /d `"$churnPath`" && if not exist .venv (echo Creating .venv... && python -m venv .venv && call .venv\Scripts\activate.bat && python -m pip install --upgrade pip && pip install -r requirements.txt) else (call .venv\Scripts\activate.bat) && streamlit run app.py`""

# 2. Forecasting FastAPI
Write-Host "[2/3] Launching DS-01 Forecasting FastAPI Gateway (Port 8000)..." -ForegroundColor Yellow
$demandPath = Join-Path $root "DS-01-Nationwide-Demand-Forecasting"
Start-Process cmd.exe -ArgumentList "/k `"cd /d `"$demandPath`" && if not exist venv (echo Creating venv... && python -m venv venv && call venv\Scripts\activate.bat && python -m pip install --upgrade pip && pip install -r requirements.txt) else (call venv\Scripts\activate.bat) && python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000`""

# 3. Forecasting Dash Control Tower
Write-Host "[3/3] Launching DS-01 Forecasting Control Tower UI (Port 8050)..." -ForegroundColor Yellow
Start-Process cmd.exe -ArgumentList "/k `"cd /d `"$demandPath`" && if exist venv (call venv\Scripts\activate.bat) && python -m src.dashboard.app`""

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "All 3 services are launching in their own terminal windows:" -ForegroundColor Green
Write-Host "  1. DS-02 Churn Intelligence Dashboard:     http://localhost:8501" -ForegroundColor White
Write-Host "  2. DS-01 Forecasting REST API Docs:        http://localhost:8000/docs" -ForegroundColor White
Write-Host "  3. DS-01 Forecasting Control Tower:        http://localhost:8050" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Green
