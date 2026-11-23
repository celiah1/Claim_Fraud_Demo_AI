@echo off
REM ClaimGuard launcher: double-click to open the app in your browser.
REM First run installs the required Python packages (needs internet, takes 1-3 minutes).
cd /d "%~dp0"
title ClaimGuard

REM ---- Find Python 3.10+ ----
set "PY="
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY ( py -3 --version >nul 2>&1 && set "PY=py -3" )
if not defined PY ( python --version >nul 2>&1 && set "PY=python" )
if not defined PY (
  echo.
  echo  Python was not found on this computer.
  echo  Install Python 3.12 from https://www.python.org/downloads/
  echo  ^(tick "Add python.exe to PATH" during install^), then double-click this file again.
  echo.
  pause
  exit /b 1
)

REM ---- Install packages on first run ----
%PY% -c "import streamlit, pandas, plotly, sklearn, joblib, anthropic, pypdf, openpyxl" >nul 2>&1
if errorlevel 1 (
  echo Installing required packages, please wait...
  %PY% -m pip install --quiet --disable-pip-version-check -r requirements.txt
  if errorlevel 1 ( echo Package install failed. Check your internet connection. & pause & exit /b 1 )
)

echo.
echo  ClaimGuard is starting at http://claimguard.localhost:8503
echo  Keep this window open while you use the app. Close it to stop the app.
echo.
start "" http://claimguard.localhost:8503
%PY% -m streamlit run app.py --server.port 8503 --server.headless true --browser.gatherUsageStats false
