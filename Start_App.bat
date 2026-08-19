@echo off
title ViralClipper AI Launcher
cd /d "%~dp0"

echo.
echo ========================================================
echo       ViralClipper AI - Dashboard Launcher
echo ========================================================
echo.

:: 1. Try embedded Python first (has all deps pre-installed)
set "PY_PATH=python\python.exe"
if exist "%PY_PATH%" (
    echo [System] Using embedded Python 3.10
    "%PY_PATH%" app_ui.py
    goto :DONE
)

:: 2. Fallback to system Python
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [System] Using system Python (ensure deps are installed)
    python app_ui.py
    goto :DONE
)

echo [ERROR] No Python found.
echo Please ensure the 'python' folder exists or install Python 3.10.
pause
exit /b

:DONE
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo App crashed or closed with an error.
    pause
)
