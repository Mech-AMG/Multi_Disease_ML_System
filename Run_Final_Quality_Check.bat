@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Multi-Disease ML System - Final Quality Check
echo ============================================================
echo.

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Project virtual environment was not found:
    echo         %PYTHON_EXE%
    echo.
    echo Make sure this file is placed in the project root,
    echo next to the .venv folder.
    echo.
    pause
    exit /b 1
)

echo Using project Python:
"%PYTHON_EXE%" -c "import sys; print(sys.executable)"
echo.

echo Checking required runtime packages...
"%PYTHON_EXE%" -c "import streamlit, seaborn, pandas, numpy, sklearn, joblib, matplotlib, scipy" >nul 2>&1

if errorlevel 1 (
    echo Some required packages are missing from .venv.
    echo Installing requirements into the project virtual environment...
    echo.
    "%PYTHON_EXE%" -m pip install -r requirements.txt

    if errorlevel 1 (
        echo.
        echo [ERROR] Package installation failed.
        pause
        exit /b 1
    )
)

echo.
echo Running final quality check with the project virtual environment...
echo.
"%PYTHON_EXE%" final_quality_check.py

echo.
pause
endlocal
