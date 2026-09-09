@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo Multi-Disease ML System
echo ============================================================
echo.

set "PYTHON_EXE="

if exist "%CD%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
)

if not defined PYTHON_EXE if exist "%CD%\.venv-1\Scripts\python.exe" (
    set "PYTHON_EXE=%CD%\.venv-1\Scripts\python.exe"
)

if not defined PYTHON_EXE (
    echo [ERROR] No project virtual environment was found.
    echo.
    echo Expected one of:
    echo   .venv\Scripts\python.exe
    echo   .venv-1\Scripts\python.exe
    echo.
    echo Place this file in the project root folder.
    pause
    exit /b 1
)

echo Using Python:
"%PYTHON_EXE%" -c "import sys; print(sys.executable)"
echo.

echo Checking Streamlit...
"%PYTHON_EXE%" -c "import streamlit" >nul 2>&1

if errorlevel 1 (
    echo Streamlit is not installed in the selected virtual environment.
    echo Installing requirements...
    "%PYTHON_EXE%" -m pip install -r requirements.txt

    if errorlevel 1 (
        echo.
        echo [ERROR] Could not install project requirements.
        pause
        exit /b 1
    )
)

echo.
echo Starting Multi-Disease ML System...
echo Browser URL: http://localhost:8501
echo.
"%PYTHON_EXE%" -m streamlit run app\main.py

echo.
pause
endlocal
