@echo off
setlocal
cd /d "%~dp0"

title Multi-Disease ML System

echo ============================================================
echo        Multi-Disease Machine Learning System
echo ============================================================
echo.
echo Project folder:
echo %CD%
echo.

set "VENV_PYTHON=%CD%\.venv-1\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] The project Python was not found:
    echo %VENV_PYTHON%
    echo.
    echo Your project must contain:
    echo   .venv-1\Scripts\python.exe
    echo.
    pause
    exit /b 1
)

echo Using project virtual environment:
"%VENV_PYTHON%" -c "import sys; print(sys.executable)"
echo.

echo Checking Streamlit inside .venv-1...
"%VENV_PYTHON%" -c "import streamlit; print('Streamlit version:', streamlit.__version__)"

if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit is not installed inside .venv-1.
    echo.
    echo Install it with:
    echo "%VENV_PYTHON%" -m pip install streamlit
    echo.
    pause
    exit /b 1
)

echo.
echo Starting application...
echo.
echo URL: http://localhost:8501
echo.
"%VENV_PYTHON%" -m streamlit run "%CD%\app\main.py"

echo.
pause
endlocal
