@echo off
setlocal
cd /d "%~dp0"

title Final Model Comparison

set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Could not find:
    echo %VENV_PYTHON%
    echo.
    pause
    exit /b 1
)

echo Generating final model comparison...
echo.

"%VENV_PYTHON%" generate_final_results.py

echo.
pause
endlocal
