@echo off
setlocal
cd /d "%~dp0"
echo ===================================================
echo Starting SafeVision Edge Offline Backend Server...
echo ===================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python was not found in PATH.
    pause
    exit /b 1
)

echo Opening browser at http://127.0.0.1:8000/ ...
start "" http://127.0.0.1:8000/
python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
pause
