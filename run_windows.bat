@echo off
cd /d "%~dp0"

echo Starting Artist Reference Asset Organizer...
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo.
    echo Run these setup commands from the project folder first:
    echo py -m venv .venv
    echo .venv\Scripts\Activate.ps1
    echo pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" app.py

echo.
echo Application closed.
pause
