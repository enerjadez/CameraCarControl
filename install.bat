@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher was not found. Install 64-bit Python 3.11 first.
  pause
  exit /b 1
)

py -3.11 -m venv .venv
if errorlevel 1 (
  echo Could not create the Python 3.11 environment.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed. Read README.md troubleshooting.
  pause
  exit /b 1
)

echo.
echo Core camera tracking and HUD are installed.
echo Run run_preview_max_leg_accuracy.bat before installing the optional gamepad backend.
pause
