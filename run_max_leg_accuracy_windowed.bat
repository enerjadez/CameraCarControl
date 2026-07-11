@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -c "import vgamepad" >nul 2>nul
if errorlevel 1 (
  echo The optional gamepad backend is not installed.
  echo Run install_gamepad_backend.bat first.
  pause
  exit /b 1
)
rem Maximum leg/foot identity accuracy with an ordinary diagnostic window.
python camera_drive.py --pedals-only --profile leg-lock --windowed-hud
if errorlevel 1 pause
