@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
rem Maximum leg/foot identity accuracy. Preview only; no virtual controller.
python camera_drive.py --pedals-only --profile leg-lock --preview-only --windowed-hud
if errorlevel 1 pause
