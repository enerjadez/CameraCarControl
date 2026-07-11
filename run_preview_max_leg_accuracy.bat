@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
rem Clickable six-point hard-lock setup. Preview only; no virtual controller.
python camera_drive.py --pedals-only --profile leg-lock --preview-only --windowed-hud --anchor-setup
if errorlevel 1 pause
