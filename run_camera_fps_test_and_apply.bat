@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
echo Close CameraDrive and every other program using the webcam before continuing.
echo This test will save the fastest measured camera mode into config.json.
echo.
python camera_fps_test.py --apply
pause
