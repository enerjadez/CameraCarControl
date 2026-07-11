@echo off
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Run install.bat first.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install -r requirements-gamepad.txt
if errorlevel 1 (
  echo Gamepad backend installation failed. Read README.md troubleshooting.
  pause
  exit /b 1
)

echo.
echo Optional virtual Xbox controller backend installed.
echo Its driver installer may require administrator approval.
echo Do not disable Windows driver-signature or security checks.
pause
