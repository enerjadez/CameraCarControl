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

rem Importing vgamepad does not prove that its ViGEmBus driver is installed.
rem Try creating a controller first so an existing working driver is not
rem needlessly reinstalled.
python -c "import vgamepad as vg; gamepad=vg.VX360Gamepad(); gamepad.reset(); gamepad.update()" >nul 2>nul
if not errorlevel 1 goto backend_ready

set "VIGEM_MSI="
for /f "usebackq delims=" %%I in (`python -c "from pathlib import Path; import vgamepad; root=Path(vgamepad.__file__).resolve().parent; matches=list(root.rglob('ViGEmBusSetup_x64.msi')); print(matches[0] if matches else '')"`) do set "VIGEM_MSI=%%I"
if not defined VIGEM_MSI (
  echo ViGEmBusSetup_x64.msi was not found inside the installed vgamepad package.
  echo Re-run install.bat, then try this installer again.
  pause
  exit /b 1
)

echo.
echo Windows will request administrator approval for the virtual controller driver.
echo Do not disable driver-signature enforcement or Windows security.
msiexec.exe /i "%VIGEM_MSI%"
set "VIGEM_EXIT=%ERRORLEVEL%"
if "%VIGEM_EXIT%"=="3010" goto backend_restart
if "%VIGEM_EXIT%"=="1641" goto backend_restart
if not "%VIGEM_EXIT%"=="0" (
  echo The ViGEmBus driver installation was cancelled or failed.
  echo Windows Installer exit code: %VIGEM_EXIT%
  pause
  exit /b 1
)

python -c "import vgamepad as vg; gamepad=vg.VX360Gamepad(); gamepad.reset(); gamepad.update()" >nul 2>nul
if errorlevel 1 (
  echo The driver installer finished, but a virtual controller could not be created yet.
  echo Restart Windows, then run this installer once more to verify it.
  pause
  exit /b 1
)

:backend_ready
echo.
echo Optional virtual Xbox controller backend and driver are working.
echo Do not disable Windows driver-signature or security checks.
pause
exit /b 0

:backend_restart
echo.
echo The virtual controller driver installed successfully and requires a restart.
echo Restart Windows before running CameraDrive gameplay mode.
pause
exit /b 0
