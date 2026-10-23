@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\setup.ps1" %*
set "setup_result=%errorlevel%"
if not "%setup_result%"=="0" echo Setup failed. Read the error above and see doc\install-windows.md.
if /i not "%KYOJIN_NO_PAUSE%"=="1" pause
exit /b %setup_result%
