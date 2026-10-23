@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows\gui.ps1" %*
set "gui_result=%errorlevel%"
if "%gui_result%"=="0" exit /b 0
echo Launcher failed. Run setup-windows.bat first, or read the error above.
if /i not "%KYOJIN_NO_PAUSE%"=="1" pause
exit /b %gui_result%
