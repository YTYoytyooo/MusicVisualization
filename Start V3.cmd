@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "TEMP=%~dp0data\v3.0\runtime\tmp"
set "TMP=%TEMP%"
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%~dp0data\training\next_va\env\Scripts\python.exe" (
  echo V3 Python environment is missing. No automatic installation or training.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8768 in your browser.
"%~dp0data\training\next_va\env\Scripts\python.exe" -u "%~dp0versions\v3.0\main.py" serve %*
if errorlevel 1 pause
