@echo off
setlocal
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" --version >nul 2>&1
if errorlevel 1 (
  python "%~dp0tools\annotate\server.py" %*
) else (
  "%~dp0venv\Scripts\python.exe" "%~dp0tools\annotate\server.py" %*
)
if errorlevel 1 pause
