@echo off
setlocal
python "%~dp0scripts\launch.py" v2 %*
if errorlevel 1 pause
