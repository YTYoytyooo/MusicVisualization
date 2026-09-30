@echo off
setlocal
python "%~dp0scripts\launch.py" v1 %*
if errorlevel 1 pause
