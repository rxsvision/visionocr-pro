@echo off
title VisionOCR Pro
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py %*
) else (
    echo [ERROR] .venv not found. Run setup.bat first.
    pause
)
if errorlevel 1 pause
