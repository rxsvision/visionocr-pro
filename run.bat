@echo off
title VisionOCR Pro
cd /d D:\rxs-repos\visionocr-pro
"C:\Users\user\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" app.py %*
if errorlevel 1 pause
