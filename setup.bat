@echo off
REM ============================================================
REM  VisionOCR Pro - One-Click Environment Setup (Windows)
REM  Usage: double-click or run "setup.bat" in project root
REM  Requires: internet connection, Python 3.11-3.13, Git
REM ============================================================
setlocal enabledelayedexpansion
title VisionOCR Pro Setup

echo ============================================================
echo   VisionOCR Pro - Environment Setup
echo ============================================================
echo.

REM --- Step 0: Locate project root (where this script lives) ---
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"
echo [INFO] Project root: %PROJECT_ROOT%
echo.

REM --- Step 1: Check Python version (3.11 - 3.13) ---
REM 已有可用 .venv 时直接复用, 不扫 PATH (避免命中无关解释器)
if exist ".venv\Scripts\python.exe" (
    echo [1/7] Found existing .venv -- reusing it, skipping PATH scan
    echo       Interpreter: %PROJECT_ROOT%\.venv\Scripts\python.exe
    goto :venv_ready
)

echo [1/7] Checking Python...
set "PYTHON_EXE="
for %%V in (python3.13 python3.12 python3.11 python3 python) do (
    if "!PYTHON_EXE!"=="" (
        where %%V >nul 2>&1
        if !errorlevel!==0 (
            for /f "tokens=2 delims= " %%P in ('%%V --version 2^>^&1') do (
                set "PYVER=%%P"
            )
            REM Check major.minor
            for /f "tokens=1,2 delims=." %%A in ("!PYVER!") do (
                if %%A==3 (
                    if %%B GEQ 11 (
                        if %%B LEQ 13 (
                            set "PYTHON_EXE=%%V"
                        )
                    )
                )
            )
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python 3.11-3.13 not found in PATH.
    echo         Please install Python from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
echo [OK] Found: %PYTHON_EXE% (version %PYVER%)
for /f "delims=" %%W in ('where %PYTHON_EXE% 2^>nul') do (
    echo      Full path: %%W
    goto :show_once
)
:show_once
echo.

REM --- Step 2: Create virtual environment ---
echo [2/7] Setting up virtual environment...
%PYTHON_EXE% -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo [OK] Created .venv

:venv_ready
set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "VENV_PIP=%PROJECT_ROOT%\.venv\Scripts\pip.exe"
echo.

REM --- Step 3: Install PyTorch (CUDA 12.6) ---
echo [3/7] Installing PyTorch with CUDA 12.6 support...
echo        (This may take 5-10 minutes on first run, ~2.5GB download)
%VENV_PIP% install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo [WARN] PyTorch CUDA install failed. Trying CPU-only fallback...
    %VENV_PIP% install -q torch torchvision
    if errorlevel 1 (
        echo [ERROR] PyTorch installation failed entirely.
        pause
        exit /b 1
    )
    echo [WARN] Installed CPU-only PyTorch. GPU acceleration unavailable.
) else (
    echo [OK] PyTorch installed
)
echo.

REM --- Step 4: Install project dependencies ---
echo [4/7] Installing project dependencies...
%VENV_PIP% install -q -r requirements.txt
if errorlevel 1 (
    echo [WARN] Some dependencies failed. Retrying with mirror...
    %VENV_PIP% install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)
REM Install test dependencies
%VENV_PIP% install -q pytest
echo [OK] Dependencies installed
echo.

REM --- Step 5: Check Ollama ---
echo [5/7] Checking Ollama (local LLM runtime)...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [WARN] Ollama not found in PATH.
    echo        Contract automation LLM features will be unavailable.
    echo        Install from: https://ollama.com/download
    echo        After installing, run: ollama pull qwen3-vl:8b
    echo.
    set "OLLAMA_OK=0"
) else (
    echo [OK] Ollama found
    REM Check if model exists
    ollama list 2>nul | findstr /i "qwen3-vl" >nul 2>&1
    if errorlevel 1 (
        echo [INFO] Pulling qwen3-vl:8b model (~6.1GB, may take 10-30 min)...
        ollama pull qwen3-vl:8b
        if errorlevel 1 (
            echo [WARN] Model pull failed. You can retry later: ollama pull qwen3-vl:8b
            set "OLLAMA_OK=0"
        ) else (
            echo [OK] qwen3-vl:8b downloaded
            set "OLLAMA_OK=1"
        )
    ) else (
        echo [OK] qwen3-vl:8b already present
        set "OLLAMA_OK=1"
    )
)
echo.

REM --- Step 6: Download OCR models ---
echo [6/7] Downloading OCR models (OvisOCR2, ~1.7GB)...
%VENV_PYTHON% scripts/download_models.py ovisocr2
if errorlevel 1 (
    echo [WARN] OvisOCR2 download failed. RapidOCR will be used as fallback.
) else (
    echo [OK] OCR models ready
)
echo.

REM --- Step 7: Verification ---
echo [7/7] Running verification...
echo.

REM doctor 环境自检 (依赖/config 完整性)
%VENV_PYTHON% scripts\doctor.py
if errorlevel 1 (
    echo   [WARN] doctor reported failures -- see above
)
echo.

REM CUDA check
%VENV_PYTHON% -c "import torch; print(f'  CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"})')"
if errorlevel 1 (
    echo   CUDA: check failed
)

REM Unit tests
echo   Running unit tests...
%VENV_PYTHON% -m pytest tests/ -q --tb=no 2>nul
if errorlevel 1 (
    echo   [WARN] Some tests failed (non-critical for first setup)
) else (
    echo   [OK] All tests passed
)
echo.

REM --- Summary ---
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo   To start the application:
echo     run.bat
echo     (or: .venv\Scripts\python.exe app.py)
echo.
echo   Browser will open at: http://localhost:7860
echo.
if "%OLLAMA_OK%"=="0" (
    echo   [NOTE] Ollama/model not ready. Contract LLM features disabled.
    echo          Install Ollama and run: ollama pull qwen3-vl:8b
    echo.
)
echo   Documentation: DEPLOY.md
echo ============================================================
echo.
pause
endlocal
