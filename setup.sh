#!/usr/bin/env bash
# ============================================================
#  VisionOCR Pro - One-Click Environment Setup (Linux / Jetson)
#  Usage: chmod +x setup.sh && ./setup.sh
#  Requires: internet connection, Python 3.11-3.13, Git
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }
info() { echo -e "[INFO] $1"; }

echo "============================================================"
echo "  VisionOCR Pro - Environment Setup (Linux)"
echo "============================================================"
echo ""
info "Project root: $PROJECT_ROOT"
echo ""

# --- Step 1: Check Python ---
# 已有可用 .venv 时直接复用, 不扫 PATH (避免命中无关解释器)
if [[ -f ".venv/bin/python" ]]; then
    echo "[1/7] Found existing .venv -- reusing it, skipping PATH scan"
    echo "      Interpreter: $PROJECT_ROOT/.venv/bin/python"
    VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
    VENV_PIP="$PROJECT_ROOT/.venv/bin/pip"
    echo ""
    echo "[3/7] Ensuring PyTorch (CUDA 12.6)..."
    if ! "$VENV_PIP" install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126; then
        warn "CUDA PyTorch failed, trying CPU-only..."
        "$VENV_PIP" install -q torch torchvision || { err "PyTorch install failed"; exit 1; }
    fi
    echo "[4/7] Ensuring project dependencies..."
    "$VENV_PIP" install -q -r requirements.txt || "$VENV_PIP" install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    "$VENV_PIP" install -q pytest
    jump_ahead=1
else
    jump_ahead=0
fi

if [[ "$jump_ahead" -eq 0 ]]; then
echo "[1/7] Checking Python..."
PYTHON_EXE=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major="${ver%%.*}"
        minor="${ver##*.}"
        if [[ "$major" == "3" && "$minor" -ge 11 && "$minor" -le 13 ]]; then
            PYTHON_EXE="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON_EXE" ]]; then
    err "Python 3.11-3.13 not found."
    echo "       Install: sudo apt install python3.11 python3.11-venv"
    exit 1
fi
ok "Found: $PYTHON_EXE (version $ver)"
echo ""

# --- Step 2: Virtual environment ---
echo "[2/7] Setting up virtual environment..."
if [[ ! -f ".venv/bin/python" ]]; then
    "$PYTHON_EXE" -m venv .venv
    ok "Created .venv"
else
    ok ".venv already exists"
fi
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
VENV_PIP="$PROJECT_ROOT/.venv/bin/pip"
echo ""

# --- Step 3: PyTorch ---
echo "[3/7] Installing PyTorch (CUDA 12.6)..."
echo "       (This may take 5-10 minutes, ~2.5GB download)"
if ! "$VENV_PIP" install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126; then
    warn "CUDA PyTorch failed, trying CPU-only..."
    "$VENV_PIP" install -q torch torchvision || { err "PyTorch install failed"; exit 1; }
    warn "Installed CPU-only PyTorch"
else
    ok "PyTorch installed"
fi
echo ""

# --- Step 4: Project dependencies ---
echo "[4/7] Installing project dependencies..."
"$VENV_PIP" install -q -r requirements.txt || {
    warn "Retrying with mirror..."
    "$VENV_PIP" install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
}
# PaddlePaddle GPU (Linux only, no conflict with torch)
info "Installing PaddlePaddle GPU (Linux-only feature)..."
"$VENV_PIP" install -q paddlepaddle-gpu 2>/dev/null && ok "PaddlePaddle GPU installed" || warn "PaddlePaddle skipped (optional)"
"$VENV_PIP" install -q pytest
ok "Dependencies installed"
echo ""
fi  # end of fresh-install path (jump_ahead == 0)

# --- Step 5: Ollama ---
echo "[5/7] Checking Ollama..."
OLLAMA_OK=0
if command -v ollama &>/dev/null; then
    ok "Ollama found"
    if ollama list 2>/dev/null | grep -qi "qwen3-vl"; then
        ok "qwen3-vl:8b already present"
        OLLAMA_OK=1
    else
        info "Pulling qwen3-vl:8b (~6.1GB, may take 10-30 min)..."
        if ollama pull qwen3-vl:8b; then
            ok "qwen3-vl:8b downloaded"
            OLLAMA_OK=1
        else
            warn "Model pull failed. Retry later: ollama pull qwen3-vl:8b"
        fi
    fi
else
    warn "Ollama not found. Contract LLM features disabled."
    echo "       Install: curl -fsSL https://ollama.com/install.sh | sh"
    echo "       Then:    ollama pull qwen3-vl:8b"
fi
echo ""

# --- Step 6: OCR models ---
echo "[6/7] Downloading OCR models (OvisOCR2, ~1.7GB)..."
if "$VENV_PYTHON" scripts/download_models.py ovisocr2; then
    ok "OCR models ready"
else
    warn "OvisOCR2 download failed. RapidOCR will be used as fallback."
fi
echo ""

# --- Step 7: Verification ---
echo "[7/7] Running verification..."
echo ""
"$VENV_PYTHON" scripts/doctor.py || warn "doctor reported issues -- see above"
echo ""
"$VENV_PYTHON" -c "
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'
print(f'  CUDA: {torch.cuda.is_available()} ({gpu})')
" || echo "  CUDA: check failed"

echo "  Running unit tests..."
if "$VENV_PYTHON" -m pytest tests/ -q --tb=no 2>/dev/null; then
    ok "All tests passed"
else
    warn "Some tests failed (non-critical)"
fi
echo ""

# --- Summary ---
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "  To start the application:"
echo "    source .venv/bin/activate && python app.py"
echo ""
echo "  Browser will open at: http://localhost:7860"
echo ""
if [[ "$OLLAMA_OK" == "0" ]]; then
    echo "  [NOTE] Ollama/model not ready. Contract LLM features disabled."
    echo ""
fi
echo "  Documentation: DEPLOY.md"
echo "============================================================"
