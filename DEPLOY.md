# VisionOCR Pro 部署指南

## 目录

1. [硬件要求](#1-硬件要求)
2. [软件前置条件](#2-软件前置条件)
3. [安装步骤](#3-安装步骤)
4. [配置](#4-配置)
5. [验证](#5-验证)
6. [故障排除](#6-故障排除)
7. [常见问题](#7-常见问题)

---

## 1. 硬件要求

| 组件 | 最低要求 | 推荐配置 | 备注 |
|------|----------|----------|------|
| GPU | NVIDIA GPU, 8GB VRAM | NVIDIA GPU, 12GB+ VRAM | 已测试: RTX 4070 Ti (12GB) |
| 内存 | 16GB RAM | 32GB RAM | 模型加载峰值占用较高 |
| 存储 | SSD, 50GB 可用空间 | NVMe SSD, 100GB+ | 模型文件 + Ollama 权重约 10GB |
| CPU | 4 核 | 8 核+ | 图像预处理使用 CPU |

> 注意: 无 NVIDIA GPU 时系统可降级为 CPU 推理, 但速度显著下降 (单张 OCR 约 15-30s), 不建议生产使用。

## 2. 软件前置条件

| 软件 | 版本要求 | 用途 | 备注 |
|------|----------|------|------|
| Python | 3.11 - 3.13 | 运行时 | **不支持 3.14** (包生态为空, 多数依赖无 wheel) |
| CUDA Driver | 12.x | GPU 加速 | 仅需驱动, 不需单独安装 CUDA Toolkit |
| Git | 2.x+ | 代码拉取 | -- |
| Ollama | 最新版 | 本地 LLM 推理 | 合同自动化 LLM 抽取引擎 |
| pip | 最新版 | 包管理 | 随 Python 自带 |

## 3. 安装步骤

### 3.1 Windows

```bat
:: 1. 克隆仓库
git clone <repo-url> visionocr-pro
cd visionocr-pro

:: 2. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

:: 3. 安装 PyTorch (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

:: 4. 安装项目依赖
pip install -r requirements.txt
```

**Windows 重要说明:**

- **不要安装 paddlepaddle-gpu。** 在 Windows 上, PaddlePaddle 的 cudnn DLL 与 PyTorch 存在冲突, 会导致 torch CUDA 初始化失败。Windows 下 OCR 由 RapidOCR 引擎处理, 功能完整, 无需 PaddleOCR。
- 启动应用请使用项目根目录的 `run.bat`。该脚本为纯 ASCII 编码, 直接指向 venv 中的 Python 解释器, 避免路径和编码问题。

```bat
:: 启动应用
run.bat
```

### 3.2 Linux / NVIDIA Jetson

```bash
# 1. 克隆仓库
git clone <repo-url> visionocr-pro
cd visionocr-pro

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装 PyTorch (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 安装 PaddlePaddle GPU (Linux 下正常工作)
pip install paddlepaddle-gpu

# 6. 启动应用
python app.py
```

**Linux/Jetson 说明:**

- paddlepaddle-gpu 在 Linux 下与 PyTorch 无冲突, 可正常安装。安装后可启用 PaddleOCR-VL 引擎, 提供更高精度的版面分析能力。
- Jetson 设备请确保已刷写对应 JetPack 版本, 且 CUDA 驱动与 PyTorch cu126 兼容。

### 3.3 安装 Ollama 及模型

```bash
# 安装 Ollama (参见 https://ollama.com 获取对应平台安装包)

# 拉取合同自动化 LLM 模型 (约 6.1GB)
ollama pull qwen3-vl:8b
```

该模型用于合同自动化场景中的 LLM 信息抽取。若不使用合同自动化功能, 可跳过此步。

## 4. 配置

### 4.1 环境变量

```bash
# 复制环境变量模板
cp .env.example .env
```

编辑 `.env`, 按需填入:

- API 密钥 (如使用云端 OCR/LLM 服务)
- SDK 路径 (如本地 SDK 部署)
- 其他服务连接信息

### 4.2 应用配置

编辑 `config.yaml`:

- `company_name`: 公司名称 (用于报告输出)
- `camera_type`: 相机类型 (根据实际硬件选择)
- 其他业务参数按需调整

## 5. 验证

### 5.1 功能验证

```bash
python app.py
```

1. 浏览器自动打开 `http://localhost:7860`
2. 切换到 **OCR** 标签页
3. 上传任意含文字的图片
4. 预期: 5 秒内返回识别结果

### 5.2 单元测试

```bash
python -m pytest tests/ -v
```

预期: 17 个测试全部通过。

### 5.3 CUDA 验证

```python
import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.get_device_name(0))  # 应输出 GPU 型号
```

## 6. 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 端口 7860 被占用 | 残留的 Python 进程未退出 | Windows: `taskkill /F /IM python.exe`; Linux: `fuser -k 7860/tcp` |
| `torch.cuda.is_available()` 返回 False | CUDA 驱动未安装或版本不匹配 | 安装/更新 NVIDIA 驱动至支持 CUDA 12.x 的版本 |
| Ollama 连接失败 | Ollama 服务未启动 | 运行 `ollama serve` 启动服务, 再重试 |
| Windows 下 paddle DLL 报错 | cudnn DLL 与 torch 冲突 | **预期行为**, 无需修复。Windows 下使用 RapidOCR 即可 |
| pip 安装超时 | 网络问题 | 使用镜像源: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| Python 3.14 下依赖安装失败 | 包生态尚未支持 3.14 | 卸载 3.14, 安装 Python 3.11-3.13 |

## 7. 常见问题

**Q: Windows 下为什么不能装 paddlepaddle-gpu?**

A: PaddlePaddle 在 Windows 上捆绑的 cudnn DLL 与 PyTorch 自带的 cudnn 版本冲突, 导致 `torch.cuda.is_available()` 返回 False 或运行时崩溃。Windows 下 RapidOCR 已覆盖全部 OCR 需求, 无需 PaddleOCR。

**Q: 为什么不支持 Python 3.14?**

A: Python 3.14 发布时间较新, 主流科学计算和深度学习包 (torch, paddle, onnxruntime 等) 尚未发布兼容 wheel, pip 安装会失败或回退到源码编译。请使用 3.11-3.13。

**Q: 开发机的 venv 路径是什么?**

A: 开发者机器使用 `C:\Users\user\AppData\Local\hermes\hermes-agent\venv\`。全新部署请忽略此路径, 使用 `python -m venv .venv` 在项目目录内创建独立环境。

**Q: Ollama 模型 qwen3-vl:8b 有多大? 必须安装吗?**

A: 约 6.1GB。仅在启用合同自动化 LLM 抽取功能时需要。若只使用 OCR 功能, 可不安装。

**Q: Jetson 上推理速度如何?**

A: 取决于 Jetson 型号和 JetPack 版本。建议启用 TensorRT 加速, 并确保 PyTorch 为 aarch64 CUDA 版本。

---
---

# VisionOCR Pro Deployment Guide

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [Software Prerequisites](#2-software-prerequisites)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Verification](#5-verification)
6. [Troubleshooting](#6-troubleshooting)
7. [FAQ](#7-faq)

---

## 1. Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU | NVIDIA GPU, 8GB VRAM | NVIDIA GPU, 12GB+ VRAM | Tested: RTX 4070 Ti (12GB) |
| RAM | 16GB | 32GB | Model loading peaks are memory-intensive |
| Storage | SSD, 50GB free | NVMe SSD, 100GB+ | Models + Ollama weights ~10GB |
| CPU | 4 cores | 8 cores+ | Image preprocessing runs on CPU |

> Note: Without an NVIDIA GPU the system falls back to CPU inference, but latency increases significantly (~15-30s per OCR image). Not recommended for production.

## 2. Software Prerequisites

| Software | Version | Purpose | Notes |
|----------|---------|---------|-------|
| Python | 3.11 - 3.13 | Runtime | **3.14 NOT supported** (empty package ecosystem, no wheels) |
| CUDA Driver | 12.x | GPU acceleration | Driver only; no separate CUDA Toolkit needed |
| Git | 2.x+ | Source control | -- |
| Ollama | Latest | Local LLM inference | Contract automation LLM extraction engine |
| pip | Latest | Package management | Bundled with Python |

## 3. Installation

### 3.1 Windows

```bat
:: 1. Clone the repository
git clone <repo-url> visionocr-pro
cd visionocr-pro

:: 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

:: 3. Install PyTorch (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

:: 4. Install project dependencies
pip install -r requirements.txt
```

**Critical Windows notes:**

- **Do NOT install paddlepaddle-gpu.** On Windows, PaddlePaddle's bundled cudnn DLLs conflict with PyTorch, breaking torch CUDA initialization. OCR on Windows is handled by RapidOCR, which provides full functionality without PaddleOCR.
- Launch the application using `run.bat` in the project root. This script is ASCII-only and points directly to the venv Python interpreter, avoiding path and encoding issues.

```bat
:: Launch the application
run.bat
```

### 3.2 Linux / NVIDIA Jetson

```bash
# 1. Clone the repository
git clone <repo-url> visionocr-pro
cd visionocr-pro

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Install PaddlePaddle GPU (works correctly on Linux)
pip install paddlepaddle-gpu

# 6. Launch the application
python app.py
```

**Linux/Jetson notes:**

- paddlepaddle-gpu has no conflicts with PyTorch on Linux. Once installed, the PaddleOCR-VL engine becomes available for higher-accuracy layout analysis.
- For Jetson devices, ensure the correct JetPack version is flashed and the CUDA driver is compatible with PyTorch cu126.

### 3.3 Install Ollama and Model

```bash
# Install Ollama (see https://ollama.com for platform-specific installers)

# Pull the contract automation LLM model (~6.1GB)
ollama pull qwen3-vl:8b
```

This model powers the contract automation LLM extraction feature. Skip this step if contract automation is not needed.

## 4. Configuration

### 4.1 Environment Variables

```bash
# Copy the environment template
cp .env.example .env
```

Edit `.env` and fill in as needed:

- API keys (if using cloud OCR/LLM services)
- SDK paths (if deploying local SDKs)
- Other service connection details

### 4.2 Application Config

Edit `config.yaml`:

- `company_name`: Your company name (used in report output)
- `camera_type`: Camera type (select based on actual hardware)
- Adjust other business parameters as needed

## 5. Verification

### 5.1 Functional Verification

```bash
python app.py
```

1. Browser opens automatically at `http://localhost:7860`
2. Navigate to the **OCR** tab
3. Upload any image containing text
4. Expected: recognition result returned within 5 seconds

### 5.2 Unit Tests

```bash
python -m pytest tests/ -v
```

Expected: all 17 tests pass.

### 5.3 CUDA Verification

```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print your GPU model
```

## 6. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Port 7860 already in use | Stale Python process | Windows: `taskkill /F /IM python.exe`; Linux: `fuser -k 7860/tcp` |
| `torch.cuda.is_available()` returns False | CUDA driver missing or version mismatch | Install/update NVIDIA driver to a CUDA 12.x-compatible version |
| Ollama connection refused | Ollama service not running | Run `ollama serve` to start the daemon, then retry |
| Paddle DLL error on Windows | cudnn DLL conflict with torch | **Expected behavior**, no fix needed. Use RapidOCR on Windows |
| pip install timeout | Network issue | Use mirror: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| Dependency install fails on Python 3.14 | Package ecosystem not yet available | Uninstall 3.14; install Python 3.11-3.13 |

## 7. FAQ

**Q: Why can't I install paddlepaddle-gpu on Windows?**

A: PaddlePaddle ships cudnn DLLs on Windows that conflict with PyTorch's bundled cudnn, causing `torch.cuda.is_available()` to return False or runtime crashes. RapidOCR covers all OCR needs on Windows without PaddleOCR.

**Q: Why is Python 3.14 not supported?**

A: Python 3.14 is too new; major scientific computing and deep learning packages (torch, paddle, onnxruntime, etc.) have not published compatible wheels. pip installs will fail or fall back to source compilation. Use Python 3.11-3.13.

**Q: What is the developer machine's venv path?**

A: The developer machine uses `C:\Users\user\AppData\Local\hermes\hermes-agent\venv\`. For fresh deployments, ignore this path and create an isolated environment with `python -m venv .venv` inside the project directory.

**Q: How large is the Ollama model qwen3-vl:8b? Is it required?**

A: Approximately 6.1GB. It is only needed for the contract automation LLM extraction feature. If you only use OCR, you can skip it.

**Q: What inference speed can I expect on Jetson?**

A: Depends on the Jetson model and JetPack version. Enable TensorRT acceleration and ensure PyTorch is the aarch64 CUDA build for best results.
