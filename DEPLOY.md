# VisionOCR Pro 部署指南

## 目录

1. [一键部署（推荐）](#1-一键部署推荐)
2. [硬件要求](#2-硬件要求)
3. [软件依赖矩阵](#3-软件依赖矩阵)
4. [模型分布架构](#4-模型分布架构)
   - 4.1 [YOLO 缺陷检测权重（本机训练）](#41-yolo-缺陷检测权重本机训练)
5. [手动安装](#5-手动安装)
6. [Docker 可选：Windows 运行 PaddleOCR](#6-docker-可选windows-运行-paddleocr)
7. [离线部署](#7-离线部署)
8. [配置](#8-配置)
9. [验证](#9-验证)
10. [故障排除](#10-故障排除)
11. [常见问题](#11-常见问题)

---

## 1. 一键部署（推荐）

**前提条件**：已安装 Python 3.11-3.13、Git，且网络可用。

```bash
# 克隆仓库
git clone https://github.com/rxsvision/visionocr-pro.git
cd visionocr-pro

# Windows: 双击或命令行运行
setup.bat

# Linux / Jetson:
chmod +x setup.sh && ./setup.sh
```

脚本自动完成以下全部步骤：

1. 检测 Python 版本（3.11-3.13）
2. 创建 `.venv` 虚拟环境
3. 安装 PyTorch（CUDA 12.6）+ 全部项目依赖
4. 检测 Ollama → 拉取 qwen3-vl:8b 模型（~6.1GB）
5. 下载 OvisOCR2 权重（~1.7GB）
6. 运行 pytest 验证环境完整性

完成后运行 `run.bat`（Windows）或 `source .venv/bin/activate && python app.py`（Linux）即可启动。

> 首次运行约需 15-40 分钟（取决于网速），后续运行无需重复。

---

## 2. 硬件要求

| 组件 | 最低要求 | 推荐配置 | 备注 |
|------|----------|----------|------|
| GPU | NVIDIA GPU, 8GB VRAM | RTX 4070 Ti 12GB+ | 需支持 CUDA 12.x |
| 内存 | 16GB RAM | 32GB RAM | 模型加载峰值占用高 |
| 存储 | SSD, 50GB 可用 | NVMe SSD, 100GB+ | 模型总计约 10-15GB |
| CPU | 4 核 | 8 核+ | 图像预处理使用 CPU |
| 网络 | 首次部署需联网 | — | 后续运行完全离线 |

> 无 NVIDIA GPU 时可降级为 CPU 推理，但单张 OCR 约 15-30s，不建议生产使用。

---

## 3. 软件依赖矩阵

### 3.1 必装项

| 软件 | 版本 | 用途 | 获取方式 |
|------|------|------|----------|
| Python | 3.11 - 3.13 | 运行时 | [python.org](https://www.python.org/downloads/) |
| NVIDIA 驱动 | ≥ 525（支持 CUDA 12.x） | GPU 加速 | [nvidia.com/drivers](https://www.nvidia.com/drivers/) |
| Git | 2.x+ | 代码拉取 | [git-scm.com](https://git-scm.com/) |
| Ollama | 最新版 | 本地 LLM 推理 | [ollama.com](https://ollama.com/download) |

### 3.2 自动安装项（setup 脚本处理）

| 包/模型 | 版本 | 用途 | 安装方式 |
|---------|------|------|----------|
| PyTorch | ≥ 2.0 + cu126 | 推理框架 | pip (torch index) |
| Gradio | ≥ 5.0 | Web UI | pip |
| RapidOCR | ≥ 1.3 | 轻量 OCR 引擎 | pip |
| transformers | ≥ 4.40 | Grounding DINO 加载 | pip |
| onnxruntime-gpu | ≥ 1.17 | RapidOCR 加速 | pip |
| NumPy / OpenCV | ≥ 1.24 / ≥ 4.8 | 数值与图像基础 (QC/相机) | pip |
| pyzbar | ≥ 0.1.9 | 条码识别 (Linux 需 `apt install libzbar0`) | pip |
| OvisOCR2 权重 | — | 高精度文档 OCR | download_models.py |
| qwen3-vl:8b | — | 合同 LLM 抽取 | ollama pull |
| Grounding DINO | — | 零样本缺陷检测 | transformers 自动缓存 |

### 3.3 可选项

| 软件 | 条件 | 用途 |
|------|------|------|
| Docker Desktop + WSL2 | Windows 需要 PaddleOCR 时 | 容器内运行 PaddleOCR-VL |
| PaddlePaddle GPU | 仅 Linux | PaddleOCR-VL 引擎（Linux 无冲突） |
| ultralytics | 需 YOLO 结构缺陷检测时 | YOLO 检测源（AGPL-3.0，Union 第三源，按产品门控；见 4.1） |
| datasette | 需质检结果看板时 | Datasette 质检看板（`python scripts/qc_dashboard.py`） |
| CUDA Toolkit | 仅开发/编译自定义算子时 | 运行时只需驱动，不需 Toolkit |

### 3.4 不支持项

| 项目 | 原因 |
|------|------|
| Python 3.14 | torch/paddle/onnxruntime 无 wheel，pip 安装失败 |
| PaddlePaddle GPU (Windows) | cudnn DLL 与 PyTorch 冲突，不可修复 |
| AMD / Intel GPU | 项目依赖 CUDA 生态，无 ROCm/SYCL 适配 |

---

## 4. 模型分布架构

模型权重**不**集中在仓库内，而是由各运行时工具管理。这是有意设计：

```
┌─────────────────────────────────────────────────────────────────┐
│                    模型分布架构                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  代码仓库 (visionocr-pro/)                                        │
│  └── models/                  ← 仅存放无工具托管的权重             │
│      └── ovis-ocr2/           1.7GB   (download_models.py 管理)   │
│                                                                   │
│  Ollama 运行时 (~/.ollama/models/)                                │
│  └── qwen3-vl:8b             5.8GB   (ollama pull/list 管理)     │
│                                                                   │
│  HuggingFace 缓存 (~/.cache/huggingface/hub/)                    │
│  ├── grounding-dino-base      892MB   (transformers 自动下载)     │
│  └── PP-OCRv6_medium          ~200MB  (paddle 自动下载)           │
│                                                                   │
│  PaddleOCR 缓存 (~/.paddleocr/)                                  │
│  └── legacy models            41MB    (paddleocr 自动下载)        │
│                                                                   │
│  pip 包内嵌 (site-packages/)                                      │
│  └── RapidOCR ONNX            ~50MB   (pip install 自带)          │
│                                                                   │
│  代码内置                                                          │
│  └── PatchCore WideResNet50   ~100MB  (首次运行 torchvision 下载) │
│                                                                   │
│  本机训练产物 (models/yolo/)                                      │
│  └── YOLO11 缺陷检测权重       ~24MB/产品 (train_yolo.py 训练+命名)│
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│  总计: ~9GB    设计原则: 各工具管各自模型, 仓库只存代码+非托管权重  │
└─────────────────────────────────────────────────────────────────┘
```

**为什么不合并到一个文件夹？**

- Ollama 只认 `~/.ollama/models`，移走即失效
- transformers 默认查 HF cache，移走需每次设环境变量
- 模型更新（如 qwen3-vl 升级）由各工具自动管理版本去重
- 仓库保持 <5MB 代码 + 1.7GB 非托管权重，克隆/备份成本低

### 4.1 YOLO 缺陷检测权重（本机训练）

YOLO 结构缺陷检测权重（`best.pt`，~24MB）**既不随仓库分发，也不从网络下载**——它是针对特定产品标注数据训练的产物，必须在本机生成：

```bash
# 1. 数据准备：VOC XML 标注 → YOLO 格式（自动分层划分 train/val）
python finetune/prepare_pcb_data.py --src "<PCB_DATASET路径>"

# 2. 训练：COCO 预训练权重微调（RTX 4070 Ti 约 15 分钟，早停）
#    默认 YOLO11n 基线；--model yolov8n/s/m/x 兼容旧版权重
python finetune/train_yolo.py --epochs 120 --batch 8 --imgsz 1280
```

权重输出到 `finetune/output_yolo/pcb_defect/weights/best.pt`。

**产品门控（Union 检测）**：Union 检测中的 YOLO 源**按产品激活**——引擎在 `models/yolo/{产品名}.pt` 查找当前产品的专属权重，找到才参与检测，否则自动跳过（由 PatchCore + DINO 兜底）。这是跨域误报防护：YOLO 只检测训练集标注的缺陷类别，跨产品会大量误报（实测 PCB 权重把金属划伤误判为「鼠咬」，单图 5+ 假框）。

因此训练完成后，需将权重按产品命名放入门控目录：

```bash
mkdir models\yolo
copy finetune\output_yolo\pcb_defect\weights\best.pt models\yolo\PCB.pt
```

之后在质检界面选择产品「PCB」时 YOLO 源才会激活；选择其他产品（或未训练的产品）时 YOLO 源自动禁用。

**引擎独立加载（非 Union 路径）**：若不经过 Union 而直接加载引擎，权重按以下优先级发现：

1. `config.yaml` 的 `yolo_defect.weights`（显式指定，缺失则报错）
2. `finetune/output_yolo/pcb_defect/weights/best.pt`
3. `models/yolo_defect.pt`

**无权重时的行为**：`yolo_defect` 引擎进入 error 状态并静默跳过，Union 检测仍由 PatchCore + Grounding DINO 兜底，不影响其他功能。切换产品必须用该产品标注数据重新训练并放入 `models/yolo/{产品名}.pt`，或设 `qc.union.enable_yolo: false` 全局关闭该检测源。

---

## 5. 手动安装

如果一键脚本不适用（如网络受限、自定义路径），可手动执行：

### 5.1 Windows

```bat
:: 1. 克隆
git clone https://github.com/rxsvision/visionocr-pro.git
cd visionocr-pro

:: 2. 虚拟环境
python -m venv .venv
.venv\Scripts\activate

:: 3. PyTorch (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

:: 4. 项目依赖
pip install -r requirements.txt

:: 5. Ollama (从 https://ollama.com/download 下载安装)
ollama pull qwen3-vl:8b

:: 6. OCR 模型
python scripts/download_models.py ovisocr2

:: 7. 启动
run.bat
```

**Windows 重要说明：**

- **不要安装 paddlepaddle-gpu。** cudnn DLL 与 PyTorch 冲突，不可修复。
- Windows 安装 **CPU 版 `paddlepaddle`**（requirements.txt 已声明），驱动 PP-OCRv6 主 OCR 引擎；RapidOCR 为轻量兜底。
- 如需 PaddleOCR-VL，参见 [第 6 节 Docker 方案](#6-docker-可选windows-运行-paddleocr)。

### 5.2 Linux / Jetson

```bash
# 1-4 同上
git clone https://github.com/rxsvision/visionocr-pro.git && cd visionocr-pro
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 5. PaddlePaddle GPU (Linux 无冲突)
pip install paddlepaddle-gpu

# 6. Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-vl:8b

# 7. OCR 模型
python scripts/download_models.py ovisocr2

# 8. 启动
python app.py
```

---

## 6. Docker 可选：Windows 运行 PaddleOCR

Windows 原生不支持 PaddlePaddle GPU（DLL 冲突），但可通过 Docker Desktop + WSL2 GPU 直通在容器内运行。

### 前提

- Docker Desktop 4.x（WSL2 后端）
- NVIDIA 驱动 ≥ 525
- Docker Desktop 设置中启用 GPU 支持

### 使用方式（v1.3.0+：常驻容器服务，引擎自动管理）

```bash
# 构建镜像（含常驻 HTTP 服务 paddle_server.py）
docker build -f docker/Dockerfile.paddleocr -t visionocr-paddleocr .
```

构建完成后**无需手动启动容器**：在 UI 中选择 PP-OCRv6 引擎时，
`PPOCRv6Engine` 会自动拉起常驻容器（`visionocr-paddle-serve`，
端口 `ocr.ppocrv6.port`，默认 8686），模型加载一次常驻内存，
推理响应亚秒级；引擎卸载或应用退出时自动清理容器。

手动调试入口：

```bash
# 健康检查
curl http://127.0.0.1:8686/health

# 单次推理 (multipart 上传图像)
curl -F "file=@test.png" http://127.0.0.1:8686/ocr

# 旧单次调用模式（兼容保留，每次 5~13s 容器开销，仅降级场景）
docker run --rm --gpus all -v "D:\images:/data" visionocr-paddleocr /data/test.png
```

> PP-OCRv6 为高精度 OCR 插件（93.3% 精确匹配）；默认引擎为 RapidOCR（纯本地）。
> 镜像过旧（未含 paddle_server.py）时引擎自动降级为旧单次调用模式并提示重建镜像。

---

## 7. 离线部署

适用于工厂内网等无外网环境。

### 7.1 在有网机器上准备离线包

```bash
# 1. 导出 pip 包
pip download -r requirements.txt -d offline_packages/
pip download torch torchvision --index-url https://download.pytorch.org/whl/cu126 -d offline_packages/

# 2. 导出 Ollama 模型
# Ollama 模型位于 ~/.ollama/models/，直接复制整个目录

# 3. 导出 HF 缓存
# 位于 ~/.cache/huggingface/hub/，直接复制

# 4. 导出 repo models/
# 位于 visionocr-pro/models/，直接复制
```

### 7.2 在目标机器上还原

```bash
# 1. 复制代码仓库（U盘/内网 git）
# 2. 创建 venv + 离线安装
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install --no-index --find-links=offline_packages/ -r requirements.txt

# 3. 还原 Ollama 模型 → 复制到 ~/.ollama/models/
# 4. 还原 HF 缓存 → 复制到 ~/.cache/huggingface/hub/
# 5. 还原 repo models/ → 复制到 visionocr-pro/models/

# 6. 启动
run.bat  # 或 python app.py
```

---

## 8. 配置

### 8.1 环境变量

```bash
cp .env.example .env
```

按需填入 API 密钥（云端 LLM 兜底）、SDK 路径等。纯本地使用无需修改。

### 8.2 应用配置 (config.yaml)

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `company.name` | 苏州锐新视科技有限公司 | 合同方向判定依据 |
| `company.aliases` | [锐新视, RXS] | 公司别名 |
| `llm.ollama.model` | qwen3-vl:8b | 本地 LLM 模型 |
| `ocr.confidence_threshold` | 0.75 | OCR 置信度拦截阈值 |
| `ocr.default_engine` | auto | 默认 OCR 引擎 |
| `camera.type` | opencv | 相机类型 |

---

## 9. 验证

### 9.1 快速验证

```bash
# 启动应用
run.bat  # Windows
# python app.py  # Linux

# 浏览器打开 http://localhost:7860
# OCR Tab → 上传任意含文字图片 → 5秒内返回结果
```

### 9.2 单元测试

```bash
.venv\Scripts\python -m pytest tests/ -v   # Windows
# .venv/bin/python -m pytest tests/ -v     # Linux
```

预期：17 个测试全部通过。

### 9.3 CUDA 验证

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # GPU 型号
```

### 9.4 Ollama 验证

```bash
ollama list              # 应显示 qwen3-vl:8b
ollama run qwen3-vl:8b "hello"   # 应返回响应
```

---

## 10. 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 端口 7860 被占用 | 残留进程 | Win: `taskkill /F /IM python.exe`; Linux: `fuser -k 7860/tcp` |
| CUDA 不可用 | 驱动过旧 | 更新 NVIDIA 驱动至 ≥ 525 |
| Ollama 连接失败 | 服务未启动 | `ollama serve` 或重启 Ollama Desktop |
| paddle DLL 报错 (Win) | 预期行为 | 忽略，使用 RapidOCR |
| pip 超时 | 网络问题 | `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| Python 3.14 安装失败 | 不支持 | 卸载 3.14，安装 3.11-3.13 |
| setup.bat 找不到 Python | 未加入 PATH | 重新安装 Python 勾选 "Add to PATH" |
| HF 模型下载慢 | 国内网络 | 脚本已自动使用 hf-mirror.com |

---

## 11. 常见问题

**Q: 一键脚本需要管理员权限吗？**

A: 不需要。所有操作在用户目录内完成（venv、pip、ollama 均为用户级安装）。

**Q: 仓库文件夹为什么有 1.6GB？**

A: `models/ovis-ocr2` 权重文件（已 gitignore，不推送到 GitHub）。代码本身 <5MB。

**Q: 能否把所有模型移到仓库内？**

A: 不建议。Ollama/transformers 各自管理模型路径，强行移动会破坏自动更新和版本管理机制。参见[第 4 节](#4-模型分布架构)。

**Q: 换电脑怎么迁移？**

A: 新电脑运行 `setup.bat` / `setup.sh` 即可重建全部环境（需联网）。或参见[第 7 节离线部署](#7-离线部署)。

**Q: Docker 方案稳定吗？**

A: Docker Desktop + WSL2 GPU 直通在 NVIDIA 驱动 ≥ 525 环境下稳定。但当前 RapidOCR 已满足 Windows OCR 需求，Docker 为可选增强。

**Q: 为什么不支持 Python 3.14？**

A: torch/paddle/onnxruntime 等核心包尚未发布 3.14 兼容 wheel。请使用 3.11-3.13。

**Q: qwen3-vl:8b 必须安装吗？**

A: 仅合同自动化 LLM 抽取需要。纯 OCR 功能无需安装，可跳过节省 6GB。

---
---

# VisionOCR Pro Deployment Guide (English)

## Table of Contents

1. [One-Click Setup (Recommended)](#1-one-click-setup-recommended)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Software Dependency Matrix](#3-software-dependency-matrix)
4. [Model Distribution Architecture](#4-model-distribution-architecture)
5. [Manual Installation](#5-manual-installation)
6. [Docker Option: PaddleOCR on Windows](#6-docker-option-paddleocr-on-windows)
7. [Offline Deployment](#7-offline-deployment)
8. [Configuration](#8-configuration)
9. [Verification](#9-verification)
10. [Troubleshooting](#10-troubleshooting)
11. [FAQ](#11-faq)

---

## 1. One-Click Setup (Recommended)

**Prerequisites**: Python 3.11-3.13, Git, and internet access.

```bash
git clone https://github.com/rxsvision/visionocr-pro.git
cd visionocr-pro

# Windows: double-click or run in terminal
setup.bat

# Linux / Jetson:
chmod +x setup.sh && ./setup.sh
```

The script automatically handles:

1. Python version detection (3.11-3.13)
2. Virtual environment creation (`.venv`)
3. PyTorch (CUDA 12.6) + all project dependencies
4. Ollama detection → qwen3-vl:8b model pull (~6.1GB)
5. OvisOCR2 weight download (~1.7GB)
6. pytest verification

After completion, run `run.bat` (Windows) or `source .venv/bin/activate && python app.py` (Linux).

> First run takes ~15-40 minutes depending on network speed. Subsequent launches are instant.

---

## 2. Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU | NVIDIA, 8GB VRAM | RTX 4070 Ti 12GB+ | Must support CUDA 12.x |
| RAM | 16GB | 32GB | Model loading peaks are high |
| Storage | SSD, 50GB free | NVMe, 100GB+ | Total models ~10-15GB |
| CPU | 4 cores | 8 cores+ | Image preprocessing on CPU |
| Network | Required for first setup | — | Fully offline after setup |

> Without NVIDIA GPU, falls back to CPU inference (~15-30s/image). Not recommended for production.

---

## 3. Software Dependency Matrix

### 3.1 Required (install manually)

| Software | Version | Purpose | Download |
|----------|---------|---------|----------|
| Python | 3.11 - 3.13 | Runtime | [python.org](https://www.python.org/downloads/) |
| NVIDIA Driver | ≥ 525 (CUDA 12.x) | GPU acceleration | [nvidia.com/drivers](https://www.nvidia.com/drivers/) |
| Git | 2.x+ | Source control | [git-scm.com](https://git-scm.com/) |
| Ollama | Latest | Local LLM inference | [ollama.com](https://ollama.com/download) |

### 3.2 Auto-installed (handled by setup script)

| Package/Model | Version | Purpose | Method |
|---------------|---------|---------|--------|
| PyTorch | ≥ 2.0 + cu126 | Inference framework | pip (torch index) |
| Gradio | ≥ 5.0 | Web UI | pip |
| RapidOCR | ≥ 1.3 | Lightweight OCR | pip |
| transformers | ≥ 4.40 | Grounding DINO loading | pip |
| onnxruntime-gpu | ≥ 1.17 | RapidOCR acceleration | pip |
| OvisOCR2 weights | — | High-accuracy doc OCR | download_models.py |
| qwen3-vl:8b | — | Contract LLM extraction | ollama pull |
| Grounding DINO | — | Zero-shot detection | transformers auto-cache |

### 3.3 Optional

| Software | When | Purpose |
|----------|------|---------|
| Docker Desktop + WSL2 | PaddleOCR needed on Windows | Containerized PaddleOCR-VL |
| PaddlePaddle GPU | Linux only | PaddleOCR-VL engine (no conflict on Linux) |
| datasette | QC results dashboard needed | Datasette QC dashboard (`python scripts/qc_dashboard.py`) |
| CUDA Toolkit | Custom operator compilation | Runtime only needs driver, not Toolkit |

### 3.4 Not Supported

| Item | Reason |
|------|--------|
| Python 3.14 | No wheels for torch/paddle/onnxruntime |
| PaddlePaddle GPU (Windows) | cudnn DLL conflict with PyTorch, unfixable |
| AMD / Intel GPU | Project depends on CUDA ecosystem |

---

## 4. Model Distribution Architecture

Model weights are **not** centralized in the repository. Each runtime manages its own:

```
Code repository (visionocr-pro/)
└── models/                  ← Only unmanaged weights
    └── ovis-ocr2/           1.7GB   (download_models.py)

Ollama runtime (~/.ollama/models/)
└── qwen3-vl:8b             5.8GB   (ollama pull/list)

HuggingFace cache (~/.cache/huggingface/hub/)
├── grounding-dino-base      892MB   (transformers auto)
└── PP-OCRv6_medium          ~200MB  (paddle auto)

PaddleOCR cache (~/.paddleocr/)
└── legacy models            41MB    (paddleocr auto)

pip embedded (site-packages/)
└── RapidOCR ONNX            ~50MB   (pip install)

Code built-in
└── PatchCore WideResNet50   ~100MB  (torchvision first-run)

Total: ~9GB
```

**Why not merge into one folder?** Each tool (Ollama, transformers) only recognizes its own path. Moving models breaks auto-update and version deduplication. The repo stays lightweight (<5MB code + 1.7GB unmanaged weights).

---

## 5. Manual Installation

### 5.1 Windows

```bat
git clone https://github.com/rxsvision/visionocr-pro.git && cd visionocr-pro
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
:: Install Ollama from https://ollama.com/download
ollama pull qwen3-vl:8b
python scripts/download_models.py ovisocr2
run.bat
```

**Do NOT install paddlepaddle-gpu on Windows** (cudnn DLL conflict with PyTorch). Windows uses the **CPU build `paddlepaddle`** (declared in requirements.txt) to drive the PP-OCRv6 primary OCR engine; RapidOCR is the lightweight fallback. For PaddleOCR-VL, see [Docker option](#6-docker-option-paddleocr-on-windows).

### 5.2 Linux / Jetson

```bash
git clone https://github.com/rxsvision/visionocr-pro.git && cd visionocr-pro
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install paddlepaddle-gpu  # Linux: no conflict
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-vl:8b
python scripts/download_models.py ovisocr2
python app.py
```

---

## 6. Docker Option: PaddleOCR on Windows

Windows cannot run PaddlePaddle GPU natively (DLL conflict). Docker Desktop + WSL2 GPU passthrough provides a workaround.

### Prerequisites

- Docker Desktop 4.x (WSL2 backend)
- NVIDIA Driver ≥ 525
- GPU support enabled in Docker Desktop settings

### Usage (v1.3.0+: resident container service, engine-managed)

```bash
# Build the image (includes the resident HTTP service paddle_server.py)
docker build -f docker/Dockerfile.paddleocr -t visionocr-paddleocr .
```

After building, **no manual container startup is needed**: when you select the
PP-OCRv6 engine in the UI, `PPOCRv6Engine` automatically launches a resident
container (`visionocr-paddle-serve`, port `ocr.ppocrv6.port`, default 8686).
The model loads once and stays in memory; inference responds in sub-second
time. The container is cleaned up on engine unload / app exit.

Manual debugging endpoints:

```bash
# Health check
curl http://127.0.0.1:8686/health

# Single inference (multipart image upload)
curl -F "file=@test.png" http://127.0.0.1:8686/ocr

# Legacy one-shot mode (kept for fallback, 5~13s container overhead per call)
docker run --rm --gpus all -v "D:\images:/data" visionocr-paddleocr /data/test.png
```

> PP-OCRv6 is the high-accuracy OCR plugin (93.3% exact match); the default engine is RapidOCR (pure local).
> With a stale image (missing paddle_server.py), the engine auto-degrades to legacy one-shot mode and prompts for a rebuild.

---

## 7. Offline Deployment

For air-gapped factory networks.

### 7.1 Prepare on internet-connected machine

```bash
pip download -r requirements.txt -d offline_packages/
pip download torch torchvision --index-url https://download.pytorch.org/whl/cu126 -d offline_packages/
# Copy: ~/.ollama/models/, ~/.cache/huggingface/hub/, visionocr-pro/models/
```

### 7.2 Restore on target machine

```bash
python -m venv .venv && .venv\Scripts\activate
pip install --no-index --find-links=offline_packages/ -r requirements.txt
# Restore Ollama models → ~/.ollama/models/
# Restore HF cache → ~/.cache/huggingface/hub/
# Restore repo models → visionocr-pro/models/
run.bat
```

---

## 8. Configuration

```bash
cp .env.example .env   # API keys (optional, for cloud fallback)
```

Edit `config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `company.name` | — | Your company name (contract direction judgment) |
| `llm.ollama.model` | qwen3-vl:8b | Local LLM model |
| `ocr.confidence_threshold` | 0.75 | OCR confidence gate |
| `ocr.default_engine` | auto | Default OCR engine |
| `camera.type` | opencv | Camera type |

---

## 9. Verification

```bash
# App launch
run.bat  # → http://localhost:7860

# Unit tests
.venv\Scripts\python -m pytest tests/ -v   # 17 tests pass

# CUDA
python -c "import torch; print(torch.cuda.is_available())"  # True

# Ollama
ollama list   # shows qwen3-vl:8b
```

---

## 10. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Port 7860 in use | Stale process | `taskkill /F /IM python.exe` or `fuser -k 7860/tcp` |
| CUDA unavailable | Driver too old | Update NVIDIA driver ≥ 525 |
| Ollama refused | Service not running | `ollama serve` or restart Ollama |
| Paddle DLL error (Win) | Expected | Ignore; use RapidOCR |
| pip timeout | Network | `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| Python 3.14 fails | Unsupported | Install 3.11-3.13 |
| HF download slow | China network | Script auto-uses hf-mirror.com |

---

## 11. FAQ

**Q: Does setup require admin privileges?**

A: No. All operations are user-level (venv, pip, ollama).

**Q: Why is the repo folder 1.6GB?**

A: `models/ovis-ocr2` weights (gitignored, not pushed to GitHub). Code itself is <5MB.

**Q: Can I move all models into the repo?**

A: Not recommended. Ollama/transformers manage their own paths. See [Section 4](#4-model-distribution-architecture).

**Q: How to migrate to a new computer?**

A: Run `setup.bat`/`setup.sh` on the new machine (requires internet). Or see [Offline Deployment](#7-offline-deployment).

**Q: Is qwen3-vl:8b required?**

A: Only for contract automation LLM extraction. Pure OCR works without it (saves 6GB).
