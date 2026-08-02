# VisionOCR Pro

**通用视觉识别与工业检测平台** | Universal OCR & Industrial Vision Detection Platform

[English](#english) | [中文](#中文)

---

## 中文

### 产品价值

VisionOCR Pro 是面向制造业的一站式视觉智能平台，将 OCR 文字识别、合同自动化管理、工业缺陷检测三大核心场景整合在一个本地优先的 Web 应用中。

核心优势：

- **全离线可运行** — 所有推理在本地 GPU 完成，无需联网，满足工厂数据安全要求
- **多引擎自动路由** — 场景分类器自动选择最优 OCR 引擎，无需人工干预
- **分级 LLM 抽取** — 本地 Ollama 优先、云端 API 兜底、规则引擎保底，三级容错
- **傻瓜式操作** — 工人一键拍照即可获得 OK/NG 判定，无需调参
- **工人/工程师双模式** — 顶部 Toggle 切换，工人模式隐藏所有调参，工程师模式完整控制
- **生产级质量管控** — 置信度阈值拦截、审计日志持久化、金额勾稽校验、启动自动备份数据库

### 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| UI | Gradio 6.x | 本地 Web 应用，5 Tab 布局 |
| 推理框架 | PyTorch 2.x + CUDA 12.6 | GPU 加速，FP16 推理 |
| OCR 引擎 | RapidOCR / PaddleOCR-VL / OvisOCR2 / HunyuanOCR / MinerU | 多引擎 LRU 显存管理 |
| LLM | Ollama (qwen3-vl:8b) + 云端 API (DeepSeek) | 分级路由 |
| 视觉检测 | Grounding DINO + PatchCore + YOLO + DINOv2 | 零样本/少样本/Union 零漏检四源 OR |
| 条码识别 | pyzbar (ZBar) | OCR Tab 自动并行检测 |
| 3D 融合 | Sizector 结构光 + pythonnet | 深度图 + RGB 融合检测 |
| 数据存储 | SQLite (WAL) | 合同、应收、审计日志 |
| 调度 | APScheduler | 回款提醒自动化 |
| 语言 | Python 3.11+ | 全栈 |

### 处理管线 (Pipeline)

```
┌─────────────────────────────────────────────────────────────────┐
│                        OCR 识别管线                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入图像 → 质量预检 (Laplacian) → 场景分类器 (自动路由)         │
│         → 透视纠偏 (Hough) → 图像增强 (CLAHE/USM/2x)           │
│         → OCR 推理 (双路径: 增强 vs 原图, 综合评分取优)          │
│         → 后处理纠错 (正则) → 置信度判定 (OK / 待人工复核)       │
│         → 审计日志落库 (SQLite)                                  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      合同自动化管线                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  上传合同 (PDF/JPG/PNG) → SHA256 去重 → 文档读取 (文本/OCR)     │
│         → 分级 LLM 抽取 (本地→云端→规则) → 方向判定 (应收/应付)  │
│         → 金额勾稽校验 → 风险扫描 → 落库 (contracts+receivables) │
│         → 人工复核门控 → 回款提醒 (逾期/7/3/1天四级)            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      工业质检管线                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  采集图像 (海康/Sizector) → 2D 检测 (Grounding DINO 零样本)      │
│         → 3D 深度融合 (OR/AND/depth_only 策略)                   │
│         → PatchCore 少样本异常检测 → IoU 重合提升置信度          │
│         → Union 零漏检 (四源 OR: PC+DINO+YOLO+DINOv2)            │
│         → NG 后 AI 解释 (智能 ROI 裁切 → 本地 VLM 局部识读)      │
│         → OK/NG 判定 + 标注图输出 + 结果落库                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

> **质检检测模式**（工程师模式三选一）：① 零样本 Grounding DINO（提示词驱动）｜② 少样本 PatchCore（OK 样本建库，可叠加 3D 深度融合）｜③ Union 零漏检（PatchCore + DINO + YOLO + DINOv2 四源 OR，任一 NG 即 NG；YOLO 按产品门控防跨域误报；DINOv2 与 PatchCore 特征互补降漏检；异常阈值 NP 校准，误报率统计可控）。漏检零容忍，误报由人工复核兜底。

> **质检看板**（可选）：质检结果落库后可一键启动 Datasette 看板（日统计 + NG 明细 + 缺陷图回溯）：`pip install datasette` 后 `python scripts/qc_dashboard.py --port 8901`。

### 模型依赖

模型权重分布在各运行时管理目录中，不随代码仓库分发。完整架构见 [DEPLOY.md](DEPLOY.md#4-模型分布架构)。

| 模型 | 用途 | 显存占用 | 权重大小 | 获取方式 |
|------|------|----------|----------|----------|
| qwen3-vl:8b | LLM 合同抽取 | ~6 GB | 6.1 GB | `ollama pull qwen3-vl:8b` |
| OvisOCR2 | 高精度文档 OCR | ~5 GB | 1.7 GB | `python scripts/download_models.py ovisocr2` |
| RapidOCR | 轻量 OCR 兜底 | ~0.5 GB | ~50 MB | pip 自动下载 (onnxruntime) |
| PaddleOCR-VL | 自然场景 OCR | ~4 GB | ~800 MB | pip 自动下载 (paddleocr) |
| Grounding DINO (tiny) | 零样本缺陷检测 | ~2.5 GB | 1.2 GB | transformers 自动缓存 |
| PatchCore (WideResNet50) | 少样本异常检测 | ~1.5 GB | ~100 MB | 代码内置 (ImageNet 预训练) |
| DINOv2-S/14 | 少样本异常检测 (Union 第4源) | ~1 GB | 85 MB | transformers 自动缓存 (Apache-2.0) |
| HunyuanOCR | 手写体 OCR (需 24GB+) | ~12 GB | ~12 GB | `python scripts/download_models.py hunyuan` |

> **硬件建议**: RTX 4070 Ti (12 GB) 可运行除 HunyuanOCR 外的所有引擎。HunyuanOCR 需要 24 GB+ 显存 (RTX 4090 / A5000)。

### 安装与使用

> 完整部署指南（含硬件要求、模型架构、离线部署、故障排查）见 [DEPLOY.md](DEPLOY.md)。

#### 环境要求

| 必装 | 版本 | 获取 |
|------|------|------|
| Python | 3.11 - 3.13（不支持 3.14） | [python.org](https://www.python.org/downloads/) |
| NVIDIA 驱动 | ≥ 525（CUDA 12.x） | [nvidia.com](https://www.nvidia.com/drivers/) |
| Git | 2.x+ | [git-scm.com](https://git-scm.com/) |
| Ollama | 最新版 | [ollama.com](https://ollama.com/download) |

#### 快速开始（一键部署）

```bash
# 1. 克隆仓库
git clone https://github.com/rxsvision/visionocr-pro.git
cd visionocr-pro

# 2. 一键安装（自动完成: venv → PyTorch → 依赖 → 模型 → 验证）
setup.bat          # Windows: 双击或命令行运行
# ./setup.sh       # Linux / Jetson

# 3. 启动
run.bat            # Windows
# source .venv/bin/activate && python app.py   # Linux

# 浏览器自动打开 http://localhost:7860
```

> 首次运行约 15-40 分钟（下载模型），后续启动无需重复。全程无需管理员权限。

#### 模型分布

模型权重由各运行时工具管理，不集中在仓库内（[详细说明](DEPLOY.md#4-模型分布架构)）：

| 模型 | 大小 | 存放位置 | 管理方式 |
|------|------|----------|----------|
| qwen3-vl:8b | 5.8 GB | `~/.ollama/models/` | `ollama pull` |
| OvisOCR2 | 1.7 GB | `models/ovis-ocr2/` | `download_models.py` |
| Grounding DINO | 892 MB | `~/.cache/huggingface/` | transformers 自动 |
| RapidOCR | ~50 MB | pip 包内嵌 | `pip install` 自带 |
| PatchCore | ~100 MB | 代码内置 | torchvision 首次下载 |
| DINOv2-S/14 | 85 MB | `~/.cache/huggingface/` | transformers 自动 |

#### 配置

编辑 `config.yaml`：

- `company.name` / `company.aliases` — 我方主体名称（合同方向判定依据）
- `llm.ollama.model` — 本地 LLM 模型名
- `llm.api.api_key` — 云端 API 密钥（可选兜底）
- `camera.type` — 相机类型 (hikvision / opencv / gigevision)
- `ocr.confidence_threshold` — OCR 置信度拦截阈值（默认 0.75）
- `ocr.scene_classifier.confidence_threshold` — 场景分类器旁路阈值（默认 0.7）
- `qc.patchcore.np_epsilon` / `qc.dinov2.np_epsilon` — NP 校准目标误报率（默认 0.10，零漏检取向：Recall 优先，误报人工复核兜底；调小更保守、调大召回更高）
- `qc.vlm_explain` — AI 解释开关与 ROI 参数（max_rois / pad_frac / rel_thresh 等）

#### 微调 (Fine-tune)

```bash
# 生成合成训练数据
python finetune/generate_synthetic.py --count 200 --difficulty hard

# 准备数据集 (train/val 分割)
python finetune/prepare_data.py --mode csv --input labels.csv

# 启动训练 (子进程隔离, 避免 torch/paddle DLL 冲突)
python finetune/train.py --epochs 100 --lr 0.001

# 评估
python finetune/evaluate.py --model output/best_accuracy

# 导出 ONNX
python finetune/export_onnx.py
```

### 项目结构

```
visionocr-pro/
├── app.py                  # 应用入口 (结构化日志 + 启动检查)
├── config.yaml             # 全局配置 (支持 ${ENV_VAR:-default})
├── .env.example            # 环境变量模板 (API密钥/SDK路径)
├── run.bat                 # Windows 启动器 (ASCII-only)
├── setup.bat               # Windows 一键部署脚本
├── setup.sh                # Linux 一键部署脚本
├── requirements.txt        # Python 依赖
├── DEPLOY.md               # 部署指南 (中英双语)
├── core/                   # 核心业务逻辑
│   ├── config.py           #   配置加载 + 环境变量替换
│   ├── database.py         #   SQLite 数据层 + 审计日志 + 自动备份
│   ├── warmup.py           #   引擎预热 (后台异步, 消除冷启动)
│   ├── infer_stats.py      #   推理耗时统计 (滑动窗口, 线程安全)
│   ├── status.py           #   运行状态聚合 (GPU/引擎/耗时)
│   ├── resilience.py       #   错误恢复与降级链路
│   ├── contract_extractor.py # 合同要素抽取 (LLM + 规则)
│   ├── payment_store.py    #   应收/回款数据操作
│   ├── risk_engine.py      #   合同风险扫描
│   ├── image_preprocess.py #   图像增强管线
│   ├── perspective_correct.py # 透视纠偏
│   ├── postprocess.py      #   OCR 后处理纠错
│   ├── depth_fusion.py     #   3D 深度融合
│   ├── defect_detector.py  #   缺陷检测调度 (DINO/PatchCore/DINOv2/Union)
│   ├── anomaly_bank.py     #   PatchCore/DINOv2 特征库 (按产品隔离)
│   ├── np_calibration.py   #   NP 校准 (异常阈值误报率统计可控)
│   ├── roi_selector.py     #   智能 ROI 裁切 (热力图/检测框/整图兜底)
│   ├── vlm_explain.py      #   VLM 局部识读 (ROI → 缺陷描述)
│   ├── qc_dashboard.py     #   Datasette 看板构建 (视图/metadata/启动)
│   ├── yolo_products.py    #   YOLO 权重产品门控 (防跨域误报)
│   ├── camera.py           #   海康相机封装
│   ├── sizector_camera.py  #   Sizector 3D 相机
│   ├── scheduler.py        #   定时提醒调度
│   ├── dedup.py            #   文件去重
│   ├── document_reader.py  #   PDF/图像文档读取
│   └── exporters/          #   导出插件 (Excel/CSV/ERP)
├── engines/                # 推理引擎层
│   ├── base.py             #   引擎基类 + 状态机
│   ├── registry.py         #   引擎注册表 + LRU 显存管理
│   ├── ocr/                #   OCR 引擎 (6个 + subprocess worker)
│   ├── vision/             #   视觉检测引擎 (6个)
│   ├── pose/               #   姿态/行为引擎 (3个)
│   └── llm/                #   LLM 引擎 (Ollama + API)
├── ui/                     # Gradio UI 层
│   ├── main.py             #   主布局 + 工人/工程师模式切换
│   ├── safe_yield.py       #   Generator 异常防御装饰器
│   ├── tab_ocr.py          #   OCR 识别 Tab
│   ├── tab_contract.py     #   合同自动化 Tab
│   ├── tab_qc.py           #   工业质检 Tab
│   ├── tab_behavior.py     #   行为分析 Tab (P2)
│   └── tab_settings.py     #   设置 + 引擎健康面板
├── finetune/               # 微调工具链 (PP-OCRv6 + YOLO 缺陷检测)
├── dashboard/              # Datasette 插件 (质检看板图片路由)
├── scripts/                # 辅助脚本 (评估/诊断/看板)
├── scenarios/              # 场景配置
├── tests/                  # pytest 单元测试 (116 tests)
└── models/                 # 模型权重 (gitignore, 本地存放)
```

### 版本历史

详见 [CHANGELOG.md](CHANGELOG.md)。

### 许可证

本项目采用 [BSL 1.1 (Business Source License)](LICENSE) 许可。

### 鸣谢

- [RapidOCR](https://github.com/RapidAI/RapidOCR) — 轻量 OCR 引擎
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — 百度飞桨 OCR
- [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) — ATH-MaaS 高精度文档 OCR
- [Grounding DINO](https://github.com/IDEA-Research/Grounding-DINO) — 零样本检测
- [Anomalib](https://github.com/open-edge-platform/anomalib) — 异常检测框架
- [Ollama](https://ollama.com) — 本地 LLM 运行时
- [Gradio](https://gradio.app) — Web UI 框架

---

## English

### Product Value

VisionOCR Pro is a manufacturing-oriented visual intelligence platform that unifies OCR text recognition, contract automation, and industrial defect detection in a single local-first web application.

Key advantages:

- **Fully offline** — All inference runs on local GPU; no internet required, meeting factory data-security policies
- **Multi-engine auto-routing** — A scene classifier selects the optimal OCR engine automatically
- **Tiered LLM extraction** — Local Ollama first, cloud API fallback, regex engine as last resort
- **One-click operation** — Workers get OK/NG verdicts from a single photo, no parameter tuning
- **Worker/Engineer dual mode** — Top-level toggle hides all tuning in worker mode; full control in engineer mode
- **Production-grade QA** — Confidence threshold gating, persistent audit trail, amount reconciliation, automatic DB backup on startup

### Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| UI | Gradio 6.x | Local web app, 5-tab layout |
| Inference | PyTorch 2.x + CUDA 12.6 | GPU-accelerated, FP16 |
| OCR Engines | RapidOCR / PaddleOCR-VL / OvisOCR2 / HunyuanOCR / MinerU | LRU VRAM management |
| LLM | Ollama (qwen3-vl:8b) + Cloud API (DeepSeek) | Tiered routing |
| Vision | Grounding DINO + PatchCore + YOLO + DINOv2 | Zero-shot / few-shot / Union zero-miss (4-source OR) |
| Barcode | pyzbar (ZBar) | Auto parallel detection in OCR tab |
| 3D Fusion | Sizector structured light + pythonnet | Depth + RGB fusion |
| Storage | SQLite (WAL) | Contracts, receivables, audit logs |
| Scheduling | APScheduler | Payment reminder automation |
| Language | Python 3.11+ | Full stack |

### Model Dependencies

Model weights are distributed across runtime-managed locations (not bundled in the code repo). See [DEPLOY.md](DEPLOY.md#4-model-distribution-architecture) for the full architecture.

| Model | Purpose | VRAM | Size | Acquisition |
|-------|---------|------|------|-------------|
| qwen3-vl:8b | LLM contract extraction | ~6 GB | 6.1 GB | `ollama pull qwen3-vl:8b` |
| OvisOCR2 | High-accuracy document OCR | ~5 GB | 1.7 GB | `python scripts/download_models.py ovisocr2` |
| RapidOCR | Lightweight OCR fallback | ~0.5 GB | ~50 MB | Auto via pip (onnxruntime) |
| PaddleOCR-VL | Natural scene OCR | ~4 GB | ~800 MB | Auto via pip (paddleocr) |
| Grounding DINO (tiny) | Zero-shot defect detection | ~2.5 GB | 1.2 GB | Auto via transformers cache |
| PatchCore (WideResNet50) | Few-shot anomaly detection | ~1.5 GB | ~100 MB | Built-in (ImageNet pretrained) |
| DINOv2-S/14 | Few-shot anomaly detection (Union 4th source) | ~1 GB | 85 MB | Auto via transformers cache (Apache-2.0) |
| HunyuanOCR | Handwriting OCR (requires 24 GB+) | ~12 GB | ~12 GB | `python scripts/download_models.py hunyuan` |

> **Hardware**: RTX 4070 Ti (12 GB) runs all engines except HunyuanOCR, which requires 24 GB+ VRAM (RTX 4090 / A5000).

### Quick Start (One-Click)

See [DEPLOY.md](DEPLOY.md) for full hardware/software requirements, model architecture, and offline deployment.

**Prerequisites**: Python 3.11-3.13, NVIDIA driver ≥ 525, Git, [Ollama](https://ollama.com/download).

```bash
# 1. Clone
git clone https://github.com/rxsvision/visionocr-pro.git
cd visionocr-pro

# 2. One-click setup (auto: venv → PyTorch → deps → models → verify)
setup.bat          # Windows: double-click or run in terminal
# ./setup.sh       # Linux / Jetson

# 3. Launch
run.bat            # Windows
# source .venv/bin/activate && python app.py   # Linux

# Opens http://localhost:7860
```

> First run takes ~15-40 min (model downloads). No admin privileges required.

**Model distribution**: Weights are managed by their respective runtimes, not bundled in the repo ([details](DEPLOY.md#4-model-distribution-architecture)):

| Model | Size | Location | Managed by |
|-------|------|----------|-----------|
| qwen3-vl:8b | 5.8 GB | `~/.ollama/models/` | `ollama pull` |
| OvisOCR2 | 1.7 GB | `models/ovis-ocr2/` | `download_models.py` |
| Grounding DINO | 892 MB | `~/.cache/huggingface/` | transformers auto |
| RapidOCR | ~50 MB | pip package | `pip install` |
| PatchCore | ~100 MB | Built-in | torchvision first-run |
| DINOv2-S/14 | 85 MB | `~/.cache/huggingface/` | transformers auto |

> **Windows note**: Do NOT install paddlepaddle-gpu (cudnn DLL conflict). RapidOCR covers all OCR needs. For PaddleOCR-VL, see [Docker option](DEPLOY.md#6-docker-option-paddleocr-on-windows).

### License

[BUSL 1.1 (Business Source License)](LICENSE)

---

*Built by [RXS Vision Technology](https://github.com/rxsvision) — Suzhou, China*
