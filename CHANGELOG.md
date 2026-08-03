# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **环境自检 doctor** (`scripts/doctor.py`): Python 版本/解释器位置/核心与重依赖导入/config.yaml 加载/模型目录/Ollama 一键体检；核心依赖或 config 失败报 FAIL (exit 1)，重依赖缺失仅 WARN 降级；setup.bat 与 setup.sh 安装末尾自动运行；检查项含 sklearn
- **新产品接入指南** (`docs/new_product_onboarding.md`): OK 库登记五步法（≥50 张 OK 采集 → PatchCore+DINOv2 双建库 → NP 校准 ε → NG 回归 Recall=100% → 双条件放行），含 NG-only 数据集不可验收、弱标签不可信、降 GDINO 阈值追召回不可行等实测误区

### Changed

- **setup.bat / setup.sh**: 已有 `.venv` 时直接复用、跳过 PATH 扫描（避免命中无关解释器）；全新安装时打印命中解释器完整路径
- **requirements.txt**: 显式声明 `paddlepaddle>=3.0`（CPU 版 Windows/macOS/Linux 通用，本机实测驱动 PP-OCRv6 主引擎）；修正"paddlepaddle Windows 不可用"过时注释（仅 `-gpu` 变体在 Windows 与 torch cudnn 冲突）；补声明 `scikit-learn>=1.3`（dinov2_anomaly 的 PCA/GMM 直接依赖，此前仅经外部继承环境隐式存在）
- **README / DEPLOY**: 新增 doctor 用法与"随插拔式"部署须知（裸 python 陷阱、禁止经 `.pth` 继承外部应用 venv）；paddle 相关表述与 requirements.txt 对齐

### Notes

- `.venv` 要求完全自包含：不继承任何外部应用（其他工具/Agent）的 site-packages。外部应用重装会静默摧毁经 `.pth` 继承的依赖（真实事故案例），`setup.bat` 创建的 venv 无此耦合

## [1.2.1] - 2026-08-03

### Added

- **质检图片持久化** (`persist_qc_image`): 检测结果落库前把图片复制到 `data/qc_images/`（内容 sha1 哈希命名，同图自动去重），看板图片直链不再因 Gradio 临时文件清理而 404；源文件缺失/复制失败时降级用原路径，不阻断检测
- **NP 校准小样本守卫**: 校准样本 n<100 时发出粒度警告（阈值被单个次序统计量钉死，实际 FPR 调节步长 ≈1/n），提示为产品登记更多 OK 样本；不阻断拟合，有限样本保证仍成立
- 测试 116 → 123：新增图片持久化 5 例（含 Gradio 临时清理端到端回归）与 NP 小样本警告 2 例

## [1.2.0] - 2026-08-03

### Added

- **NP 校准层** (`core/np_calibration.py`): split-conformal 异常阈值，正常件误报率统计可控（`np_epsilon` 配置）；npz 持久化，旧 bank 文件向后兼容（无校准器时回落 legacy P99×1.2）
- **DINOv2 异常检测引擎** (`engines/vision/dinov2_anomaly.py`): ViT-S/14 (Apache-2.0) + PCA64 白化 + GMM 分布建模 + NP 校准，Union 零漏检第 4 源；与 PatchCore 特征互补降漏检（KolektorSDD holdout AUROC 0.9668，排序显著优于 PatchCore）；`data/banks_dinov2/` 独立建库，best-effort 不阻塞 PatchCore
- **Phase 3 VLM 智能 ROI 裁切解释** (`core/roi_selector.py` + `core/vlm_explain.py`): Union NG 后 UI「AI 解释」按钮触发，热力图连通域 / 检测框 / 整图兜底三路径裁切候选区 → 本地 VLM 局部识读；实测同一缺陷图整图解释"无缺陷"而 ROI 裁切后正确回答"划痕"
- **Phase 4 Datasette 质检看板** (`core/qc_dashboard.py` + `dashboard/qc_image_plugin.py` + `scripts/qc_dashboard.py`): 日统计 + NG 明细视图（含缺陷摘要与图片链接）、图片路由（200/404/413 分支），`pip install datasette` 后 `python scripts/qc_dashboard.py` 一键启动
- **验收评估工具链** (`scripts/eval_acceptance.py` 等 6 脚本): kolektor/pcb/yolo/paired/bootstrap 五模式，argv 驱动零硬编码路径；真实图集验收结论 P1 AUROC 0.968 / P2 YOLO Recall 100% / P3 bootstrap 0.977
- **OllamaEngine 部署友好性**: `OLLAMA_HOST` 环境变量支持（标准 Ollama 约定，可重定向备用实例）；大图下采样保护 `MAX_VLM_SIDE=1568`（15000×4096 线扫原样 base64 会产生 ~240MB 负载挂死服务）
- 测试 75 → 116：新增 np_calibration / dinov2_anomaly / roi_selector / vlm_explain / ollama_provider / qc_dashboard 六个测试模块

### Changed

- **np_epsilon 出厂默认 0.02 → 0.10**（patchcore + dinov2）: 数据驱动对齐零漏检政策。KolektorSDD holdout 实测 Recall 38.5%→76.9% (PC) / 5.8%→67.3% (DV)，代价 FPR 4.3%→11.4% / 0%→2.9%；误报由人工复核兜底。零漏检实际保障链 = Union OR + 复核，非单阈值
- **PatchCore 建库可复现性**: coreset 最远点采样起点固定种子（原全局未播种导致 bank 跨运行漂移，评估不可复现）
- **Union 零漏检模式**扩为四源 OR（PatchCore + Grounding DINO + YOLO + DINOv2），配置段 `qc.union.enable_dinov2`
- 质检结果落库（`save_qc_result`）供看板消费；README 同步技术栈/结构树/管线说明

### Fixed

- **PatchCore NP 校准自匹配偏差**: 校准集改为 20% held-out（校准图不再入 bank）——原实现建库集自评，校准分数被压低导致 NP 阈值偏小、FPR 膨胀；与 DINOv2 引擎策略一致
- **datasette 中文 Windows 三处崩溃**: 插件加载用平台默认编码（插件文件改纯 ASCII）、CLI 读 metadata 用 cp936（启动注入 PYTHONUTF8=1）、表 metadata 字符串触发 500（结构化 dict + 回归断言）

## [1.1.0] - 2026-08-01

### Added

- **YOLO 少样本结构缺陷检测** (`engines/vision/yolo_defect.py`): ultralytics/YOLOv8 引擎，Union 零漏检第三检测源；GPU OOM 自动降级 CPU；中文类别名映射
- **Union 零漏检模式接线到 UI**: 工程师模式新增第三检测模式「Union 零漏检 (三源OR)」——PatchCore + Grounding DINO + YOLO 任一 NG 即 NG；明细表带源前缀，判定显示触发源，结果落库
- **YOLO 产品门控** (`core/yolo_products.py`): 权重按产品绑定 `models/yolo/{产品名}.pt`，无产品上下文或该产品未训练时 Union 自动跳过 YOLO 源，根除跨域误报（实测 PCB 权重把金属划伤误判为「鼠咬」）
- **YOLO 微调管线**: `finetune/prepare_pcb_data.py`（VOC XML → YOLO 格式，分层 train/val）+ `finetune/train_yolo.py`（ultralytics 编排）
- **条码识别引擎** (`engines/vision/barcode.py`): pyzbar/ZBar 后端，OCR Tab 自动并行检测，结果写入 meta.barcodes 供 MES/ERP
- **配置分层**: `profiles/` 目录 + `--profile` 启动参数，多产品/多产线配置切换
- **后台异步预热** (`core/warmup.py`): 次要引擎后台预加载，消除首次切换延迟
- **结构化日志 + 运行状态卡片**: JSONL 日志（`logs/visionocr.jsonl`，RotatingFileHandler）；Gradio 状态卡片（GPU/引擎/耗时，`gr.Timer` 自动刷新）
- **推理耗时统计** (`core/infer_stats.py`): 滑动窗口平均，线程安全，`Timer` 上下文管理器
- **错误恢复与降级链路** (`core/resilience.py`): 产线级容错
- **热力图审计保存**: 检测结果 PNG 持久化（产线追溯）
- **PP-OCRv6 Docker 引擎** (`engines/ocr/ppocrv6.py`): 容器隔离，规避 Windows paddle 3.x bug；主引擎 + RapidOCR 兜底
- **CI 最小测试依赖** (`requirements-test.txt`): GitHub Actions 在 ubuntu/windows × py3.11/3.12 真实跑通 pytest
- **DEPLOY.md 4.1**: YOLO 权重本机训练 + 产品门控部署段落；`scripts/validate_cross_domain.py` 跨域验证工具

### Fixed

- **CI 最小依赖缺漏**: 原 `ci.yml` 仅装 pytest+pyyaml，但 `test_barcode` 模块级 import cv2/numpy、fixture 需 pyzbar → 收集阶段即失败；现装 `requirements-test.txt`，ubuntu 补 `apt libzbar0`
- **YOLO 测试 skip 条件混淆**: 真实依赖是 ultralytics 而非「冒烟权重存在」，补 `pytest.importorskip("ultralytics")` 使轻量环境与本地行为一致
- **requirements.txt 运行时缺口**: 补 numpy/opencv-python（cv2 被 QC/相机直接 import，干净部署原会崩溃）、pyzbar
- **PatchCore 生产级优化**: GPU coreset（160s→5s）+ 有效区域裁切 + held-out 校准（P99×1.2）+ 自适应阈值持久化（save_bank/load_bank）
- **infer() grid_size 未定义**: 修复 PCB 推理崩溃

### Changed

- **requirements.txt**: numpy/opencv-python/pyzbar 入核心依赖；ultralytics 列为可选（AGPL-3.0，仅 YOLO 检测源需要）
- **README**: 测试数 17 → 45；技术栈补 YOLO/Union/条码；质检管线补「工程师模式三选一」；core/ 结构树补 infer_stats/status/resilience/anomaly_bank/yolo_products
- **DEPLOY.md**: 依赖矩阵与 requirements.txt 对齐（3.2 补 NumPy/OpenCV/pyzbar，3.3 补 ultralytics）
- **归档**: `test_ppocrv6_accuracy.py` → `scripts/eval_ppocrv6_accuracy.py`（去 test_ 前缀免 pytest 误收集）

## [1.0.2] - 2026-07-31

### Fixed

- **B-1**: `test_phase3a.py` module-level `sys.exit()` crashed pytest collection — wrapped in `main()` + `__main__` guard
- **B-2**: Overdue reminder test false failure — `check_reminders` requires `reviewed=1`; test now sets prerequisite correctly

### Added

- **docker/Dockerfile.paddleocr**: PaddleOCR-VL container for Windows Docker Desktop (WSL2 GPU passthrough)
- **Behavior tab**: "Coming soon" placeholder with YOLO-Pose + rule engine architecture roadmap

### Changed

- Behavior tab controls disabled (`interactive=False`) — clear development status instead of broken widgets
- One-click deployment scripts (`setup.bat` / `setup.sh`) and complete DEPLOY.md rewrite (v1.0.2 includes prior commit)

## [1.0.1] - 2026-07-31

### Added

- **Structured Logging**: RotatingFileHandler (10 MB x 5), 47 print-to-logger migration across 12 modules
- **Engine Warmup**: Pre-load default OCR engine before UI opens, eliminating cold-start delay for workers
- **SQLite Auto-Backup**: `backup_db()` using sqlite3 backup API on startup, 5-copy rotation in `data/backups/`
- **safe_generator Decorator**: Catches unhandled exceptions in Gradio generators, logs traceback, yields user-visible error instead of silent UI freeze
- **Worker/Engineer Mode Toggle**: Top-level Radio switch — worker mode hides all tuning params and non-essential tabs; engineer mode exposes full control
- **Config Env Var Substitution**: `${VAR:-default}` pattern in config.yaml, `.env.example` template for secrets/SDK paths
- **pytest Skeleton**: 17 unit tests (config, database, safe_yield) with shared fixtures
- **PaddleOCR Subprocess Worker**: `_paddle_worker.py` isolates paddle from torch cudnn conflict; ready for Linux/Jetson deployment
- **run.bat**: ASCII-only Windows launcher pointing to venv Python
- **DEPLOY.md**: Bilingual deployment guide with hardware/software requirements and troubleshooting

### Fixed

- **FP16 Production Blocker**: Grounding DINO defaults to FP32 (transformers 5.x `dtype` API bug workaround)
- **Stub Engine Marking**: 7 placeholder engines clearly prefixed `[stub]` in description

### Changed

- Health dashboard: real VRAM monitoring via torch.cuda, stub detection, functional unload-all
- Config normalization: hardcoded paths/credentials replaced with env var references
- PaddleOCR-VL: auto-fallback to RapidOCR on Windows (paddle 3.x DLL/PIR bugs); subprocess path for Linux

## [1.0.0] - 2026-07-30

### Added

- **OCR Tab (P1)**: Confidence threshold slider + NG interception — results below threshold are flagged "待人工复核" with per-ROI verdict table
- **Fine-tune Pipeline (P2)**: PP-OCRv6 fine-tuning toolchain — synthetic data generation, train/val split, subprocess-isolated training (avoids torch/paddle DLL conflict), CER/WER evaluation, ONNX export
- **OvisOCR2 Engine (P3)**: ATH-MaaS/OvisOCR2 integration — 1.7 GB model, structured HTML/Markdown output, ~5 GB VRAM
- **Contract Automation E2E**: Full pipeline — upload → dedup → OCR → tiered LLM extraction → direction judgment → amount reconciliation → risk scan → SQLite → review gate → reminders
- **OCR Audit Trail (M-4)**: Persistent SQLite logging of every OCR inference (image hash, engine, scene, confidence, verdict, corrections, elapsed time)
- **Dual-path Comparison (M-2)**: Composite scoring (confidence 70% + text completeness 30%) when comparing preprocessed vs original image OCR results
- **Scene Classifier Config Alignment (M-3)**: Threshold and fallback engine now read from `config.yaml` instead of hardcoded values
- **Perspective Correction**: Hough-based skew detection + quadrilateral perspective transform (conservative strategy)
- **OCR Post-processing**: Regex-based error correction pipeline
- **3D Depth Fusion (Phase 4C)**: Sizector structured-light camera integration via pythonnet — OR/AND/depth_only fusion strategies, IoU overlap confidence boost
- **PatchCore Anomaly Detection (Phase 4B)**: Self-contained WideResNet50 implementation, Coreset sampling, per-product isolation banks
- **Grounding DINO Zero-shot (Phase 4A)**: Chinese prompt auto-translation (50+ defect terms), text-driven detection without training
- **LRU VRAM Management**: Engine registry with automatic eviction to stay within 12 GB budget
- **Tiered LLM Routing**: Local Ollama → Cloud API → Regex fallback with confidence-based escalation
- **Payment Reminders**: 4-level desktop notifications (overdue / 7-day / 3-day / 1-day)
- **ERP Export Plugins**: Excel (dual-sheet), CSV (utf-8-sig), Yonyou/Kingdee connectors (stub)

### Fixed

- **C-1**: Perspective correction result was discarded — `preprocess_for_ocr` now receives the corrected path
- **C-2**: Unicode file I/O — all cv2 imread/imwrite replaced with `np.fromfile`/`imdecode` and `imencode`/`tofile`
- **C-3**: Thread safety — double-check locking in `EngineRegistry.ensure_loaded`
- **H-1**: Direction judgment supports "unknown" when our-party is unconfigured
- **H-2**: Relative date anchors — only "签订/签署/生效" resolve to absolute dates; future events (验收/交付) leave due_date empty
- **H-4**: Temp file lifecycle — UUID-based naming + tracked cleanup in generator
- **Contract regex direction bug**: `_regex_extract` now parses "X方向Y方" patterns and maps to actual party names for correct payable/receivable judgment
- **Party name noise**: `_split_parties` strips parenthetical role descriptors like "（买方）"
- **HunyuanOCR VRAM**: Hard block on < 24 GB cards instead of silent failure

### Changed

- HunyuanOCR marked as not recommended for current hardware (RTX 4070 Ti 12 GB)
- `_smart_truncate` replaces naive `text[:12000]` — section-aware truncation preserving payment/penalty paragraphs
- Scene classifier bypass threshold: 0.6 → config-driven (default 0.7)
- Scene classifier fallback engine: hardcoded "rapidocr" → config-driven (default "paddleocr_vl")
- Company name configured: 苏州锐新视科技有限公司 (aliases: 锐新视, RXS, 锐新视科技)

## [0.4.0] - 2026-07-29

### Added

- GPU acceleration stack + FP16 inference optimization
- Enhanced QC annotation — large OK/NG badge + numbered defect boxes

## [0.3.0] - 2026-07-28

### Added

- Phase 4C: Sizector 3D structured-light depth fusion detection
- Phase 4B: PatchCore few-shot anomaly detection
- Phase 4A: Grounding DINO zero-shot detection

## [0.2.0] - 2026-07-27

### Added

- Phase 3A-3F: Contract automation — receivables data model, tiered LLM routing, review gate, error panel, reminders, dashboard KPI

## [0.1.0] - 2026-07-26

### Added

- Phase 0-2 baseline: OCR engines (RapidOCR/PaddleOCR-VL), contract extraction, Ollama integration, Gradio 5-tab UI
