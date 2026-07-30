# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
