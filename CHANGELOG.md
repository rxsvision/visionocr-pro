# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.1] - 2026-08-04

### Fixed

- 补齐 v1.4.0 报告中 §5.1 PixOOD 实现（`_reinit_dead_etalons`/`_fit_etalon_np_stats`）
- 补入 §5.3 `scripts/mine_prompts.py` 提示词挖掘工具链
- 同步 §5.4 YOLO11 训练基线默认值（`train_yolo.py` default=yolo11n）
- 补入 `eval_acceptance.py` 的 dvab/prompts 评估模式

### Added

- **SubspaceAD 快速换线辅助通道** (`engines/vision/subspace_ad.py`): CVPR 2026 (arXiv 2602.23013, Apache-2.0 已核验) 免训练异常检测的工程适配独立实现（不 vendor 原始代码，依 Apache-2.0 署名）。DINOv2-S/14 patch tokens 多层(-4,-5)均值聚合 → PCA 子空间（累计解释方差 τ=0.99 自动选维）→ 重构残差 → 高斯模糊后 top-1% 均值。1-4 张 OK 图触发快速换线模式（旋转增广建库），≥5 张走标准模式。与 dinov2 引擎共享权重，显存 ~1GB，`resident=False` 按需加载。新增 `tests/test_subspace_ad.py` 20 例
- **SubspaceAD 特征库管理** (`core/anomaly_bank.py`): 独立目录 `data/banks_subspacead/`，注册/加载/删除/自动发现唯一库；`run_subspace_detection` 产出热力图叠加（橙色 REVIEW 标注）+ 审计 PNG 持久化
- **QC 面板快速换线 UI** (`ui/tab_qc.py`): 检测模式新增「快速换线辅助 (SubspaceAD)」；工程师面板新增辅助通道建库区（1-4 张快速 / ≥10 张标准）；检测结果明示「仅供参考, 需人工复核」
- **防退化验收指标** (`scripts/eval_acceptance.py`): `mode_subspacead` + `_recall_at_matched_fpr` —— 阈值锚定共同 holdout 正常图的匹配 FPR 口径，杜绝自校准阈值塌陷（全判 NG）虚高 Recall 的假 PASS
- **分阶段融合决策** (`core/fusion.py`, §5.5): Union 判决按 NP 校准样本量 n_cal 分阶段收紧 —— Stage 1 (n_cal<10) 沿用纯 OR 零漏检；Stage 2 (≥10) 双源互证（≥2 校准源 NG 才判 NG，单源孤证降级 REVIEW 黄牌转人工，从不静默放行）；Stage 3 (≥50) 附加 DriftMonitor 滑动窗漂移预警（仅告警不改判决）。校准源不足 2 个时自动回退 OR（宁可误报不漏检）。KolektorSDD 验收 PASS：自主 NG FPR-hold 11.43%→2.86% (-75%)，有效 Recall 80.77% 不降，泄漏缺陷 0。UI 同步展示 REVIEW 判定与融合阶段/n_cal
- **校准协议** (`core/calibration_protocol.py`, §6.2): 解决"n_cal=3 问题"——建库期仅尾部 20% holdout 校准导致融合停留 Stage 1。建库后补采 ≥30 张独立 OK 图（建议光照/角度 3 组）逐源重标定 NP 阈值（保证不变 P(正常>τ)≤ε），重标定写回 bank，校准集存档 `data/calibration/{产品}/{时间戳}/`；可选 NG 样本做 Recall 回归；验收报告含 τ(旧→新)/阶段变化/小样本提示。工程师面板新增「📐 校准协议」入口，建库反馈明示 n_cal 与阶段警告。`recalibrate_engine` 为三引擎共用重标定助手。新增 `tests/test_calibration_protocol.py` 25 例
- **DINOv2 引擎 PixOOD 思想借鉴升级点** (`engines/vision/dinov2_anomaly.py`, §5.1): P1 死 etalon 重初始化（权重 < dead_weight_frac/K 的分量判死，在 top-1% NLL 高分点重新播种并以 means_init 重拟合，`qc.dinov2.reinit_dead_etalons` 默认关）；P4 per-etalon 局部 NP 归一化（逐 etalon NLL 中位数/MAD 稳健统计，按归属分量标准化后进入全局 NP 阈值，`qc.dinov2.per_etalon_np` 默认关）。完全自研实现（PixOOD 为 CC-BY-NC-SA 4.0 + Toyota 专利，仅借鉴思想，无代码/权重继承），统计量随 bank npz 持久化且旧库兼容回退。P3（MLP 二维密度估计）因需训练循环、收益未证实，推迟 v1.5。`tests/test_dinov2_anomaly.py` 新增 7 例
- **dvab A/B 评估模式** (`scripts/eval_acceptance.py dvab`): DINOv2 四变体（baseline/reinit/localnp/reinit+localnp）KolektorSDD 同口径对比，特征缓存避免 4× 骨干重复推理；预注册判决规则：Recall 降幅 ≤2pts 前提下 FPR-hold/AUROC 更优才升级，否则保持基线
- **GDDM 提示词挖掘工具链** (`scripts/mine_prompts.py` + `eval_acceptance.py prompts`, §5.3): 轻量自研实现 GDDM 思想（原版属 GS-CLIP，仓库无 LICENSE 全版权保留，仅借鉴"离群区域→提示词挖掘"思路）——缺陷 mask 连通域 → 光度学描述（极性/长宽比/紧凑度）→ 规则映射候选词，可选 `--vlm` 本地 qwen3-vl 精炼；prompts 模式做基线 vs 挖掘词集多 conf A/B，预注册工作点门控（Recall≥20% ∧ FPR-hold≤10% ∧ 平均框≤3）

### Changed

- **SubspaceAD 定位为辅助提示通道（诚实降级）**: KolektorSDD 实测 1-shot 未达 §5.2 验收门槛 —— 匹配 FPR=0.10 口径 Recall 比值 56.4% < 85%，AUROC 0.81 vs PatchCore 全库 0.89，且 4-shot (26.9%) 反低于 1-shot (42.3%) 不单调。故不参与 Union OR、不给自主 OK/NG 判定，仅分数+热力图供人工复核。旋转空角填充由黑 (fillcolor=0) 改为图像边缘均值色（A/B 实测 1-shot AUROC 0.68→0.85，黑角污染 PCA 子空间并抬高背景分）
- **YOLO 训练基线 YOLOv8 → YOLO11** (§5.4): `finetune/train_yolo.py` 默认 `yolo11n`（需 ultralytics ≥8.3，`--model yolov8n/s/m/x` 兼容加载旧版权重），引擎 meta、requirements、DEPLOY §4.1、新产品接入指南同步更新；投产路径明确为 best.pt → `models/yolo/{产品名}.pt` 产品门控。已训练的 YOLOv8 权重不受影响
- **§5.1 PixOOD 升级点 A/B 判决：保持基线默认关闭（诚实记录）**: KolektorSDD dvab 实测 —— reinit Recall 67.3%→80.8% (+13.5pts) 但 FPR-hold 2.86%→5.71%（翻倍）；localnp AUROC 0.9684≈基线 0.9668 但 Recall -9.6pts；组合变体最差（AUROC 0.892/Recall 36.5%）。无变体满足预注册判决规则（Recall 降幅 ≤2pts 且 FPR/AUROC 更优），两个升级点维持配置默认关。注：reinit 的"Recall 升/FPR 升"取舍与漏检零容忍铁律存在张力，但单源 FPR 翻倍对融合层的影响未经 Union 级验证，不作默认；Recall 优先场景可显式开启 `reinit_dead_etalons` 试用并重跑验收
- **§5.3 GDDM 提示词挖掘：探索完毕，判决砍（诚实记录）**: KolektorSDD 实测 —— 399 缺陷图挖出 52 离群区域 → 9 候选词；conf=0.2 时挖掘词集 Recall 25.0%（基线 9.6%）但 FPR-hold 12.9% 超 10% 门槛，conf≥0.3 时两组词集均零检出（与既往"GDINO 对表面缺陷无可用工作点"结论一致）。无满足预注册门控的工作点 → 不进默认配置，DEFAULT_PROMPT 不变；工具链保留（其他有 mask 标注的产品可复用挖掘+A/B）。注：FPR 仅超门槛 2.9pts 且分阶段融合下单源孤证转 REVIEW，若接受更高复检量可在产品配方中试用挖掘词 conf 0.2，此为留档选项而非默认

### Fixed

- **快速模式退化自校准 → REVIEW 契约**: 增广视图自评分数系统性偏低（KolektorSDD 实测 tau≈0.14 vs 真实正常件均值≈0.53），快速模式若给自主判定会退化为全 NG；现快速模式 `infer()` 恒返回 `pred_label="REVIEW"` + `review_required=True`，仅在累积 ≥10 张真实 OK 图切换标准模式后才给判定
- **灰度图热力图叠加崩溃** (`core/anomaly_bank.py`): KolektorSDD 等工业灰度图 (2D) 与 3D 彩色热力图 `cv2.addWeighted` 尺寸不匹配报错；SubspaceAD 与 PatchCore 两条叠加路径均加灰度→BGR 保护

### Notes

- SubspaceAD 论文 MVTec 1-shot 报 97.1 与本次 KolektorSDD AUROC 0.81 差距大，主因 KolektorSDD 类内几何变化（接插件位姿）远高于 MVTec —— 单一基准未达标即降级为辅助通道，不做单基准过拟合调参
- 测试 138 → 158 → 217 (217 passed, 4 skipped; skipped 均为 ultralytics 可选依赖未安装的 YOLO 测试)

## [1.3.0] - 2026-08-04

### Added

- **PP-OCRv6 常驻容器服务** (`docker/paddle_server.py`): 容器内 FastAPI 常驻服务，模型加载一次常驻内存，推理从每次 5~13s 容器开销降至亚秒级；宿主引擎自动管理容器（启动/健康轮询/崩溃自愈重启/unload 停止），旧镜像自动降级单次 `docker run --rm` 模式零中断；`/health` `/ocr` 端点可 curl 调试；新增 `tests/test_ppocrv6_server.py` 19 例（协议/降级决策/自愈/生命周期）
- **常驻引擎保障** (`EngineMeta.resident`): 常驻引擎不参与 LRU 显存驱逐与空闲卸载，4 个检测源（PatchCore/DINOv2/GDINO/YOLO 门控）在 12GB 预算内常驻；`registry.status()` 输出 resident 列表
- **空闲卸载后台线程**: `vram.idle_unload_sec` 配置正式生效（此前为无后台线程的死配置），空闲超时自动卸载非常驻引擎释放显存
- **环境自检 doctor** (`scripts/doctor.py`): Python 版本/解释器位置/核心与重依赖导入/config.yaml 加载/模型目录/Ollama 一键体检；核心依赖或 config 失败报 FAIL (exit 1)，重依赖缺失仅 WARN 降级；setup.bat 与 setup.sh 安装末尾自动运行；检查项含 sklearn
- **新产品接入指南** (`docs/new_product_onboarding.md`): OK 库登记五步法（≥50 张 OK 采集 → PatchCore+DINOv2 双建库 → NP 校准 ε → NG 回归 Recall=100% → 双条件放行），含 NG-only 数据集不可验收、弱标签不可信、降 GDINO 阈值追召回不可行等实测误区
- **`.dockerignore` 白名单模式**: paddleocr 镜像构建上下文从全仓库（含 GB 级 models/data）降至 ~10KB，杜绝本地图像/数据进入构建上下文
- **UI OCR 引擎选择**: 新增「PP-OCRv6 (高精度·需Docker)」明示选项，替换误导性的「PP-OCRv6 (CPU快速)」命名

### Changed

- **默认 OCR 引擎 ppocrv6 → rapidocr**: 纯本地零容器开销；PP-OCRv6 转为显式调用的高精度插件（93.3% 精确匹配场景）；推理期降级不再静默——UI 日志明示引擎切换与精度风险
- **Union 并行推理**: PatchCore + DINOv2 准备阶段串行、推理 ThreadPoolExecutor 并行，产线 4096×3000 图表面双源段 247ms→175ms (-42%)；单源就绪时退化为串行，ng_sources 顺序不变
- **Grounding DINO 本地缓存优先加载**: `local_files_only=True` 缓存命中路径全离线（消除每次加载的 HF 308 联网探测），缓存缺失回退联网下载（首次）
- **`vram.idle_unload_sec` 默认 300 → 1800**: 与常驻引擎策略配套，避免检测引擎被频繁卸载重载
- **setup.bat / setup.sh**: 已有 `.venv` 时直接复用、跳过 PATH 扫描（避免命中无关解释器）；全新安装时打印命中解释器完整路径
- **requirements.txt**: 显式声明 `paddlepaddle>=3.0`（CPU 版 Windows/macOS/Linux 通用，本机实测驱动 PP-OCRv6 主引擎）；修正"paddlepaddle Windows 不可用"过时注释（仅 `-gpu` 变体在 Windows 与 torch cudnn 冲突）；补声明 `scikit-learn>=1.3`（dinov2_anomaly 的 PCA/GMM 直接依赖，此前仅经外部继承环境隐式存在）
- **worker.py 提取 `format_ocr_result()`**: 单次/常驻两条推理路径复用同一结果格式化逻辑，输出协议不变
- **README / DEPLOY**: 新增 doctor 用法与"随插拔式"部署须知（裸 python 陷阱、禁止经 `.pth` 继承外部应用 venv）；DEPLOY 第 6 节中英双语同步常驻容器服务架构；paddle 相关表述与 requirements.txt 对齐
- 测试 123 → 138

### Fixed

- **进程退出引擎残留**: `registry.shutdown()` 原仅停空闲卸载线程、且 `app.py` 从未注册它 → PP-OCRv6 常驻容器（`--restart unless-stopped`）在应用退出乃至系统重启后仍残留运行；现 shutdown 卸载全部已加载引擎（常驻豁免仅限运行期，退出时仍释放），`app.py` 经 atexit 注册
- **特征库自动发现**: 重启后无产品上下文时 PatchCore/DINOv2 特征库不加载 → 缺陷全部判 OK 的严重问题；唯一 bank 自动加载、多 bank 明确告警；所有检测源被跳过时 verdict=OK 不可信告警
- **onnxruntime-gpu 安装残缺**: CPU/GPU 包共存导致共享文件损坏（`no attribute '__version__'`），以 `--force-reinstall --no-deps onnxruntime-gpu==1.28.0` 修复并验证 CUDA/TensorRT provider 可用

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
