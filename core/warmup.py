"""引擎预热 - 启动时加载默认引擎并执行 dummy 推理, 消除首次操作延迟

设计原则:
- 必要引擎 (OCR 主引擎) 在 app.launch() 之前同步加载, 浏览器打开时即可用
- 次要引擎 (场景分类/条码/质检) 后台异步加载, 不阻塞启动
- 预热失败不阻断启动, 降级为按需加载模式
- 产线开机后第一张图不让工人等
"""
from __future__ import annotations

import logging
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("visionocr.warmup")

# dummy 图像路径 (延迟创建)
_DUMMY_PATH: str | None = None

# 后台预热状态 (供状态面板查询)
_background_status: dict[str, str] = {}  # {engine_name: "loading"/"ready"/"failed"}


def _get_dummy_path() -> str:
    """生成一张 256x256 灰度噪声 PNG 作为预热输入。"""
    global _DUMMY_PATH
    if _DUMMY_PATH is None or not Path(_DUMMY_PATH).exists():
        from PIL import Image
        rng = np.random.default_rng(42)
        arr = rng.integers(60, 200, (256, 256), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="warmup_")
        img.save(tmp.name)
        _DUMMY_PATH = tmp.name
    return _DUMMY_PATH


def warmup_engines(registry, config: dict) -> dict:
    """同步预热必要引擎 + 启动后台异步预热次要引擎。

    策略:
    - [同步] 预热 OCR default_engine (工人最高频, 必须启动即可用)
    - [同步] 若主引擎失败, 降级预热 fallback_engine
    - [异步] 后台预加载: scene_classifier, barcode, anomalib (不阻塞启动)

    Returns:
        {"ocr": {...}, "background": [...], "total_sec": float}
    """
    report = {"ocr": {}, "background": [], "total_sec": 0}
    t_start = time.time()

    # ─── 同步: 必要 OCR 引擎 ─────────────────────────────────
    ocr_cfg = config.get("ocr", {}) or {}
    default_engine = ocr_cfg.get("default_engine", "rapidocr")
    fallback_engine = ocr_cfg.get("fallback_engine", "rapidocr")

    if default_engine == "auto":
        default_engine = "rapidocr"

    ok = _warmup_one(registry, default_engine, report)

    if not ok and fallback_engine != default_engine:
        logger.info("主引擎 %s 预热失败, 降级预热 %s",
                    default_engine, fallback_engine)
        _warmup_one(registry, fallback_engine, report)

    # ─── 异步: 次要引擎后台预加载 ─────────────────────────────
    secondary_engines = _get_secondary_engines(config, default_engine)
    if secondary_engines:
        report["background"] = secondary_engines
        _start_background_warmup(registry, secondary_engines)

    report["total_sec"] = round(time.time() - t_start, 2)
    return report


def get_background_status() -> dict[str, str]:
    """获取后台预热状态 (供状态面板)。"""
    return dict(_background_status)


def _get_secondary_engines(config: dict, primary: str) -> list[str]:
    """根据配置确定需要后台预加载的次要引擎列表。"""
    secondary = []

    # 场景分类器 (OCR 自动路由依赖)
    sc_cfg = (config.get("ocr", {}) or {}).get("scene_classifier", {})
    if sc_cfg.get("enabled", True):
        secondary.append("scene_classifier")

    # 条码 (轻量, 加载快)
    secondary.append("barcode")

    # PatchCore (如果 QC 功能启用)
    qc_cfg = config.get("qc", {}) or {}
    if qc_cfg.get("patchcore", {}).get("input_size"):
        secondary.append("anomalib")

    # 排除已同步加载的主引擎
    secondary = [e for e in secondary if e != primary]
    return secondary


def _start_background_warmup(registry, engines: list[str]) -> None:
    """在后台线程中逐个预加载次要引擎。"""
    def _bg_load():
        for name in engines:
            _background_status[name] = "loading"
            try:
                t0 = time.time()
                engine = registry.ensure_loaded(name)
                elapsed = time.time() - t0
                if engine.is_ready():
                    _background_status[name] = "ready"
                    logger.info("后台预热完成: %s (%.1fs)", name, elapsed)
                else:
                    _background_status[name] = "failed"
                    logger.debug("后台预热跳过: %s (状态: %s)", name, engine.state.value)
            except Exception as e:
                _background_status[name] = "failed"
                logger.debug("后台预热失败: %s (%s)", name, e)

    thread = threading.Thread(target=_bg_load, daemon=True, name="warmup-bg")
    thread.start()
    logger.info("后台预热已启动: %s", ", ".join(engines))


def _warmup_one(registry, engine_name: str, report: dict) -> bool:
    """预热单个引擎, 成功返回 True"""
    logger.info("预热引擎: %s ...", engine_name)
    try:
        t0 = time.time()
        engine = registry.ensure_loaded(engine_name)
        load_sec = time.time() - t0

        if not engine.is_ready():
            logger.warning("引擎 %s 加载失败, 跳过预热", engine_name)
            report["ocr"] = {"engine": engine_name, "ok": False,
                             "error": "load failed"}
            return False

        # dummy 推理: 触发 CUDA kernel JIT 编译 + 内存分配
        t0 = time.time()
        dummy_path = _get_dummy_path()
        result = engine.infer(dummy_path)
        infer_sec = time.time() - t0

        # 检查推理是否真正成功 (Docker 引擎可能 load 成功但 infer 报错)
        if isinstance(result, dict) and result.get("error"):
            logger.warning("引擎 %s 推理失败: %s", engine_name,
                           result["error"])
            report["ocr"] = {"engine": engine_name, "ok": False,
                             "error": result["error"]}
            return False

        report["ocr"] = {
            "engine": engine_name,
            "load_sec": round(load_sec, 2),
            "infer_sec": round(infer_sec, 2),
            "ok": True,
        }
        logger.info("预热完成: %s (加载 %.1fs + 推理 %.2fs)",
                    engine_name, load_sec, infer_sec)
        return True

    except Exception as e:
        logger.warning("预热异常 (非致命): %s: %s", engine_name, e)
        report["ocr"] = {"engine": engine_name, "ok": False, "error": str(e)}
        return False
