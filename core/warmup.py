"""引擎预热 - 核心检测链路优先 (定位对齐: 启动可慢, 检测要快)

设计原则:
- [同步] 核心检测引擎按 qc.union 的 enabled 开关预热, 启动完成即检测就绪
- [异步] OCR (辅助插件) + 场景分类/条码后台预热, 不阻塞启动
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
    """同步预热核心检测引擎 + 启动后台异步预热辅助引擎。

    策略 (定位对齐: 检测是核心, OCR 是辅助插件):
    - [同步] qc.union 各 enabled 源 (anomalib/dinov2/grounding_dino),
      启动完成即检测就绪; YOLO 受产品门控, 按需加载不入预热
    - [异步] 后台: OCR default_engine (失败降级 fallback_engine) +
      scene_classifier + barcode

    Returns:
        {"ok": bool, "core": {name: {...}}, "background": [...],
         "total_sec": float}
        ok 为 False 时表示至少一个核心检测源未就绪 (首检可能较慢/受限)。
    """
    report = {"ok": True, "core": {}, "background": [], "total_sec": 0}
    t_start = time.time()

    # 同步: 核心检测引擎 (按 qc.union enabled 开关)
    for name in _get_core_engines(config):
        _warmup_core_one(registry, name, report)

    # 异步: 辅助引擎后台预热 (OCR 插件 + 场景分类 + 条码)
    ocr_cfg = config.get("ocr", {}) or {}
    primary = ocr_cfg.get("default_engine", "rapidocr")
    if primary == "auto":
        primary = "rapidocr"
    background = _get_background_engines(config, primary)
    if background:
        report["background"] = background
        _start_background_warmup(
            registry, background,
            fallback=ocr_cfg.get("fallback_engine", primary))

    report["total_sec"] = round(time.time() - t_start, 2)
    return report


def get_background_status() -> dict[str, str]:
    """获取后台预热状态 (供状态面板)。"""
    return dict(_background_status)


def _get_core_engines(config: dict) -> list[str]:
    """按 qc.union 的 enabled 开关确定同步预热的核心检测源。

    YOLO (产品门控, 无专属权重不激活) 与 SubspaceAD (降级通道)
    不入预热: 前者按需 load_for_product, 后者按需加载不占常驻预算。
    """
    union_cfg = (config.get("qc", {}) or {}).get("union", {})
    core = []
    if union_cfg.get("enable_patchcore", True):
        core.append("anomalib")
    if union_cfg.get("enable_dinov2", True):
        core.append("dinov2_anomaly")
    if union_cfg.get("enable_dino", True):
        core.append("grounding_dino")
    return core


def _get_background_engines(config: dict, primary_ocr: str) -> list[str]:
    """后台预热列表: OCR 主引擎优先, 其后场景分类器与条码。"""
    engines = [primary_ocr]
    sc_cfg = (config.get("ocr", {}) or {}).get("scene_classifier", {})
    if sc_cfg.get("enabled", True):
        engines.append("scene_classifier")
    engines.append("barcode")
    return list(dict.fromkeys(engines))  # 去重保序


def _start_background_warmup(registry, engines: list[str],
                             fallback: str = "") -> None:
    """在后台线程中逐个预加载辅助引擎; OCR 主引擎失败时降级 fallback。"""
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
                    logger.debug("后台预热跳过: %s (状态: %s)",
                                 name, engine.state.value)
            except Exception as e:
                _background_status[name] = "failed"
                logger.debug("后台预热失败: %s (%s)", name, e)
            # OCR 主引擎失败 → 追加降级引擎 (同原同步降级语义, 移入后台)
            if (_background_status.get(name) == "failed"
                    and fallback and fallback != name
                    and registry.get(fallback) is not None
                    and not registry.get(fallback).is_ready()
                    and fallback not in engines):
                logger.info("OCR 主引擎 %s 预热失败, 后台降级预热 %s",
                            name, fallback)
                engines.append(fallback)

    thread = threading.Thread(target=_bg_load, daemon=True, name="warmup-bg")
    thread.start()
    logger.info("后台预热已启动: %s", ", ".join(engines))


def _warmup_core_one(registry, engine_name: str, report: dict) -> bool:
    """同步预热单个核心检测引擎, 成功返回 True。

    与 OCR 预热不同: 检测引擎只做加载 + dummy 推理预热 CUDA,
    dummy 推理的业务报错 (如 PatchCore 无记忆库) 不影响"加载就绪"判定。
    """
    logger.info("预热核心检测引擎: %s ...", engine_name)
    entry = report["core"].setdefault(engine_name, {})
    try:
        t0 = time.time()
        engine = registry.ensure_loaded(engine_name)
        entry["load_sec"] = round(time.time() - t0, 2)

        if not engine.is_ready():
            entry["ok"] = False
            entry["error"] = "load failed"
            report["ok"] = False
            logger.warning("核心引擎 %s 加载失败, 首检将受限", engine_name)
            return False

        # dummy 推理: 触发 CUDA kernel JIT 编译 (业务报错不影响预热判定)
        t0 = time.time()
        try:
            engine.infer(_get_dummy_path())
        except Exception as e:
            logger.debug("核心引擎 %s dummy 推理跳过 (预期内): %s",
                         engine_name, e)
        entry["infer_sec"] = round(time.time() - t0, 2)
        entry["ok"] = True
        logger.info("核心预热完成: %s (加载 %.1fs + 预热推理 %.2fs)",
                    engine_name, entry["load_sec"], entry["infer_sec"])
        return True

    except Exception as e:
        entry["ok"] = False
        entry["error"] = str(e)
        report["ok"] = False
        logger.warning("核心预热异常 (非致命): %s: %s", engine_name, e)
        return False
