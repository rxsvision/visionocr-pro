"""引擎预热 - 启动时加载默认引擎并执行 dummy 推理, 消除首次操作延迟

设计原则:
- 在 app.launch() 之前同步执行, 浏览器打开时系统已就绪
- 仅预热 OCR 默认引擎 (工人最高频操作), 其他引擎按需加载 (LRU)
- 预热失败不阻断启动, 降级为按需加载模式
"""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("visionocr.warmup")

# dummy 图像路径 (延迟创建)
_DUMMY_PATH: str | None = None


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
    """预热默认引擎, 返回预热报告。

    策略:
    - 尝试预热 default_engine (ppocrv6)
    - 若失败 (Docker 未运行等), 自动降级预热 fallback_engine (rapidocr)
    - 预热失败不阻断启动

    Returns:
        {"ocr": {"engine": str, "load_sec": float, "infer_sec": float, "ok": bool},
         "total_sec": float}
    """
    report = {"ocr": {}, "total_sec": 0}
    t_start = time.time()

    # 确定默认 OCR 引擎
    ocr_cfg = config.get("ocr", {}) or {}
    default_engine = ocr_cfg.get("default_engine", "rapidocr")
    fallback_engine = ocr_cfg.get("fallback_engine", "rapidocr")

    if default_engine == "auto":
        default_engine = "rapidocr"

    # 尝试预热主引擎
    ok = _warmup_one(registry, default_engine, report)

    # 主引擎失败 → 降级预热 fallback (确保启动后有可用 OCR)
    if not ok and fallback_engine != default_engine:
        logger.info("主引擎 %s 预热失败, 降级预热 %s",
                    default_engine, fallback_engine)
        _warmup_one(registry, fallback_engine, report)

    report["total_sec"] = round(time.time() - t_start, 2)
    return report


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
