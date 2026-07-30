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

    Returns:
        {"ocr": {"engine": str, "load_sec": float, "infer_sec": float, "ok": bool},
         "total_sec": float}
    """
    report = {"ocr": {}, "total_sec": 0}
    t_start = time.time()

    # 确定默认 OCR 引擎
    ocr_cfg = config.get("ocr", {}) or {}
    default_engine = ocr_cfg.get("default_engine", "auto")

    # auto 模式: 按优先级选择
    if default_engine == "auto":
        # 优先 rapidocr (轻量, 0.5GB, 启动快), 作为基础引擎预热
        # paddleocr_vl / ovisocr2 按需加载 (大模型, 首次使用时再加载)
        default_engine = "rapidocr"

    logger.info("预热引擎: %s ...", default_engine)

    try:
        t0 = time.time()
        engine = registry.ensure_loaded(default_engine)
        load_sec = time.time() - t0

        if not engine.is_ready():
            logger.warning("引擎 %s 加载失败, 跳过预热", default_engine)
            report["ocr"] = {"engine": default_engine, "ok": False,
                             "error": "load failed"}
            return report

        # dummy 推理: 触发 CUDA kernel JIT 编译 + 内存分配
        t0 = time.time()
        dummy_path = _get_dummy_path()
        result = engine.infer(dummy_path)
        infer_sec = time.time() - t0

        report["ocr"] = {
            "engine": default_engine,
            "load_sec": round(load_sec, 2),
            "infer_sec": round(infer_sec, 2),
            "ok": True,
        }
        logger.info("预热完成: %s (加载 %.1fs + 推理 %.2fs)",
                    default_engine, load_sec, infer_sec)

    except Exception as e:
        logger.warning("预热异常 (非致命, 降级为按需加载): %s", e)
        report["ocr"] = {"engine": default_engine, "ok": False, "error": str(e)}

    report["total_sec"] = round(time.time() - t_start, 2)
    return report
