"""错误恢复与降级链路 — 产线级容错

设计原则:
- 用户永远不看到 traceback, 只看到可读中文提示
- 每级降级有明确日志 (INFO级, 产线可追溯)
- 模型加载有超时保护 (产线节拍不允许无限等待)
- Docker/GPU 不可用时自动切 CPU 引擎, 不中断生产

降级链:
  PP-OCRv6 (Docker GPU) → RapidOCR (CPU)
  OvisOCR2 (GPU) → RapidOCR (CPU)
  PaddleOCR-VL (GPU) → RapidOCR (CPU)
  PatchCore (GPU) → PatchCore (CPU) → 跳过质检
  Grounding DINO (GPU) → 跳过结构检测
  Scene Classifier (GPU) → 默认 rapidocr
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("visionocr.resilience")

# ─── 降级链配置 ─────────────────────────────────────────────
# key: 主引擎, value: 按优先级排列的降级目标
DEGRADATION_CHAINS: dict[str, list[str]] = {
    "ppocrv6": ["rapidocr"],
    "ovisocr2": ["rapidocr"],
    "paddleocr_vl": ["rapidocr"],
    "hunyuan_ocr": ["rapidocr"],
    "anomalib": [],  # PatchCore 无替代, 降级为跳过
    "grounding_dino": [],  # 结构检测无替代, 降级为跳过
    "scene_classifier": [],  # 分类器降级为默认引擎
}

# 模型加载超时 (秒)
LOAD_TIMEOUTS: dict[str, float] = {
    "ppocrv6": 30.0,   # Docker 启动 + 模型加载
    "ovisocr2": 60.0,  # 大模型
    "paddleocr_vl": 60.0,
    "rapidocr": 10.0,
    "anomalib": 30.0,
    "grounding_dino": 30.0,
    "barcode": 5.0,
    "_default": 30.0,
}


class EngineLoadError(Exception):
    """引擎加载失败 (带用户可读消息)。"""

    def __init__(self, engine_name: str, reason: str, user_msg: str):
        self.engine_name = engine_name
        self.reason = reason
        self.user_msg = user_msg
        super().__init__(f"[{engine_name}] {reason}")


def safe_ensure_loaded(registry, name: str,
                       timeout: Optional[float] = None) -> tuple[Any, str]:
    """安全加载引擎, 带超时保护和降级。

    Returns:
        (engine, message) — engine 可能是降级后的替代引擎
        如果所有降级都失败, engine=None, message 包含用户可读错误

    用户永远不会看到 traceback。
    """
    if timeout is None:
        timeout = LOAD_TIMEOUTS.get(name, LOAD_TIMEOUTS["_default"])

    # 尝试加载主引擎
    engine, msg = _try_load_with_timeout(registry, name, timeout)
    if engine is not None:
        return engine, msg

    # 主引擎失败, 尝试降级链
    chain = DEGRADATION_CHAINS.get(name, [])
    for fallback_name in chain:
        fb_timeout = LOAD_TIMEOUTS.get(fallback_name, LOAD_TIMEOUTS["_default"])
        logger.info("降级: %s → %s", name, fallback_name)
        engine, fb_msg = _try_load_with_timeout(registry, fallback_name, fb_timeout)
        if engine is not None:
            user_msg = f"引擎 {name} 不可用 ({msg}), 已自动切换至 {fallback_name}"
            logger.warning("降级成功: %s → %s (原因: %s)", name, fallback_name, msg)
            return engine, user_msg

    # 所有降级都失败
    user_msg = _user_friendly_error(name, msg)
    logger.error("引擎 %s 及所有降级目标均不可用: %s", name, msg)
    return None, user_msg


def _try_load_with_timeout(registry, name: str,
                           timeout: float) -> tuple[Any, str]:
    """在指定超时内尝试加载引擎。

    Returns:
        (engine, "") 成功 / (None, error_reason) 失败
    """
    engine = registry.get(name)
    if engine is None:
        return None, f"引擎 '{name}' 未注册"

    if engine.is_ready():
        return engine, ""

    # 带超时的加载
    result = [None]
    error = [""]

    def _load():
        try:
            registry.ensure_loaded(name)
            if engine.is_ready():
                result[0] = engine
            else:
                error[0] = f"加载后状态异常: {engine.state.value}"
        except Exception as e:
            error[0] = str(e)

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None, f"加载超时 ({timeout:.0f}s)"
    if result[0] is not None:
        return result[0], ""
    return None, error[0] or "未知错误"


def _user_friendly_error(engine_name: str, reason: str) -> str:
    """将技术错误转换为用户可读的中文提示。"""
    reason_lower = reason.lower()

    if "docker" in reason_lower or "container" in reason_lower:
        return (f"⚠ {engine_name} 需要 Docker 运行环境。"
                f"请确认 Docker Desktop 已启动, 或切换至 CPU 引擎。")
    if "cuda" in reason_lower or "gpu" in reason_lower or "out of memory" in reason_lower:
        return (f"⚠ {engine_name} GPU 显存不足或 CUDA 不可用。"
                f"已尝试降级, 请关闭其他 GPU 程序后重试。")
    if "timeout" in reason_lower or "超时" in reason:
        return (f"⚠ {engine_name} 加载超时。"
                f"首次加载可能需要下载模型, 请检查网络后重试。")
    if "not registered" in reason_lower or "未注册" in reason:
        return f"⚠ 引擎 {engine_name} 未安装。请检查依赖是否完整。"
    if "import" in reason_lower or "module" in reason_lower:
        return f"⚠ {engine_name} 依赖缺失。请运行 pip install 安装所需包。"

    return f"⚠ {engine_name} 不可用: {reason}"


def check_docker_available() -> bool:
    """快速检测 Docker 是否可用 (2s 超时)。"""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def check_gpu_available() -> bool:
    """检测 CUDA GPU 是否可用。"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_system_status() -> dict:
    """获取系统运行状态 (供状态面板使用)。"""
    status = {
        "gpu_available": check_gpu_available(),
        "docker_available": check_docker_available(),
    }
    if status["gpu_available"]:
        try:
            import torch
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["gpu_vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            status["gpu_vram_used_gb"] = round(
                torch.cuda.memory_allocated(0) / 1e9, 2)
        except Exception:
            pass
    return status
