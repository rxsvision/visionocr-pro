"""推理耗时与调用量统计 — 供状态面板与产线节拍分析

设计原则:
- 纯内存, 线程安全, 零依赖
- 不持久化 (重启清零, 避免状态污染)
- 滑动窗口均值 (最近 N 次), 避免历史拖尾掩盖当前性能
"""
from __future__ import annotations

import threading
import time
from collections import deque

_lock = threading.Lock()
# name -> {"count": int, "last_ms": float, "recent": deque[float]}
_stats: dict[str, dict] = {}

_WINDOW = 20  # 保留最近 20 次推理计算均值


def record(engine_name: str, elapsed_sec: float) -> None:
    """记录一次推理耗时。"""
    ms = max(0.0, elapsed_sec) * 1000.0
    with _lock:
        s = _stats.get(engine_name)
        if s is None:
            s = {"count": 0, "last_ms": 0.0, "recent": deque(maxlen=_WINDOW)}
            _stats[engine_name] = s
        s["count"] += 1
        s["last_ms"] = ms
        s["recent"].append(ms)


def get_stats() -> dict[str, dict]:
    """返回所有引擎的统计快照 (供状态面板)。

    Returns:
        {engine_name: {"count": int, "last_ms": float, "avg_ms": float}}
    """
    with _lock:
        out = {}
        for name, s in _stats.items():
            recent = s["recent"]
            avg = sum(recent) / len(recent) if recent else 0.0
            out[name] = {
                "count": s["count"],
                "last_ms": round(s["last_ms"], 1),
                "avg_ms": round(avg, 1),
            }
        return out


def reset() -> None:
    """清空统计 (测试/重置用)。"""
    with _lock:
        _stats.clear()


class Timer:
    """上下文管理器: 自动记录代码块耗时到指定引擎。

    用法:
        with infer_timer("rapidocr"):
            result = engine.infer(path)
    """

    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        self._t0 = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.time() - self._t0
        # 仅在推理成功时记录 (失败耗时不反映真实性能)
        if exc_type is None:
            record(self.engine_name, self.elapsed)
        return False
