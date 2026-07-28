"""LLM 路由器 - 本地 Ollama 优先, 云端 API 降级

对外暴露 get_llm(registry) -> 可用引擎 (带 .chat 接口) 或 None。
"""
from __future__ import annotations

from typing import Any, Optional


def get_llm(registry, config: Optional[dict] = None) -> Optional[Any]:
    """按 config.llm.provider 选择并加载 LLM, 失败则尝试另一路径。

    返回具备 .chat(messages) 与 .is_ready() 的引擎实例, 全部失败返回 None。
    """
    config = config or getattr(registry, "config", {}) or {}
    preferred = config.get("llm", {}).get("provider", "ollama")

    order = ["ollama_vlm", "api_vlm"] if preferred == "ollama" else ["api_vlm", "ollama_vlm"]

    for name in order:
        engine = registry.get(name)
        if engine is None:
            continue
        try:
            loaded = registry.ensure_loaded(name)
            if loaded.is_ready():
                return loaded
        except Exception as e:  # noqa: BLE001
            print(f"[LLM Router] {name} 加载失败: {e}")
    return None
