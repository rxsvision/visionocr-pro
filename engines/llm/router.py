"""LLM 分级路由器 - 本地优先, 能力不足自动升级云端

设计 (对齐"小显存 + 本地优先 + 云端兜底"):
- 本地 Ollama 优先 (机型好跑 qwen3-vl, 显存小跑小文本模型, config 驱动)。
- 云端 API (DeepSeek 等 OpenAI 兼容文本接口) 作为能力不足时的兜底。
- "能力不足"判定:
    1. 本地引擎不可用/加载失败
    2. 本地抽取置信度 < confidence_threshold
    3. 金额勾稽 / JSON 校验失败 (escalate_on_validation_fail)

对外接口:
- get_llm(registry, config): 返回首选可用引擎 (兼容旧调用)
- route_extract(registry, config, extract_fn): 分级抽取, 返回 (result, used_tier)
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def _routing_cfg(config: dict) -> dict:
    return (config.get("llm", {}) or {}).get("routing", {}) or {}


def get_llm(registry, config: Optional[dict] = None) -> Optional[Any]:
    """按路由策略返回首选可用引擎 (具备 .chat / .is_ready)。

    policy:
      local_only                 -> 仅本地
      local_first_cloud_fallback -> 本地优先, 不可用则云端
      cloud_first                -> 云端优先, 不可用则本地
    """
    config = config or getattr(registry, "config", {}) or {}
    policy = _routing_cfg(config).get("policy", "local_first_cloud_fallback")

    if policy == "local_only":
        order = ["ollama_vlm"]
    elif policy == "cloud_first":
        order = ["api_vlm", "ollama_vlm"]
    else:  # local_first_cloud_fallback
        order = ["ollama_vlm", "api_vlm"]

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


def _load_tier(registry, name: str) -> Optional[Any]:
    engine = registry.get(name)
    if engine is None:
        return None
    try:
        loaded = registry.ensure_loaded(name)
        return loaded if loaded.is_ready() else None
    except Exception as e:  # noqa: BLE001
        print(f"[LLM Router] {name} 加载失败: {e}")
        return None


def route_extract(registry, config: Optional[dict],
                  extract_fn: Callable[[Any], dict]) -> tuple[dict, str]:
    """分级抽取: 本地优先, 能力不足升级云端。

    Args:
        extract_fn: 接收一个 llm 引擎, 返回抽取结果 dict。
                    结果需含 "confidence" (0~1) 与 "valid" (bool, 校验是否通过)
                    供升级判定; 缺省时按可用处理。

    Returns:
        (result, used_tier) 其中 used_tier ∈ {"local","cloud","none"}
    """
    config = config or getattr(registry, "config", {}) or {}
    rcfg = _routing_cfg(config)
    policy = rcfg.get("policy", "local_first_cloud_fallback")
    conf_thr = float(rcfg.get("confidence_threshold", 0.6))
    escalate_on_fail = bool(rcfg.get("escalate_on_validation_fail", True))

    # 决定尝试顺序
    if policy == "cloud_first":
        tiers = [("api_vlm", "cloud"), ("ollama_vlm", "local")]
    elif policy == "local_only":
        tiers = [("ollama_vlm", "local")]
    else:
        tiers = [("ollama_vlm", "local"), ("api_vlm", "cloud")]

    last_result: dict = {}
    for name, tier in tiers:
        llm = _load_tier(registry, name)
        if llm is None:
            continue
        try:
            result = extract_fn(llm)
        except Exception as e:  # noqa: BLE001
            print(f"[LLM Router] {name} 抽取异常: {e}")
            continue
        last_result = result or {}

        conf = float(last_result.get("confidence", 1.0))
        valid = bool(last_result.get("valid", True))
        # 本地结果足够好 → 直接采用
        good = (conf >= conf_thr) and (valid or not escalate_on_fail)
        if good or tier == "cloud":
            return last_result, tier
        # 本地能力不足 → 升级云端
        print(f"[LLM Router] 本地能力不足 (conf={conf:.2f}, valid={valid}), 升级云端兜底")

    return last_result, ("none" if not last_result else "local")
