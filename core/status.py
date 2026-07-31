"""运行状态聚合 — 引擎健康 / GPU 占用 / 推理耗时 / 后台预热

供 Gradio 状态卡片与排障使用。所有函数只读, 不改变系统状态。
"""
from __future__ import annotations

import logging

from core import infer_stats
from core.resilience import get_system_status

logger = logging.getLogger("visionocr.status")

# 引擎状态 → 中文标签 + 标记符号 (状态卡片用)
_STATE_LABELS = {
    "ready": ("就绪", "🟢"),
    "loading": ("加载中", "🟡"),
    "unloaded": ("未加载", "⚪"),
    "error": ("异常", "🔴"),
}


def collect_status(registry) -> dict:
    """聚合完整运行状态快照。

    Returns:
        {
          "gpu": {...},                  # GPU/Docker 可用性 + 显存
          "engines": [ {name, display_name, category, state, state_label,
                        vram_gb, count, avg_ms, last_ms} ],
          "summary": {"ready": int, "loading": int, "error": int,
                      "unloaded": int, "total": int,
                      "budget_gb": float, "used_gb": float},
          "background": {...},           # 后台预热状态
        }
    """
    # GPU / Docker
    try:
        gpu = get_system_status()
    except Exception as e:  # pragma: no cover - 防御性
        logger.debug("get_system_status 失败: %s", e)
        gpu = {"gpu_available": False, "docker_available": False}

    # 引擎列表 + 耗时统计
    stats = infer_stats.get_stats()
    engines = []
    counts = {"ready": 0, "loading": 0, "error": 0, "unloaded": 0}
    for eng in registry.list_engines():
        state = eng.get("state", "unloaded")
        label, icon = _STATE_LABELS.get(state, (state, "⚪"))
        counts[state] = counts.get(state, 0) + 1
        st = stats.get(eng["name"], {})
        engines.append({
            "name": eng["name"],
            "display_name": eng.get("display_name", eng["name"]),
            "category": eng.get("category", ""),
            "state": state,
            "state_label": f"{icon} {label}",
            "vram_gb": eng.get("vram_gb", 0.0),
            "count": st.get("count", 0),
            "avg_ms": st.get("avg_ms", 0.0),
            "last_ms": st.get("last_ms", 0.0),
        })

    # 显存预算
    reg_status = registry.status()

    # 后台预热
    try:
        from core.warmup import get_background_status
        background = get_background_status()
    except Exception:  # pragma: no cover
        background = {}

    return {
        "gpu": gpu,
        "engines": engines,
        "summary": {
            **counts,
            "total": len(engines),
            "budget_gb": reg_status.get("max_budget_gb", 0.0),
            "used_gb": reg_status.get("used_gb", 0.0),
        },
        "background": background,
    }


def format_status_markdown(registry) -> str:
    """渲染状态卡片为 Markdown (Gradio 显示用)。"""
    data = collect_status(registry)
    gpu = data["gpu"]
    s = data["summary"]

    lines = []

    # ── 系统行 ──
    gpu_txt = (f"🟢 {gpu.get('gpu_name', 'GPU')}"
               if gpu.get("gpu_available") else "🔴 GPU 不可用")
    if gpu.get("gpu_available") and gpu.get("gpu_vram_total_gb"):
        gpu_txt += f" · 显存 {gpu.get('gpu_vram_used_gb', 0)}/{gpu['gpu_vram_total_gb']} GB"
    docker_txt = "🟢 Docker" if gpu.get("docker_available") else "⚪ Docker 未运行"
    budget_txt = f"预算 {s['used_gb']}/{s['budget_gb']} GB"
    lines.append(f"**{gpu_txt}** | {docker_txt} | {budget_txt}")
    lines.append("")

    # ── 引擎统计行 ──
    lines.append(
        f"引擎: 🟢 就绪 {s['ready']} · 🟡 加载 {s['loading']} · "
        f"🔴 异常 {s['error']} · ⚪ 未加载 {s['unloaded']} (共 {s['total']})"
    )
    lines.append("")

    # ── 已就绪 / 有流量的引擎明细表 ──
    active = [e for e in data["engines"]
              if e["state"] == "ready" or e["count"] > 0]
    if active:
        lines.append("| 引擎 | 状态 | 调用 | 均耗时 | 显存 |")
        lines.append("|---|---|---|---|---|")
        for e in active:
            avg = f"{e['avg_ms']:.0f} ms" if e["count"] else "—"
            vram = f"{e['vram_gb']} GB" if e["vram_gb"] else "—"
            lines.append(
                f"| {e['display_name']} | {e['state_label']} | "
                f"{e['count']} | {avg} | {vram} |"
            )
        lines.append("")

    # ── 后台预热 ──
    bg = data["background"]
    if bg:
        bg_parts = [f"{k}: {v}" for k, v in bg.items()]
        lines.append(f"后台预热: {' · '.join(bg_parts)}")

    return "\n".join(lines)
