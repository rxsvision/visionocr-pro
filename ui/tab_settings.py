"""设置 Tab - 引擎健康看板 + 系统配置"""
import logging
import time

import gradio as gr

logger = logging.getLogger("visionocr.settings")

# 模块级 registry 引用 (由 app.py 注入)
_registry = None


def set_registry(reg):
    global _registry
    _registry = reg


def create_tab_settings(config: dict, registry):
    with gr.Row():
        gr.Markdown("### 引擎健康看板")

    with gr.Row():
        health_summary = gr.Markdown(value=_build_health_summary())

    with gr.Row():
        with gr.Column(scale=3):
            status_btn = gr.Button("刷新引擎状态", variant="secondary")
            engine_table = gr.Dataframe(
                headers=["状态", "引擎", "类别", "显存(GB)", "许可", "说明"],
                label="已注册引擎",
                interactive=False,
            )

        with gr.Column(scale=1):
            gr.Markdown("### 显存管理")
            vram_display = gr.JSON(label="GPU 显存")
            unload_btn = gr.Button("卸载全部引擎", variant="stop")
            vram_refresh_btn = gr.Button("刷新显存", variant="secondary")

    with gr.Accordion("LLM 配置", open=False):
        llm_provider = gr.Radio(
            choices=["ollama", "api"],
            value=config.get("llm", {}).get("provider", "ollama"),
            label="LLM 提供者",
        )
        ollama_model = gr.Textbox(
            value=config.get("llm", {}).get("ollama", {}).get("model", "qwen3-vl:8b"),
            label="Ollama 模型",
        )
        api_base = gr.Textbox(
            value=config.get("llm", {}).get("api", {}).get("base_url", ""),
            label="API Base URL",
        )
        api_key = gr.Textbox(
            value="", label="API Key", type="password",
            placeholder="或设置环境变量 VISIONOCR_API_KEY",
        )

    with gr.Accordion("相机配置", open=False):
        cam_type = gr.Dropdown(
            choices=["opencv", "gigevision", "hikvision"],
            value=config.get("camera", {}).get("type", "opencv"),
            label="相机类型",
        )
        cam_index = gr.Number(
            value=config.get("camera", {}).get("index", 0),
            label="设备号 (OpenCV)",
            precision=0,
        )

    with gr.Accordion("系统信息", open=False):
        sys_info = gr.JSON(label="运行环境")
        sys_btn = gr.Button("检测环境", variant="secondary")

    # 事件绑定
    status_btn.click(fn=_refresh_status, outputs=[engine_table, vram_display, health_summary])
    unload_btn.click(fn=_unload_all, outputs=[vram_display, engine_table, health_summary])
    vram_refresh_btn.click(fn=_get_vram_info, outputs=[vram_display])
    sys_btn.click(fn=_get_sys_info, outputs=[sys_info])

    # 初始加载
    engine_table.value = _get_engine_table()


# ─── 内部函数 ─────────────────────────────────────────────────────────────────

_STATE_ICONS = {
    "ready": "🟢",
    "error": "🔴",
    "unloaded": "⚪",
    "loading": "🟡",
}

_STUB_NOTE = "[stub] 未接入"


def _build_health_summary() -> str:
    """构建健康摘要 Markdown。"""
    global _registry
    if _registry is None:
        return "⚠️ Registry 未初始化"

    engines = _registry.list_engines()
    ready = sum(1 for e in engines if e["state"] == "ready")
    error = sum(1 for e in engines if e["state"] == "error")
    unloaded = sum(1 for e in engines if e["state"] == "unloaded")
    total = len(engines)

    vram_str = ""
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_name = torch.cuda.get_device_name(0)
            vram_str = f" | GPU: {gpu_name} | 显存: {alloc:.1f}/{total_vram:.1f} GB"
    except ImportError:
        pass

    status = "✅ 正常" if error == 0 else f"⚠️ {error}个引擎异常"
    return (
        f"**{status}** — 共 {total} 个引擎: "
        f"🟢 {ready} 就绪 / 🔴 {error} 异常 / ⚪ {unloaded} 待命"
        f"{vram_str}"
    )


def _get_engine_table() -> list:
    """获取引擎状态表格数据。"""
    global _registry
    if _registry is None:
        return []

    engines = _registry.list_engines()
    rows = []
    for e in engines:
        state = e.get("state", "unknown")
        icon = _STATE_ICONS.get(state, "❓")
        desc = e.get("description", "")
        # 标记 stub 引擎
        if "[stub]" in desc or "未接入" in desc:
            desc = _STUB_NOTE
        rows.append([
            icon,
            e.get("display_name", e.get("name", "?")),
            e.get("category", "?"),
            f"{e.get('vram_gb', 0):.1f}",
            e.get("license", "?"),
            desc[:40],
        ])
    return rows


def _get_vram_info() -> dict:
    """获取 GPU 显存信息。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu": "未检测到 CUDA 设备", "mode": "CPU"}
        props = torch.cuda.get_device_properties(0)
        return {
            "gpu": torch.cuda.get_device_name(0),
            "total_gb": round(props.total_memory / 1e9, 1),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            "free_gb": round((props.total_memory - torch.cuda.memory_allocated()) / 1e9, 1),
        }
    except ImportError:
        return {"error": "PyTorch 未安装"}


def _get_sys_info() -> dict:
    """收集运行环境信息。"""
    import platform
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch
        info["pytorch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["pytorch"] = "未安装"

    try:
        import gradio
        info["gradio"] = gradio.__version__
    except ImportError:
        pass

    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass

    try:
        import paddleocr
        info["paddleocr"] = paddleocr.__version__
    except (ImportError, AttributeError):
        pass

    return info


def _refresh_status():
    """刷新引擎状态 + 显存 + 健康摘要。"""
    return _get_engine_table(), _get_vram_info(), _build_health_summary()


def _unload_all():
    """卸载全部已加载引擎, 释放显存。"""
    global _registry
    if _registry is None:
        return {"error": "registry 未初始化"}, [], "⚠️ Registry 未初始化"

    unloaded_count = 0
    engines = getattr(_registry, "_engines", {})
    for name, eng in engines.items():
        if hasattr(eng, "is_ready") and eng.is_ready():
            try:
                eng.unload()
                unloaded_count += 1
                logger.info("已卸载引擎: %s", name)
            except Exception as e:
                logger.warning("卸载 %s 失败: %s", name, e)

    # 释放 CUDA 缓存
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    logger.info("已卸载 %d 个引擎, 显存已释放", unloaded_count)
    return _get_vram_info(), _get_engine_table(), _build_health_summary()
