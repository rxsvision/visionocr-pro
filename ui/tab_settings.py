"""设置 Tab"""
import gradio as gr


def create_tab_settings(config: dict, registry):
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 引擎状态")
            status_btn = gr.Button("刷新引擎状态")
            engine_table = gr.Dataframe(
                headers=["引擎", "类别", "状态", "显存(GB)", "许可"],
                label="已注册引擎",
            )

        with gr.Column():
            gr.Markdown("### 显存管理")
            vram_display = gr.JSON(label="显存使用")
            unload_btn = gr.Button("卸载全部引擎")

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

    status_btn.click(fn=_refresh_status, outputs=[engine_table, vram_display])
    unload_btn.click(fn=_unload_all, outputs=[vram_display])


def _refresh_status(registry_ref=None):
    # 通过闭包获取 registry
    return _get_registry_status()


def _unload_all():
    return {"message": "已卸载全部引擎 (占位)"}


# 模块级 registry 引用 (由 main 注入)
_registry = None


def set_registry(reg):
    global _registry
    _registry = reg


def _get_registry_status():
    global _registry
    if _registry is None:
        return [], {"error": "registry not initialized"}
    engines = _registry.list_engines()
    table = [
        [e["display_name"], e["category"], e["state"], e["vram_gb"], e["license"]]
        for e in engines
    ]
    return table, _registry.status()
