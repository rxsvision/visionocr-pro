"""Gradio 主界面 - 5 Tab 布局 + 工人/工程师模式切换"""
import gradio as gr

from core.status import format_status_markdown
from ui.tab_ocr import create_tab_ocr
from ui.tab_contract import create_tab_contract
from ui.tab_qc import create_tab_qc
from ui.tab_behavior import create_tab_behavior
from ui.tab_settings import create_tab_settings

THEME = gr.themes.Soft()
CSS = """
.main-header { text-align: center; margin-bottom: 8px; }
.status-bar { font-size: 0.85em; color: #666; }
.mode-toggle { max-width: 320px; margin: 0 auto 12px; }
.log-panel textarea {
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace !important;
    font-size: 0.82em !important;
    background: #1e1e2e !important;
    color: #a6e3a1 !important;
    border-radius: 6px;
}
"""


def create_app(config: dict, registry) -> gr.Blocks:
    with gr.Blocks(title="VisionOCR Pro") as app:
        gr.Markdown("# VisionOCR Pro", elem_classes="main-header")
        gr.Markdown(
            "通用视觉识别与检测平台 · 精度第一 · 全离线",
            elem_classes="status-bar",
        )

        # ─── 全局模式切换 ─────────────────────────────────────
        mode_toggle = gr.Radio(
            choices=["工人模式", "工程师模式"],
            value="工人模式",
            label="操作模式",
            info="工人模式: 一键操作, 隐藏调参 | 工程师模式: 完整控制",
            elem_classes="mode-toggle",
        )

        # ─── 运行状态面板 (工程师模式可见) ────────────────────
        with gr.Accordion(
            "运行状态 · 引擎 / GPU / 耗时", open=False, visible=False
        ) as status_panel:
            status_card = gr.Markdown("加载中 ...")

        def refresh_status():
            try:
                return format_status_markdown(registry)
            except Exception as e:  # pragma: no cover - 状态面板不应阻断UI
                return f"⚠ 状态采集失败: {e}"

        # 每 5 秒自动刷新状态卡片
        status_timer = gr.Timer(value=5.0)
        status_timer.tick(fn=refresh_status, outputs=[status_card])

        with gr.Tabs():
            with gr.Tab("OCR 识别", id="ocr"):
                create_tab_ocr(config, registry, mode_toggle)
            with gr.Tab("工业质检", id="qc"):
                create_tab_qc(config, registry, mode_toggle)
            with gr.Tab("合同自动化", id="contract", visible=False) as tab_contract:
                create_tab_contract(config, registry)
            with gr.Tab("行为分析", id="behavior", visible=False) as tab_behavior:
                create_tab_behavior(config, registry)
            with gr.Tab("设置", id="settings", visible=False) as tab_settings:
                create_tab_settings(config, registry)

        # ─── 模式切换 → Tab 可见性 + 状态面板 ─────────────────
        def _on_mode_change(mode: str):
            is_eng = (mode == "工程师模式")
            status_md = refresh_status() if is_eng else "加载中 ..."
            return (
                gr.update(visible=is_eng),  # contract
                gr.update(visible=is_eng),  # behavior
                gr.update(visible=is_eng),  # settings
                gr.update(visible=is_eng),  # status_panel
                status_md,                  # status_card
            )

        mode_toggle.change(
            fn=_on_mode_change,
            inputs=[mode_toggle],
            outputs=[tab_contract, tab_behavior, tab_settings,
                     status_panel, status_card],
        )

    return app
