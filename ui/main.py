"""Gradio 主界面 - 5 Tab 布局"""
import gradio as gr

from ui.tab_ocr import create_tab_ocr
from ui.tab_contract import create_tab_contract
from ui.tab_qc import create_tab_qc
from ui.tab_behavior import create_tab_behavior
from ui.tab_settings import create_tab_settings

THEME = gr.themes.Soft()
CSS = """
.main-header { text-align: center; margin-bottom: 8px; }
.status-bar { font-size: 0.85em; color: #666; }
"""


def create_app(config: dict, registry) -> gr.Blocks:
    with gr.Blocks(title="VisionOCR Pro") as app:
        gr.Markdown("# VisionOCR Pro", elem_classes="main-header")
        gr.Markdown(
            "通用视觉识别与检测平台 · 精度第一 · 全离线",
            elem_classes="status-bar",
        )

        with gr.Tabs():
            with gr.Tab("OCR 识别", id="ocr"):
                create_tab_ocr(config, registry)
            with gr.Tab("合同自动化", id="contract"):
                create_tab_contract(config, registry)
            with gr.Tab("工业质检", id="qc"):
                create_tab_qc(config, registry)
            with gr.Tab("行为分析", id="behavior"):
                create_tab_behavior(config, registry)
            with gr.Tab("设置", id="settings"):
                create_tab_settings(config, registry)

    return app
