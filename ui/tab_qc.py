"""工业质检 Tab (Phase 4A: Grounding DINO 零样本缺陷检测)

工作流: 拍照/上传 → 选择产品配方(或手动输入缺陷词) → 一键检测 → OK/NG 判定
设计原则: 工人傻瓜式操作, 一键出结果, 无需理解模型细节。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

from core.config import load_config
from core.database import get_conn
from core.defect_detector import (
    DEFAULT_PROMPT, run_detection, save_qc_result,
    list_recipes, load_recipe, save_recipe, delete_recipe,
)

_registry = None
_config = None


def set_registry(registry) -> None:
    global _registry
    _registry = registry


def _get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def create_tab_qc(config: dict, registry):
    set_registry(registry)

    with gr.Row():
        # ─── 左栏: 输入 + 配置 ─────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 图像输入")
            qc_image = gr.Image(label="拍照 / 上传待检图像", type="filepath")
            with gr.Row():
                camera_btn = gr.Button("📷 相机采集", scale=1)
                detect_btn = gr.Button("🔍 一键检测", variant="primary", scale=2)

            gr.Markdown("---")
            gr.Markdown("### 检测配置")

            recipe_choice = gr.Dropdown(
                label="产品配方 (快速切换)",
                choices=["(自定义)"] + list_recipes(),
                value="(自定义)",
            )
            prompt_input = gr.Textbox(
                label="缺陷提示词 (英文, 点号分隔)",
                value=DEFAULT_PROMPT,
                placeholder="scratch.dent.crack.stain.burr.missing part",
                lines=2,
            )
            threshold_slider = gr.Slider(
                0.1, 0.9, value=0.3, step=0.05,
                label="置信度阈值 (越低越敏感, 推荐 0.25~0.4)",
            )

            gr.Markdown("---")
            gr.Markdown("### 配方管理")
            with gr.Row():
                recipe_name_input = gr.Textbox(
                    label="配方名称", placeholder="如: 铝合金外壳", scale=2)
                recipe_save_btn = gr.Button("保存", scale=1)
                recipe_del_btn = gr.Button("删除", scale=1)
            recipe_msg = gr.Markdown("")

        # ─── 右栏: 结果展示 ─────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 检测结果")
            result_image = gr.Image(label="标注结果 (红框=缺陷)", height=450)
            with gr.Row():
                verdict_box = gr.Textbox(
                    label="判定", scale=1,
                    interactive=False,
                )
                score_box = gr.Textbox(
                    label="最高置信度", scale=1, interactive=False)
                count_box = gr.Textbox(
                    label="缺陷数", scale=1, interactive=False)

            detail_table = gr.Dataframe(
                headers=["缺陷类型", "置信度", "位置 (x1,y1,x2,y2)"],
                label="检测明细",
                wrap=True,
            )
            status_msg = gr.Markdown("")

    # ─── 事件绑定 ────────────────────────────────────────────
    detect_btn.click(
        fn=_run_detect,
        inputs=[qc_image, prompt_input, threshold_slider],
        outputs=[result_image, verdict_box, score_box, count_box,
                 detail_table, status_msg],
    )
    camera_btn.click(
        fn=_camera_capture,
        outputs=[qc_image, status_msg],
    )
    recipe_choice.change(
        fn=_on_recipe_change,
        inputs=[recipe_choice],
        outputs=[prompt_input, threshold_slider],
    )
    recipe_save_btn.click(
        fn=_save_recipe_ui,
        inputs=[recipe_name_input, prompt_input, threshold_slider],
        outputs=[recipe_msg, recipe_choice],
    )
    recipe_del_btn.click(
        fn=_delete_recipe_ui,
        inputs=[recipe_name_input],
        outputs=[recipe_msg, recipe_choice],
    )


# ─── 回调函数 ────────────────────────────────────────────────
def _run_detect(image_path, prompt, threshold):
    """一键检测: 调用 Grounding DINO, 返回标注图 + 判定。"""
    if not image_path:
        return (None, "—", "—", "—", [],
                "⚠ 请先上传图片或使用相机采集。")

    registry = _registry
    if registry is None:
        return (None, "ERROR", "—", "—", [],
                "⚠ 引擎未初始化, 请重启应用。")

    result = run_detection(registry, image_path, prompt=prompt,
                           threshold=threshold)

    if result.get("error"):
        return (result.get("image"), f"ERROR", "—", "—", [],
                f"⚠ {result['error']}")

    verdict = result["verdict"]
    max_score = result["max_score"]
    count = result["count"]

    # 判定显示
    if verdict == "OK":
        verdict_str = "✓ OK (合格)"
    else:
        verdict_str = f"✗ NG (不合格 · {count}处缺陷)"

    # 明细表
    table = []
    for det in result["detections"]:
        box = det["box"]
        table.append([
            det["label"],
            f"{det['score']:.2%}",
            f"({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})",
        ])

    # 落库
    try:
        cfg = _get_config()
        conn = get_conn(cfg.get("data_dir", "data"))
        save_qc_result(conn, image_path, verdict, result["detections"],
                       max_score, prompt)
        conn.close()
    except Exception:
        pass  # 落库失败不阻断检测

    status = f"检测完成 · 提示词: {prompt[:60]}... · 阈值: {threshold}"
    return (result["image"], verdict_str, f"{max_score:.2%}",
            str(count), table, status)


def _camera_capture():
    """海康相机采集一帧。"""
    try:
        cfg = _get_config()
        from core.camera import create_camera
        cam = create_camera(cfg)
        if cam.open():
            frame = cam.grab()
            cam.close()
            if frame is not None:
                import cv2
                tmp = Path(tempfile.gettempdir()) / "visionocr_qc_capture.png"
                cv2.imwrite(str(tmp), frame)
                return str(tmp), "📷 采集成功"
        return None, "⚠ 相机打开失败, 请检查连接和 MVS 配置"
    except Exception as e:
        return None, f"⚠ 相机异常: {e}"


def _on_recipe_change(recipe_name):
    """切换配方时自动填充提示词和阈值。"""
    if not recipe_name or recipe_name == "(自定义)":
        return DEFAULT_PROMPT, 0.3
    recipe = load_recipe(recipe_name)
    if recipe:
        return recipe.get("prompt", DEFAULT_PROMPT), recipe.get("threshold", 0.3)
    return DEFAULT_PROMPT, 0.3


def _save_recipe_ui(name, prompt, threshold):
    """保存当前配置为产品配方。"""
    if not name.strip():
        return "⚠ 请输入配方名称。", gr.update()
    save_recipe(name.strip(), prompt, threshold)
    recipes = ["(自定义)"] + list_recipes()
    return f"✓ 配方「{name.strip()}」已保存。", gr.update(choices=recipes, value=name.strip())


def _delete_recipe_ui(name):
    """删除产品配方。"""
    if not name.strip():
        return "⚠ 请输入要删除的配方名称。", gr.update()
    if delete_recipe(name.strip()):
        recipes = ["(自定义)"] + list_recipes()
        return f"✓ 配方「{name.strip()}」已删除。", gr.update(choices=recipes, value="(自定义)")
    return f"⚠ 配方「{name.strip()}」不存在。", gr.update()
