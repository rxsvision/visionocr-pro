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
from core.anomaly_bank import (
    list_banks, bank_exists, delete_bank,
    register_ok_samples, run_anomaly_detection,
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
            gr.Markdown("### 检测模式")
            detect_mode = gr.Radio(
                choices=["零样本 (Grounding DINO)", "少样本 (PatchCore)"],
                value="零样本 (Grounding DINO)",
                label="检测模式",
            )

            gr.Markdown("### 检测配置")

            recipe_choice = gr.Dropdown(
                label="产品配方 (快速切换)",
                choices=["(自定义)"] + list_recipes(),
                value="(自定义)",
            )
            prompt_input = gr.Textbox(
                label="缺陷提示词 (中文或英文, 点号分隔)",
                value=DEFAULT_PROMPT,
                placeholder="划痕.凹陷.裂纹.污渍.毛刺.色差 (自动翻译为英文)",
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

            gr.Markdown("---")
            gr.Markdown("### 少样本注册 (PatchCore)")
            pc_product = gr.Dropdown(
                label="产品特征库",
                choices=["(新建)"] + list_banks(),
                value="(新建)",
            )
            pc_product_name = gr.Textbox(
                label="新产品名称 (新建时填写)",
                placeholder="如: PCB板_型号A",
            )
            pc_ok_upload = gr.File(
                label="上传 OK 样本 (10~30张合格品图片)",
                file_count="multiple",
                file_types=[".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
            )
            pc_register_btn = gr.Button("📦 注册建库", variant="secondary")
            pc_status = gr.Markdown("")

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
        inputs=[qc_image, prompt_input, threshold_slider, detect_mode, pc_product],
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
    pc_register_btn.click(
        fn=_register_bank,
        inputs=[pc_product, pc_product_name, pc_ok_upload],
        outputs=[pc_status, pc_product],
    )


# ─── 回调函数 ────────────────────────────────────────────────
def _run_detect(image_path, prompt, threshold, mode, pc_product):
    """一键检测: 根据模式调用 Grounding DINO 或 PatchCore。"""
    if not image_path:
        return (None, "—", "—", "—", [],
                "⚠ 请先上传图片或使用相机采集。")

    registry = _registry
    if registry is None:
        return (None, "ERROR", "—", "—", [],
                "⚠ 引擎未初始化, 请重启应用。")

    # ─── PatchCore 少样本模式 ─────────────────────────────
    if "PatchCore" in (mode or ""):
        product = "" if pc_product == "(新建)" else (pc_product or "")
        result = run_anomaly_detection(registry, image_path,
                                       product_name=product,
                                       threshold=threshold)
        if result.get("error"):
            return (None, "ERROR", "—", "—", [],
                    f"⚠ {result['error']}")

        verdict = result["pred_label"]
        score = result.get("score", 0)
        overlay = result.get("heatmap_overlay")

        if verdict == "OK":
            verdict_str = "✓ OK (合格)"
        else:
            verdict_str = f"✗ NG (异常分数 {score:.3f})"

        table = [["异常热力图", f"{score:.4f}", "见标注图"]]
        status = f"PatchCore 检测 · 产品: {product or '默认'} · 阈值: {threshold}"
        return (overlay, verdict_str, f"{score:.4f}", "1" if verdict == "NG" else "0",
                table, status)

    # ─── Grounding DINO 零样本模式 (默认) ─────────────────
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


def _register_bank(pc_product, pc_product_name, files):
    """注册 OK 样本, 构建 PatchCore 特征库。"""
    # 确定产品名
    if pc_product and pc_product != "(新建)":
        product = pc_product
    elif pc_product_name and pc_product_name.strip():
        product = pc_product_name.strip()
    else:
        return "⚠ 请选择已有产品或输入新产品名称。", gr.update()

    if not files:
        return "⚠ 请上传 OK 样本图片 (建议 10~30 张合格品)。", gr.update()

    # 收集有效路径
    paths = []
    for f in files:
        p = f.name if hasattr(f, "name") else str(f)
        if os.path.isfile(p):
            paths.append(p)

    if len(paths) < 3:
        return f"⚠ 有效图片仅 {len(paths)} 张, 建议至少 10 张。", gr.update()

    registry = _registry
    if registry is None:
        return "⚠ 引擎未初始化。", gr.update()

    result = register_ok_samples(registry, product, paths)
    if result.get("error"):
        return f"⚠ 建库失败: {result['error']}", gr.update()

    banks = ["(新建)"] + list_banks()
    msg = (f"✓ 产品「{product}」特征库已建立\n\n"
           f"- OK 样本: {result.get('n_images', 0)} 张\n"
           f"- 特征库大小: {result.get('bank_size', 0)} patches\n"
           f"- 保存位置: {result.get('saved_to', '')}")
    return msg, gr.update(choices=banks, value=product)
