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

# 缓存最近一次 3D 深度帧 (DepthFrame), 供"一键检测"阶段做深度融合判定。
# 工人流程: 选"3D深度相机"采集 -> 深度帧存入此处 -> 检测时自动融合。
_last_depth_frame = None


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
            image_source = gr.Radio(
                choices=["上传图像", "2D彩色相机", "3D深度相机 (Sizector)"],
                value="上传图像",
                label="图像来源",
            )
            qc_image = gr.Image(label="拍照 / 上传待检图像", type="filepath")
            depth_preview = gr.Image(
                label="3D 深度图预览 (伪彩, 仅深度相机)",
                visible=False, height=200,
            )
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
            gr.Markdown("### 3D 深度融合 (结构光)")
            fusion_enable = gr.Checkbox(
                label="启用 3D 深度融合 (深度几何 + 2D 纹理联合判定)",
                value=True,
            )
            depth_threshold = gr.Slider(
                0.1, 3.0, value=0.5, step=0.1,
                label="深度偏差阈值 mm (越小越敏感, 按件公差设定)",
            )
            fusion_mode_radio = gr.Radio(
                choices=["OR (高召回, 推荐)", "AND (高精确)", "仅深度"],
                value="OR (高召回, 推荐)",
                label="融合判定策略",
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
                headers=["#", "缺陷类型", "置信度", "位置 (x1,y1,x2,y2)"],
                label="检测明细 (编号对应图上标注)",
                wrap=True,
            )
            status_msg = gr.Markdown("")

    # ─── 事件绑定 ────────────────────────────────────────────
    detect_btn.click(
        fn=_run_detect,
        inputs=[qc_image, prompt_input, threshold_slider, detect_mode, pc_product,
                fusion_enable, depth_threshold, fusion_mode_radio],
        outputs=[result_image, verdict_box, score_box, count_box,
                 detail_table, status_msg],
    )
    camera_btn.click(
        fn=_camera_capture,
        inputs=[image_source],
        outputs=[qc_image, depth_preview, status_msg],
    )
    image_source.change(
        fn=_on_source_change,
        inputs=[image_source],
        outputs=[depth_preview],
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
def _run_detect(image_path, prompt, threshold, mode, pc_product,
                fusion_enable=False, depth_threshold=0.5, fusion_mode="OR (高召回, 推荐)"):
    """一键检测: 根据模式调用 Grounding DINO / PatchCore / 3D 深度融合。"""
    if not image_path:
        return (None, "—", "—", "—", [],
                "⚠ 请先上传图片或使用相机采集。")

    registry = _registry
    if registry is None:
        return (None, "ERROR", "—", "—", [],
                "⚠ 引擎未初始化, 请重启应用。")

    # ─── 3D 深度融合模式 (深度相机已采集 + 开关开启) ─────────
    global _last_depth_frame
    if fusion_enable and _last_depth_frame is not None and "PatchCore" not in (mode or ""):
        return _run_fusion_detect(registry, image_path, prompt, threshold,
                                  depth_threshold, fusion_mode)

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

        table = [["1", "异常热力图", f"{score:.4f}", "见标注图"]]
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
    for idx, det in enumerate(result["detections"], 1):
        box = det["box"]
        table.append([
            str(idx),
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


def _camera_capture(image_source):
    """按图像来源采集一帧。返回 (待检图路径, 深度预览图, 状态)。"""
    global _last_depth_frame
    source = image_source or "上传图像"

    # ─── 3D 深度相机 (Sizector) ─────────────────────────────
    if "3D" in source:
        try:
            cfg = _get_config()
            import cv2
            from core.sizector_camera import create_depth_camera
            cam = create_depth_camera(cfg)
            if not cam.open():
                _last_depth_frame = None
                return None, None, "⚠ 深度相机打开失败, 请检查 USB3.0 连接 / SDK 配置 (或开启 sizector.mock)"
            frame = cam.capture()
            cam.close()
            if frame is None:
                _last_depth_frame = None
                return None, None, "⚠ 深度采集失败, 请检查曝光 / 工作距离"

            _last_depth_frame = frame
            tmp_dir = Path(tempfile.gettempdir())

            # 待检图: 优先 RGB, 否则灰度, 再否则深度伪彩
            if frame.rgb is not None:
                bgr = frame.rgb
            elif frame.gray is not None:
                bgr = cv2.cvtColor(frame.gray, cv2.COLOR_GRAY2BGR)
            else:
                bgr = frame.depth_colormap()
            img_path = tmp_dir / "visionocr_qc_3d.png"
            cv2.imwrite(str(img_path), bgr)

            depth_vis = frame.depth_colormap()
            info = (f"📷 3D 采集成功 · {frame.width}x{frame.height} · "
                    f"Z=[{frame.z_min:.2f}, {frame.z_max:.2f}]mm · "
                    f"有效率 {frame.valid_ratio:.0%}")
            return str(img_path), depth_vis, info
        except Exception as e:  # noqa: BLE001
            _last_depth_frame = None
            return None, None, f"⚠ 深度相机异常: {e}"

    # ─── 2D 彩色相机 (海康等) ───────────────────────────────
    if "2D" in source:
        try:
            cfg = _get_config()
            import cv2
            from core.camera import create_camera
            cam = create_camera(cfg)
            if cam.open():
                bgr = cam.grab()
                cam.close()
                if bgr is not None:
                    _last_depth_frame = None  # 2D 来源清空深度帧
                    tmp = Path(tempfile.gettempdir()) / "visionocr_qc_capture.png"
                    cv2.imwrite(str(tmp), bgr)
                    return str(tmp), None, "📷 2D 采集成功"
            return None, None, "⚠ 相机打开失败, 请检查连接和 MVS 配置"
        except Exception as e:  # noqa: BLE001
            return None, None, f"⚠ 相机异常: {e}"

    # ─── 上传图像 ───────────────────────────────────────────
    _last_depth_frame = None
    return None, None, "ℹ 请在上方组件直接上传待检图像。"


def _on_source_change(image_source):
    """切换图像来源时控制深度预览组件显隐。"""
    visible = bool(image_source and "3D" in image_source)
    return gr.update(visible=visible)


def _run_fusion_detect(registry, image_path, prompt, threshold,
                       depth_threshold, fusion_mode_label):
    """3D 深度融合检测: 2D 纹理 (Grounding DINO) + 深度几何联合判定。"""
    import cv2
    from core.defect_detector import run_detection
    from core.depth_fusion import fuse_detection, annotate_depth

    frame = _last_depth_frame
    if frame is None:
        return (None, "ERROR", "—", "—", [],
                "⚠ 无深度帧, 请先用 3D 深度相机采集。")

    # 融合策略文案 -> 内部枚举
    if "AND" in (fusion_mode_label or ""):
        fusion_mode = "and"
    elif "仅深度" in (fusion_mode_label or ""):
        fusion_mode = "depth_only"
    else:
        fusion_mode = "or"

    # 先跑 2D 检测 (仅深度模式时跳过以省时)
    result_2d = None
    if fusion_mode != "depth_only":
        result_2d = run_detection(registry, image_path, prompt=prompt,
                                  threshold=threshold)

    fused = fuse_detection(frame, result_2d,
                           depth_threshold_mm=depth_threshold,
                           fusion_mode=fusion_mode)

    verdict = fused["verdict"]
    count = fused["count"]
    annotated = annotate_depth(frame, fused)

    if verdict == "OK":
        verdict_str = "✓ OK (合格)"
        score_str = "—"
    else:
        max_conf = max((d["confidence"] for d in fused["fused_defects"]), default=0)
        verdict_str = f"✗ NG (不合格 · {count}处)"
        score_str = f"{max_conf:.2f}"

    # 明细表
    table = []
    for idx, d in enumerate(fused["fused_defects"], 1):
        x1, y1, x2, y2 = d["bbox"]
        table.append([
            str(idx),
            f"{d['source']} · {d['type']}",
            f"{d['confidence']:.0%}",
            f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})",
        ])

    # 落库
    try:
        cfg = _get_config()
        conn = get_conn(cfg.get("data_dir", "data"))
        save_qc_result(conn, image_path, verdict, fused["fused_defects"],
                       float(score_str) if score_str != "—" else 0.0,
                       f"[3D融合] {prompt}")
        conn.close()
    except Exception:  # noqa: BLE001
        pass  # 落库失败不阻断检测

    status = (f"3D 深度融合 · 策略 {fusion_mode.upper()} · "
              f"深度阈值 {depth_threshold}mm · {fused['reason']}")
    return (annotated, verdict_str, score_str, str(count), table, status)


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
