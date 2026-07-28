"""OCR 通用识别 Tab - 三引擎自动路由"""
import time
import gradio as gr
from pathlib import Path


def create_tab_ocr(config: dict, registry):
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传图片 / 拍照 / 截图", type="filepath")
            engine_choice = gr.Dropdown(
                choices=["自动路由 (推荐)", "OvisOCR2 (印刷文档)",
                         "PaddleOCR-VL (相机照片)", "HunyuanOCR (手写体)",
                         "PP-OCRv6 (CPU快速)"],
                value="自动路由 (推荐)",
                label="引擎选择",
            )
            with gr.Row():
                run_btn = gr.Button("识别", variant="primary", scale=2)
                camera_btn = gr.Button("相机采集", scale=1)

        with gr.Column(scale=2):
            output_text = gr.Textbox(label="识别结果", lines=12)
            with gr.Row():
                scene_box = gr.Textbox(label="场景判定", scale=1)
                engine_box = gr.Textbox(label="使用引擎", scale=1)
                time_box = gr.Textbox(label="耗时", scale=1)
            output_meta = gr.JSON(label="详细结果 (置信度/行框/结构化)")

    run_btn.click(
        fn=_run_ocr,
        inputs=[input_image, engine_choice],
        outputs=[output_text, scene_box, engine_box, time_box, output_meta],
    )
    camera_btn.click(
        fn=_grab_camera,
        inputs=[],
        outputs=[input_image],
    )


# ─── 引擎名称映射 ───────────────────────────────────────────
ENGINE_MAP = {
    "自动路由 (推荐)": "auto",
    "OvisOCR2 (印刷文档)": "ovisocr2",
    "PaddleOCR-VL (相机照片)": "paddleocr_vl",
    "HunyuanOCR (手写体)": "hunyuan_ocr",
    "PP-OCRv6 (CPU快速)": "rapidocr",
}

# 场景 → 引擎映射
SCENE_ENGINE_MAP = {
    "document": "ovisocr2",
    "camera": "paddleocr_vl",
    "handwriting": "hunyuan_ocr",
    "cpu_fallback": "rapidocr",
}


def _run_ocr(image_path: str, engine_label: str) -> tuple:
    """执行 OCR 识别，支持自动路由"""
    if not image_path:
        return "请先上传图片或使用相机采集", "—", "—", "—", {}

    t0 = time.time()
    engine_key = ENGINE_MAP.get(engine_label, "auto")

    registry = _get_registry()
    if registry is None:
        return "引擎注册表未初始化", "—", "—", "—", {"error": "registry is None"}

    # ─── 自动路由 ───────────────────────────────────────────
    scene_result = {}
    if engine_key == "auto":
        try:
            classifier = registry.ensure_loaded("scene_classifier")
            scene_result = classifier.infer(image_path)
            scene = scene_result.get("scene", "camera")
            engine_key = SCENE_ENGINE_MAP.get(scene, "paddleocr_vl")
            # 如果目标引擎不可用，降级到 rapidocr
            target = registry.get(engine_key)
            if target is None or target.state.value == "error":
                engine_key = "rapidocr"
        except Exception:
            engine_key = "rapidocr"
            scene_result = {"scene": "unknown", "confidence": 0.0}

    # ─── 加载并推理 ─────────────────────────────────────────
    try:
        engine = registry.ensure_loaded(engine_key)
        # 确认引擎真正就绪 (模型可能下载失败)
        if not engine.is_ready():
            raise RuntimeError(f"{engine_key} 加载后仍未就绪")
    except Exception as e:
        try:
            engine = registry.ensure_loaded("rapidocr")
            engine_key = "rapidocr"
        except Exception:
            return f"引擎加载失败: {e}", "—", engine_key, "—", {"error": str(e)}

    try:
        result = engine.infer(image_path)
        # 如果推理返回错误且不是 rapidocr，尝试降级
        if "error" in result and engine_key != "rapidocr":
            fallback = registry.ensure_loaded("rapidocr")
            result = fallback.infer(image_path)
            engine_key = "rapidocr"
            engine = fallback
    except Exception as e:
        # 推理异常，降级到 rapidocr
        if engine_key != "rapidocr":
            try:
                engine = registry.ensure_loaded("rapidocr")
                engine_key = "rapidocr"
                result = engine.infer(image_path)
            except Exception:
                return f"推理失败: {e}", "—", engine_key, "—", {"error": str(e)}
        else:
            return f"推理失败: {e}", "—", engine_key, "—", {"error": str(e)}

    elapsed = time.time() - t0

    # ─── 格式化输出 ─────────────────────────────────────────
    text = result.get("text", "")
    if not text and "markdown" in result:
        text = result["markdown"]
    if not text and "error" in result:
        text = f"[错误] {result['error']}"

    scene_display = scene_result.get("scene", "手动选择")
    conf = scene_result.get("confidence", None)
    if conf is not None:
        scene_display += f" ({conf:.0%})"

    engine_display = engine.meta.display_name if hasattr(engine, 'meta') else engine_key

    meta = {
        "engine": engine_key,
        "confidence": result.get("confidence"),
        "lines_count": len(result.get("lines", [])),
        "scene_rules": scene_result.get("rules_triggered", []),
    }

    return text, scene_display, engine_display, f"{elapsed:.2f}s", meta


def _grab_camera():
    """从配置的相机采集一帧"""
    registry = _get_registry()
    if registry is None:
        return None
    try:
        from core.camera import create_camera
        config = registry.config
        cam = create_camera(config)
        if cam.open():
            try:
                frame = cam.grab()
            finally:
                cam.close()  # M4 修复: 确保设备句柄释放
            if frame is not None:
                import cv2
                import tempfile
                # PNG 无损, 避免 JPEG 压缩伪影影响 OCR 精度
                tmp = Path(tempfile.gettempdir()) / "visionocr_capture.png"
                cv2.imwrite(str(tmp), frame)
                return str(tmp)
    except Exception as e:
        print(f"[Camera] 采集失败: {e}")
    return None


# ─── Registry 注入 ──────────────────────────────────────────
_registry = None


def set_registry(reg):
    global _registry
    _registry = reg


def _get_registry():
    return _registry
