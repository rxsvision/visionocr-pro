"""OCR 通用识别 Tab - 三引擎自动路由 + 图像预处理 + 进度日志"""
import time
import gradio as gr
from pathlib import Path

from ui.safe_yield import safe_generator


def _imread_safe(path: str, flags=None):
    """Unicode 安全的 cv2.imread (Windows 中文路径兼容)。"""
    import cv2
    import numpy as np
    if flags is None:
        flags = cv2.IMREAD_COLOR
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _imwrite_safe(path: str, img) -> bool:
    """Unicode 安全的 cv2.imwrite + 返回值检查。"""
    import cv2
    ext = Path(path).suffix or ".png"
    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


def create_tab_ocr(config: dict, registry, mode_toggle=None):
    with gr.Row():
        with gr.Column(scale=1):
            input_editor = gr.ImageEditor(
                label="上传图片 → 用画笔画框选择 ROI (不画=全图识别)",
                type="filepath",
                height=320,
                brush=gr.Brush(
                    default_size=5,
                    colors=["rgb(0, 220, 80)"],
                    default_color="rgb(0, 220, 80)",
                    color_mode="fixed",
                ),
                elem_id="ocr-roi-editor",
            )
            # ─── 工程师专属控件 (工人模式隐藏) ─────────────────
            with gr.Column(visible=False) as eng_panel:
                engine_choice = gr.Dropdown(
                    choices=["自动路由 (推荐)", "OvisOCR2 (印刷文档)",
                             "PaddleOCR-VL (相机照片)",
                             "HunyuanOCR (手写体·需24GB显存)",
                             "PP-OCRv6 (CPU快速)"],
                    value="自动路由 (推荐)",
                    label="引擎选择",
                )
                conf_threshold = gr.Slider(
                    minimum=0.50, maximum=0.95, step=0.05,
                    value=config.get("ocr", {}).get("confidence_threshold", 0.75),
                    label="置信度门槛 (低于此值 → 待人工复核)",
                )
                perspective_enable = gr.Checkbox(
                    label="几何矫正 (自动纠偏 + 透视校正 · 拍照场景建议开启)",
                    value=True,
                )
                with gr.Accordion("预处理设置 (拍摄良好时无需开启 · 双路径对比会加倍耗时)", open=False):
                    pp_enable = gr.Checkbox(label="启用预处理 + 双路径对比 (耗时翻倍)", value=False)
                    pp_clahe = gr.Checkbox(label="CLAHE 局部对比度 (clip=2, grid=4)", value=True)
                    pp_sharpen = gr.Checkbox(label="锐化 (工业刻字建议关闭)", value=False)
                    pp_upscale = gr.Checkbox(label="小图放大 2x (极小字符时开启)", value=False)
                    pp_binarize = gr.Checkbox(
                        label="自适应二值化 (仅黑白高对比标记)", value=False)
                with gr.Accordion("后处理纠错 (正则规则修正混淆字符)", open=False):
                    postprocess_enable = gr.Checkbox(
                        label="启用后处理纠错 (数字中 O→0, I→1, S→5, B→8)",
                        value=True)
                    custom_regex = gr.Textbox(
                        label="自定义正则规则 (每行一条: pattern|||replacement)",
                        placeholder="例: (?<=\\d)G(?=E)|||6  →  将数字后GE中的G修正为6",
                        lines=3,
                    )
            with gr.Row():
                run_btn = gr.Button("识别", variant="primary", scale=2)
                camera_btn = gr.Button("相机采集", scale=1)

        with gr.Column(scale=2):
            verdict_box = gr.Dataframe(
                headers=["区域", "置信度", "判定"],
                label="判定结果 (置信度门槛拦截)",
                column_count=(3, "fixed"),
                row_count=(1, "dynamic"),
                interactive=False,
            )
            output_text = gr.Textbox(label="识别结果", lines=10)
            with gr.Row():
                scene_box = gr.Textbox(label="场景判定", scale=1)
                engine_box = gr.Textbox(label="使用引擎", scale=1)
                time_box = gr.Textbox(label="耗时", scale=1)
            output_meta = gr.JSON(label="详细结果 (置信度/行框/结构化)")
            log_box = gr.Textbox(
                label="运行日志 (进度 / 报错)",
                lines=6, max_lines=12, interactive=False,
                elem_classes=["log-panel"],
            )

    run_btn.click(
        fn=_run_ocr_stream,
        inputs=[input_editor, engine_choice, conf_threshold, perspective_enable,
                pp_enable, pp_clahe, pp_sharpen, pp_upscale, pp_binarize,
                postprocess_enable, custom_regex],
        outputs=[verdict_box, output_text, scene_box, engine_box, time_box,
                 output_meta, log_box],
    )
    camera_btn.click(
        fn=_grab_camera_editor,
        inputs=[],
        outputs=[input_editor],
    )

    # ─── 模式切换 → 工程师面板可见性 ─────────────────────────
    if mode_toggle is not None:
        mode_toggle.change(
            fn=lambda m: gr.update(visible=(m == "工程师模式")),
            inputs=[mode_toggle],
            outputs=[eng_panel],
        )


# ─── 引擎名称映射 ───────────────────────────────────────────
ENGINE_MAP = {
    "自动路由 (推荐)": "auto",
    "OvisOCR2 (印刷文档)": "ovisocr2",
    "PaddleOCR-VL (相机照片)": "paddleocr_vl",
    "HunyuanOCR (手写体·需24GB显存)": "hunyuan_ocr",
    "PP-OCRv6 (CPU快速)": "rapidocr",
}

# 场景 → 引擎映射
SCENE_ENGINE_MAP = {
    "document": "ovisocr2",
    "camera": "paddleocr_vl",
    "handwriting": "hunyuan_ocr",
    "cpu_fallback": "rapidocr",
}


@safe_generator(lambda e: (None, f"内部错误: {e}", "—", "—", "—", {},
                          f"[ERROR] 未捕获异常: {e}"))
def _run_ocr_stream(editor_data, engine_label, conf_threshold, perspective_enable,
                    pp_enable, pp_clahe, pp_sharpen, pp_upscale, pp_binarize,
                    postprocess_enable, custom_regex):
    """Generator: 流式输出进度日志 + 最终结果。"""
    logs = []
    _temp_files = []  # 追踪临时文件, generator 结束时清理

    def log(msg):
        logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        return "\n".join(logs)

    def _track_temp(path):
        """注册临时文件路径, 供结束时清理。"""
        if path:
            _temp_files.append(path)
        return path

    def _cleanup_temps():
        """删除本次推理产生的所有临时文件。"""
        import os
        for p in _temp_files:
            try:
                if os.path.isfile(p):
                    os.unlink(p)
            except OSError:
                pass

    # ─── 从 ImageEditor 提取图像路径 + ROI ──────────────────
    # type="filepath" 时 preprocess 返回:
    #   {"background": str|None, "layers": [str,...], "composite": str|None}
    image_path = None
    roi_boxes = []  # [(x1,y1,x2,y2), ...]

    def _extract_path(val):
        """兼容 str / dict / FileData 对象三种形态。"""
        if val is None:
            return None
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            return val.get("path")
        return getattr(val, "path", None)

    if editor_data is not None:
        if isinstance(editor_data, dict):
            bg = editor_data.get("background")
            layers_raw = editor_data.get("layers") or []
        else:
            bg = getattr(editor_data, "background", None)
            layers_raw = getattr(editor_data, "layers", None) or []

        image_path = _extract_path(bg)

        # 从 layers 提取用户画框的 bounding box (支持多框)
        MIN_ROI_AREA = 400  # 最小有效 ROI 面积 (px²), 过滤误触小点
        for layer in layers_raw:
            layer_path = _extract_path(layer)
            if layer_path:
                try:
                    import cv2
                    import numpy as np
                    mask = _imread_safe(layer_path, cv2.IMREAD_UNCHANGED)
                    if mask is not None:
                        # 取 alpha 通道或灰度非零区域
                        if mask.ndim == 3 and mask.shape[2] == 4:
                            alpha = mask[:, :, 3]
                        else:
                            alpha = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.ndim == 3 else mask
                        # 用轮廓检测分离独立画框区域
                        contours, _ = cv2.findContours(
                            alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        lh, lw = mask.shape[:2]
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if w * h >= MIN_ROI_AREA:
                                roi_boxes.append((x, y, x + w, y + h, lw, lh))
                except Exception:
                    pass

    if not image_path:
        yield (None, "请先上传图片 (可选: 用画笔画框选择 ROI 区域)", "—", "—", "—", {},
               log(f"⚠ 未检测到输入图像 (editor_data type={type(editor_data).__name__})"))
        return

    yield (None, "", "—", "—", "—", {},
           log(f"✓ 图像已加载: {Path(image_path).name}"
               + (f" · 检测到 {len(roi_boxes)} 个画框" if roi_boxes else " · 全图模式")))

    t0 = time.time()

    registry = _get_registry()
    if registry is None:
        yield (None, "引擎注册表未初始化", "—", "—", "—",
               {"error": "registry is None"}, log("⚠ Registry 未初始化"))
        return

    # ─── 构建推理目标列表 (多 ROI 分别识别) ─────────────────
    import tempfile
    import uuid as _uuid
    import cv2
    import numpy as np

    targets = []  # [(label, path), ...]
    roi_warnings = []

    if roi_boxes:
        img = _imread_safe(image_path)
        if img is None:
            yield (None, "图像读取失败", "—", "—", "—", {},
                   log("✗ cv2.imread 返回 None"))
            return
        h_img, w_img = img.shape[:2]

        # 按从上到下、从左到右排序 (符合阅读顺序)
        roi_boxes.sort(key=lambda b: (b[1], b[0]))

        for idx, (x1, y1, x2, y2, lw, lh) in enumerate(roi_boxes, 1):
            # 坐标缩放 (layer 分辨率可能 ≠ 原图)
            sx = w_img / lw if lw and lw != w_img else 1.0
            sy = h_img / lh if lh and lh != h_img else 1.0
            if sx != 1.0 or sy != 1.0:
                x1, x2 = int(x1 * sx), int(x2 * sx)
                y1, y2 = int(y1 * sy), int(y2 * sy)
            # 边界保护
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            rw, rh = x2 - x1, y2 - y1

            # 验证: 太小的框视为无效
            if rw < 15 or rh < 15:
                roi_warnings.append(
                    f"ROI-{idx} ({rw}x{rh}px) 过小, 已跳过")
                continue

            cropped = img[y1:y2, x1:x2]
            uid = _uuid.uuid4().hex[:8]
            roi_path = Path(tempfile.gettempdir()) / f"visionocr_roi_{uid}.png"
            if not _imwrite_safe(str(roi_path), cropped):
                roi_warnings.append(f"ROI-{idx} 写入失败 (磁盘满或权限不足)")
                continue
            _track_temp(str(roi_path))
            targets.append((f"ROI-{idx} ({x1},{y1})-({x2},{y2})", str(roi_path)))

        # 报告无效框
        for warn in roi_warnings:
            yield (None, "", "—", "—", "—", {}, log(f"⚠ {warn}"))

        if not targets:
            yield (None, "所有画框区域均无效 (过小), 请重新框选", "—", "—", "—", {},
                   log("✗ 无有效 ROI, 识别终止"))
            return

        yield (None, "", "—", "—", "—", {},
               log(f"✓ 有效 ROI: {len(targets)} 个"
                   + (f" · 跳过 {len(roi_warnings)} 个无效框" if roi_warnings else "")))
    else:
        targets.append(("全图", image_path))

    roi_applied = len(roi_boxes) > 0 and len(targets) > 0

    # ─── 图像质量预检 (对第一个目标) ─────────────────────────
    try:
        from core.image_preprocess import check_image_quality
        quality = check_image_quality(targets[0][1])
        if not quality["ok"]:
            yield (None, "", "—", "—", "—", {},
                   log(f"⚠ 质量预检: {quality['warning']}"))
        else:
            yield (None, "", "—", "—", "—", {},
                   log(f"✓ 质量预检通过 (清晰度 {quality['blur_score']})"))
    except Exception:
        pass

    # ─── 自动路由 + 加载引擎 (只执行一次) ────────────────────
    engine_key = ENGINE_MAP.get(engine_label, "auto")
    scene_result = {}
    if engine_key == "auto":
        # M-3: 从 config 读取分类器阈值和降级引擎, 不硬编码
        _sc_cfg = (registry.config.get("ocr", {}) or {}).get("scene_classifier", {}) or {}
        _sc_threshold = float(_sc_cfg.get("confidence_threshold", 0.7))
        _sc_fallback = _sc_cfg.get("fallback_engine", "rapidocr")

        yield (None, "", "—", "—", "—", {}, log("▶ 场景分类中..."))
        try:
            classifier = registry.ensure_loaded("scene_classifier")
            scene_result = classifier.infer(image_path)
            scene = scene_result.get("scene", "camera")
            conf = scene_result.get("confidence", 0)
            if conf < _sc_threshold:
                engine_key = _sc_fallback
                yield (None, "", "—", "—", "—", {},
                       log(f"✓ 分类器旁路 (置信度 {conf:.0%} < {_sc_threshold:.0%}), 默认 {engine_key}"))
            else:
                engine_key = SCENE_ENGINE_MAP.get(scene, _sc_fallback)
                target_eng = registry.get(engine_key)
                if target_eng is None or target_eng.state.value == "error":
                    engine_key = _sc_fallback
                yield (None, "", "—", "—", "—", {},
                       log(f"✓ 场景: {scene} ({conf:.0%}) → 引擎: {engine_key}"))
        except Exception as e:
            engine_key = _sc_fallback
            scene_result = {"scene": "unknown", "confidence": 0.0}
            yield (None, "", "—", "—", "—", {},
                   log(f"⚠ 场景分类失败 ({e}), 降级 {engine_key}"))

    yield (None, "", "—", "—", "—", {}, log(f"▶ 加载引擎: {engine_key} ..."))
    try:
        engine = registry.ensure_loaded(engine_key)
        if not engine.is_ready():
            raise RuntimeError(f"{engine_key} 加载后仍未就绪")
        yield (None, "", "—", "—", "—", {}, log(f"✓ 引擎就绪: {engine_key}"))
    except Exception as e:
        yield (None, "", "—", "—", "—", {},
               log(f"⚠ {engine_key} 加载失败 ({e}), 降级 rapidocr"))
        try:
            engine = registry.ensure_loaded("rapidocr")
            engine_key = "rapidocr"
        except Exception:
            yield (None, f"引擎加载失败: {e}", "—", engine_key, "—",
                   {"error": str(e)}, log("✗ 所有引擎不可用"))
            return

    # ─── 逐目标推理 ─────────────────────────────────────────
    all_texts = []
    all_results = []
    total_conf = 0.0
    total_lines = 0

    for label, target_path in targets:
        yield (None, "", "—", "—", "—", {},
               log(f"▶ 识别 [{label}] ..."))

        # 几何矫正 (透视 + 纠偏)
        geo_path = target_path
        if perspective_enable:
            try:
                from core.perspective_correct import correct_perspective
                geo_path, geo_meta = correct_perspective(target_path)
                if geo_path != target_path:
                    _track_temp(geo_path)
                if geo_meta.get("corrected"):
                    info_parts = []
                    if geo_meta.get("perspective"):
                        info_parts.append("透视校正")
                    if geo_meta.get("deskew_angle"):
                        info_parts.append(f"纠偏 {geo_meta['deskew_angle']}°")
                    yield (None, "", "—", "—", "—", {},
                           log(f"✓ 几何矫正: {' + '.join(info_parts)}"))
            except Exception as e:
                geo_path = target_path
                yield (None, "", "—", "—", "—", {},
                       log(f"⚠ 几何矫正跳过 ({e})"))

        # 预处理 (每个目标独立, 基于几何矫正后的图像)
        infer_path = geo_path
        pp_meta = {}
        if pp_enable:
            try:
                from core.image_preprocess import preprocess_for_ocr
                pp_cfg = {
                    "enabled": True, "clahe": pp_clahe, "sharpen": pp_sharpen,
                    "upscale": pp_upscale, "binarize": pp_binarize, "denoise": True,
                }
                infer_path, pp_meta = preprocess_for_ocr(geo_path, pp_cfg)
                if infer_path != geo_path:
                    _track_temp(infer_path)
            except Exception:
                infer_path = geo_path

        # 推理
        try:
            result = engine.infer(infer_path)
            if "error" in result and engine_key != "rapidocr":
                fallback = registry.ensure_loaded("rapidocr")
                result = fallback.infer(infer_path)
        except Exception as e:
            if engine_key != "rapidocr":
                try:
                    engine_fb = registry.ensure_loaded("rapidocr")
                    result = engine_fb.infer(infer_path)
                except Exception:
                    result = {"text": "", "error": str(e), "confidence": 0}
            else:
                result = {"text": "", "error": str(e), "confidence": 0}

        # 双路径对比 (预处理 vs 原图) — M-2: 综合置信度+文本完整度
        if pp_enable and infer_path != geo_path and pp_meta.get("steps"):
            try:
                result_orig = engine.infer(geo_path)
                text_pp = result.get("text", "") or ""
                text_orig = result_orig.get("text", "") or ""
                if text_orig:  # 原图有结果才比较
                    conf_pp = result.get("confidence", 0) or 0
                    conf_orig = result_orig.get("confidence", 0) or 0
                    lines_pp = len(result.get("lines", []) or [])
                    lines_orig = len(result_orig.get("lines", []) or [])
                    max_lines = max(lines_pp, lines_orig, 1)
                    # 综合评分: 置信度 70% + 文本完整度 30%
                    score_pp = conf_pp * 0.7 + (lines_pp / max_lines) * 0.3
                    score_orig = conf_orig * 0.7 + (lines_orig / max_lines) * 0.3
                    if score_orig > score_pp:
                        result = result_orig
            except Exception:
                pass

        # 提取文本
        text = result.get("text", "") or result.get("markdown", "")
        conf = result.get("confidence", 0) or 0
        lines = result.get("lines", [])

        if text:
            all_texts.append((label, text))
            total_conf += conf
            total_lines += len(lines)
            yield (None, "", "—", "—", "—", {},
                   log(f"✓ [{label}] {conf:.1%} · {len(lines)} 行"))
        else:
            err = result.get("error", "无识别结果")
            all_texts.append((label, f"[识别失败: {err}]"))
            yield (None, "", "—", "—", "—", {},
                   log(f"⚠ [{label}] 识别失败: {err}"))

        all_results.append({"label": label, "confidence": conf,
                            "lines": len(lines), "error": result.get("error")})

    elapsed = time.time() - t0

    # ─── 条码自动检测 (与OCR并行, 工业追溯刚需) ─────────────
    barcode_info = ""
    barcode_codes = []
    try:
        barcode_engine = registry.ensure_loaded("barcode")
        if barcode_engine and barcode_engine.is_ready():
            bc_result = barcode_engine.infer(image_path)
            barcode_codes = bc_result.get("codes", [])
            if barcode_codes:
                bc_lines = []
                for bc in barcode_codes:
                    bc_lines.append(f"[{bc['type']}] {bc['content']}")
                barcode_info = "\n".join(bc_lines)
                yield (None, "", "—", "—", "—", {},
                       log(f"✓ 条码: {len(barcode_codes)} 个 ({', '.join(bc['type'] for bc in barcode_codes)})"))
    except Exception as e:
        yield (None, "", "—", "—", "—", {},
               log(f"· 条码检测跳过 ({e})"))

    # ─── 合并文本 + 后处理纠错 ──────────────────────────────
    if len(all_texts) == 1:
        combined_text = all_texts[0][1]
    else:
        # 多 ROI: 带标签分段输出
        parts = []
        for label, txt in all_texts:
            parts.append(f"【{label}】\n{txt}")
        combined_text = "\n\n".join(parts)

    raw_text = combined_text  # 后处理前的原始文本 (审计用)
    corrections = []
    if postprocess_enable and combined_text and "[识别失败" not in combined_text:
        try:
            from core.postprocess import apply_corrections
            custom_rules = []
            if custom_regex and custom_regex.strip():
                for line in custom_regex.strip().split("\n"):
                    line = line.strip()
                    if "|||" in line:
                        pat, rep = line.split("|||", 1)
                        custom_rules.append({
                            "pattern": pat.strip(),
                            "replacement": rep.strip(),
                            "name": "user_rule",
                        })
            combined_text, corrections = apply_corrections(
                combined_text, custom_rules=custom_rules)
            if corrections:
                yield (None, "", "—", "—", "—", {},
                       log(f"✓ 后处理纠错 ({len(corrections)} 处): " +
                           "; ".join(corrections[:3])))
        except Exception as e:
            yield (None, "", "—", "—", "—", {},
                   log(f"⚠ 后处理跳过 ({e})"))

    # ─── 置信度门槛判定 (NG 拦截) ────────────────────────────
    threshold = float(conf_threshold) if conf_threshold else 0.75
    verdict_rows = []  # [[区域, 置信度, 判定], ...]
    ng_regions = []

    for r in all_results:
        label = r["label"]
        c = r["confidence"]
        if r.get("error") and not r.get("lines"):
            verdict_rows.append([label, "—", "✗ 识别失败"])
            ng_regions.append(f"{label} 识别失败")
        elif c >= threshold:
            verdict_rows.append([label, f"{c:.1%}", "✓ OK"])
        else:
            verdict_rows.append([label, f"{c:.1%}", "⚠ 待人工复核"])
            ng_regions.append(f"{label} conf {c:.1%} < {threshold:.0%}")

    if ng_regions:
        overall_verdict = "NG — 待人工复核"
    else:
        overall_verdict = "OK"

    # 在识别结果文本顶部插入判定摘要
    verdict_header = f"═══ 判定: {overall_verdict} ═══"
    if ng_regions:
        verdict_header += "\n" + " | ".join(ng_regions)
    combined_text = verdict_header + "\n\n" + combined_text

    # 条码结果附加 (工业追溯: 批次号/工件ID/产线路由)
    if barcode_info:
        combined_text += f"\n\n─── 条码 ───\n{barcode_info}"

    # ─── 格式化输出 ─────────────────────────────────────────
    scene_display = scene_result.get("scene", "手动选择")
    conf_sc = scene_result.get("confidence", None)
    if conf_sc is not None:
        scene_display += f" ({conf_sc:.0%})"

    engine_display = engine.meta.display_name if hasattr(engine, 'meta') else engine_key
    avg_conf = total_conf / len(targets) if targets else 0

    meta = {
        "engine": engine_key,
        "confidence": round(avg_conf, 4),
        "confidence_threshold": threshold,
        "verdict": overall_verdict,
        "ng_regions": ng_regions,
        "roi_count": len(targets),
        "roi_results": all_results,
        "lines_count": total_lines,
        "scene_rules": scene_result.get("rules_triggered", []),
        "corrections": corrections,
        "roi_applied": roi_applied,
        "roi_warnings": roi_warnings,
        "barcodes": barcode_codes,
    }

    final_log = log(f"✓ 完成 · 耗时 {elapsed:.2f}s · "
                    f"判定: {overall_verdict} · "
                    f"{len(targets)} 个区域 · "
                    f"平均置信度 {avg_conf:.1%} · {total_lines} 行")
    # ─── 审计日志持久化 (生产追溯) ─────────────────────────
    try:
        import hashlib
        from core.database import log_ocr_audit
        img_hash = ""
        try:
            with open(image_path, "rb") as f:
                img_hash = hashlib.sha256(f.read(1024 * 1024)).hexdigest()[:16]
        except OSError:
            pass
        log_ocr_audit("data", {
            "image_path": image_path,
            "image_hash": img_hash,
            "engine": engine_key,
            "scene": scene_result.get("scene", "manual"),
            "roi_count": len(targets),
            "raw_text": raw_text,
            "corrected_text": combined_text.split("═══\n\n", 1)[-1] if "═══" in combined_text else combined_text,
            "confidence": avg_conf,
            "confidence_threshold": threshold,
            "verdict": overall_verdict,
            "ng_regions": ng_regions,
            "corrections": corrections,
            "elapsed_sec": round(elapsed, 3),
        })
    except Exception:
        pass  # 审计失败不阻断主流程

    _cleanup_temps()
    yield (verdict_rows, combined_text, scene_display, engine_display,
           f"{elapsed:.2f}s", meta, final_log)


def _grab_camera_editor():
    """从配置的相机采集一帧, 返回路径供 ImageEditor 显示。"""
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
                cam.close()
            if frame is not None:
                import cv2
                import tempfile
                import uuid as _uuid
                uid = _uuid.uuid4().hex[:8]
                tmp = Path(tempfile.gettempdir()) / f"visionocr_capture_{uid}.png"
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
