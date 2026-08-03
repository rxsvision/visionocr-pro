"""工业质检 Tab (Phase 4A: Grounding DINO 零样本缺陷检测)

工作流: 拍照/上传 → 选择产品配方(或手动输入缺陷词) → 一键检测 → OK/NG 判定
设计原则: 工人傻瓜式操作, 一键出结果, 无需理解模型细节。
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from ui.safe_yield import safe_generator

import gradio as gr

from core.config import load_config
from core.database import get_conn
from core.defect_detector import (
    DEFAULT_PROMPT, run_detection, run_union_detection, save_qc_result,
    persist_qc_image,
    list_recipes, load_recipe, save_recipe, delete_recipe,
)
from core.anomaly_bank import (
    list_banks, bank_exists, delete_bank,
    register_ok_samples, run_anomaly_detection,
    list_banks_subspace, register_subspace_bank, run_subspace_detection,
)
from core.calibration_protocol import (
    MIN_CAL_RECOMMENDED, format_report_md, recalibrate_product,
)
from core.fusion import calibrated_n_samples, fusion_stage

logger = logging.getLogger("visionocr.tab_qc")

_registry = None
_config = None

# 缓存最近一次 3D 深度帧 (DepthFrame), 供"一键检测"阶段做深度融合判定。
# 工人流程: 选"3D深度相机"采集 -> 深度帧存入此处 -> 检测时自动融合。
_last_depth_frame = None

# 缓存最近一次 Union 检测 (image_path + result), 供 "AI 缺陷解释" 按钮
# 复用检测结果做 VLM ROI 裁切, 避免重复检测。
_last_union = None


def set_registry(registry) -> None:
    global _registry
    _registry = registry


def _get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def create_tab_qc(config: dict, registry, mode_toggle=None):
    set_registry(registry)

    # ═══ 单行双列: 左=输入/配置, 右=输出/结果 ═══════════════════
    with gr.Row():
        # ─── 左列: 图像输入 + 全部配置 ─────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 图像输入")
            image_source = gr.Radio(
                choices=["上传图像", "2D彩色相机", "3D深度相机 (Sizector)"],
                value="上传图像",
                label="图像来源",
            )
            qc_image = gr.Image(label="拍照 / 上传待检图像", type="filepath")
            depth_preview = gr.Image(
                label="3D 深度图预览 (伪彩, 仅深度相机)",
                visible=False, height=180,
            )
            with gr.Row():
                camera_btn = gr.Button("📷 相机采集", scale=1)
                detect_btn = gr.Button("🔍 一键检测", variant="primary", scale=2)

            # ─── 工程师专属控件 (工人模式隐藏) ─────────────────
            with gr.Column(visible=False) as eng_panel:
                gr.Markdown("---")
                gr.Markdown("### 检测模式")
                detect_mode = gr.Radio(
                    choices=["零样本 (Grounding DINO)", "少样本 (PatchCore)",
                             "Union 零漏检 (三源OR)",
                             "快速换线辅助 (SubspaceAD)"],
                    value="零样本 (Grounding DINO)",
                    label="检测模式",
                    info="零样本: 提示词驱动 | 少样本: OK样本建库 | "
                         "Union: PatchCore+DINO+YOLO 任一NG即NG (漏检零容忍) | "
                         "SubspaceAD辅助: 1-4张OK图极速建库, 仅分数+热力图提示, "
                         "不做自主判定, 人工复核",
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

                gr.Markdown("---")
                gr.Markdown("### 快速换线注册 (SubspaceAD 辅助通道)")
                gr.Markdown(
                    "_1-4 张 OK 图即可建库 (旋转增广), 仅作辅助提示; "
                    "正式量产请补足 ≥10 张并改用 PatchCore 主判。_")
                sa_product = gr.Dropdown(
                    label="产品特征库 (辅助通道)",
                    choices=["(新建)"] + list_banks_subspace(),
                    value="(新建)",
                )
                sa_product_name = gr.Textbox(
                    label="新产品名称 (新建时填写)",
                    placeholder="如: 新品首件_型号B",
                )
                sa_ok_upload = gr.File(
                    label="上传 OK 样本 (快速换线 1~4 张; ≥10 张走标准建库)",
                    file_count="multiple",
                    file_types=[".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
                )
                sa_register_btn = gr.Button("📦 注册建库 (辅助)",
                                            variant="secondary")
                sa_status = gr.Markdown("")

                gr.Markdown("---")
                gr.Markdown("### 📐 校准协议 (NP 校准扩充, §6.2)")
                gr.Markdown(
                    "_建库后补采 **≥30 张独立 OK 图** (不得是建库图; "
                    "建议变换光照/角度拍 3 组) 重标定 NP 阈值 → "
                    "n_cal 达标后融合自动升级双源互证, 误报大降、漏检不变。_")
                cal_product = gr.Dropdown(
                    label="待校准产品 (须已建库)",
                    choices=list_banks(),
                )
                cal_ok_upload = gr.File(
                    label=f"上传校准 OK 图 (≥{MIN_CAL_RECOMMENDED} 张, 独立于建库图)",
                    file_count="multiple",
                    file_types=[".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
                )
                cal_ng_upload = gr.File(
                    label="(可选) NG 缺陷样本 — 用于 Recall 回归实测",
                    file_count="multiple",
                    file_types=[".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
                )
                cal_run_btn = gr.Button("📐 执行校准协议", variant="secondary")
                cal_status = gr.Markdown("")

        # ─── 右列: 全部输出/结果 ─────────────────────────────
        with gr.Column(scale=3):
            gr.Markdown("### 检测结果")
            result_image = gr.Image(label="标注结果 (红框=缺陷)", height=420)
            with gr.Row():
                verdict_box = gr.Textbox(
                    label="判定", scale=1, interactive=False)
                score_box = gr.Textbox(
                    label="最高置信度", scale=1, interactive=False)
                count_box = gr.Textbox(
                    label="缺陷数", scale=1, interactive=False)
            detail_table = gr.Dataframe(
                headers=["#", "缺陷类型", "置信度", "位置 (x1,y1,x2,y2)"],
                label="检测明细 (编号对应图上标注)",
                wrap=True,
            )
            with gr.Accordion("🔍 AI 缺陷解释 (VLM 局部放大)", open=False):
                explain_btn = gr.Button(
                    "AI 解释 (裁剪可疑区域 → 本地 VLM 识读)",
                    variant="secondary")
                explain_gallery = gr.Gallery(
                    label="候选区域 (自动裁切)", columns=3, height=180)
                explain_md = gr.Markdown("")
            status_msg = gr.Markdown("")
            log_box = gr.Textbox(
                label="运行日志 (进度 / 报错)",
                lines=8, max_lines=15, interactive=False,
                elem_classes=["log-panel"],
            )

    # ─── 事件绑定 ────────────────────────────────────────────
    detect_btn.click(
        fn=_run_detect,
        inputs=[qc_image, prompt_input, threshold_slider, detect_mode, pc_product,
                fusion_enable, depth_threshold, fusion_mode_radio, sa_product],
        outputs=[result_image, verdict_box, score_box, count_box,
                 detail_table, status_msg, log_box],
    )
    explain_btn.click(
        fn=_explain_union,
        inputs=[],
        outputs=[explain_gallery, explain_md],
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
    sa_register_btn.click(
        fn=_register_subspace_bank,
        inputs=[sa_product, sa_product_name, sa_ok_upload],
        outputs=[sa_status, sa_product],
    )
    cal_run_btn.click(
        fn=_run_calibration,
        inputs=[cal_product, cal_ok_upload, cal_ng_upload],
        outputs=[cal_status],
    )

    # ─── 模式切换 → 工程师面板可见性 ─────────────────────────
    if mode_toggle is not None:
        mode_toggle.change(
            fn=lambda m: gr.update(visible=(m == "工程师模式")),
            inputs=[mode_toggle],
            outputs=[eng_panel],
        )


# ─── 回调函数 ────────────────────────────────────────────────
@safe_generator(lambda e: (None, "ERROR", "—", "—", [], "",
                          f"[ERROR] 未捕获异常: {e}"))
def _run_detect(image_path, prompt, threshold, mode, pc_product,
                fusion_enable=False, depth_threshold=0.5, fusion_mode="OR (高召回, 推荐)",
                sa_product=""):
    """一键检测 (Generator): 流式输出进度日志 + 最终结果。"""
    import time as _time
    global _last_union
    _last_union = None  # 每次检测重置, 防止 AI 解释复用过期结果
    logs = []

    def log(msg):
        logs.append(f"[{_time.strftime('%H:%M:%S')}] {msg}")
        return "\n".join(logs)

    _EMPTY = (None, "—", "—", "—", [], "", "")

    if not image_path:
        yield (None, "—", "—", "—", [],
               "⚠ 请先上传图片或使用相机采集。", log("⚠ 无输入图像"))
        return

    registry = _registry
    if registry is None:
        yield (None, "ERROR", "—", "—", [],
               "⚠ 引擎未初始化, 请重启应用。", log("✗ Registry 为 None"))
        return

    # ─── 3D 深度融合模式 ─────────────────────────────────────
    global _last_depth_frame
    if fusion_enable and _last_depth_frame is not None \
            and "PatchCore" not in (mode or "") \
            and "SubspaceAD" not in (mode or ""):
        yield _EMPTY[:6] + (log("▶ 3D 深度融合检测启动..."),)
        result = _run_fusion_detect(registry, image_path, prompt, threshold,
                                    depth_threshold, fusion_mode)
        yield result[:6] + (log(f"✓ 融合检测完成 · {result[5]}"),)
        return

    # ─── PatchCore 少样本模式 ─────────────────────────────
    if "PatchCore" in (mode or ""):
        yield _EMPTY[:6] + (log("▶ PatchCore 少样本检测..."),)
        product = "" if pc_product == "(新建)" else (pc_product or "")
        result = run_anomaly_detection(registry, image_path,
                                       product_name=product,
                                       threshold=threshold)
        if result.get("error"):
            yield (None, "ERROR", "—", "—", [],
                   f"⚠ {result['error']}", log(f"✗ {result['error']}"))
            return

        verdict = result["pred_label"]
        score = result.get("score", 0)
        overlay = result.get("heatmap_overlay")

        if verdict == "OK":
            verdict_str = "✓ OK (合格)"
        else:
            verdict_str = f"✗ NG (异常分数 {score:.3f})"

        table = [["1", "异常热力图", f"{score:.4f}", "见标注图"]]
        status = f"PatchCore 检测 · 产品: {product or '默认'} · 阈值: {threshold}"
        yield (overlay, verdict_str, f"{score:.4f}", "1" if verdict == "NG" else "0",
               table, status, log(f"✓ {verdict_str}"))
        return

    # ─── SubspaceAD 快速换线辅助模式 (不给自主判定, 人工复核) ──
    if "SubspaceAD" in (mode or ""):
        yield _EMPTY[:6] + (log("▶ SubspaceAD 辅助提示 (快速换线, 仅供参考)..."),)
        product = "" if sa_product in ("(新建)", None) else (sa_product or "")
        result = run_subspace_detection(registry, image_path,
                                        product_name=product)
        if result.get("error"):
            yield (None, "ERROR", "—", "—", [],
                   f"⚠ {result['error']}", log(f"✗ {result['error']}"))
            return

        score = result.get("score", 0)
        overlay = result.get("heatmap_overlay")
        pred = result.get("pred_label")
        if pred == "REVIEW":
            verdict_str = f"◐ 仅供参考 (分数 {score:.3f}, 需人工复核)"
        elif pred == "NG":
            verdict_str = f"✗ NG (异常分数 {score:.3f})"
        else:
            verdict_str = f"✓ OK (分数 {score:.3f})"
        table = [["1", "异常热力图 (辅助)", f"{score:.4f}", "见标注图"]]
        status = (f"SubspaceAD 辅助提示 · 产品: {product or '自动'} · "
                  f"本通道仅供参考, 最终判定以人工/主判通道为准")
        yield (overlay, verdict_str, f"{score:.4f}", "—",
               table, status, log(f"✓ {verdict_str}"))
        return

    # ─── Union 零漏检模式 (四源 OR: PatchCore + DINO + YOLO + DINOv2) ──
    if "Union" in (mode or ""):
        yield _EMPTY[:6] + (log("▶ Union 零漏检 (PatchCore+DINO+YOLO+DINOv2 "
                                "分阶段融合: 双源互证→NG, 单源孤证→REVIEW 黄牌)..."),)
        cfg = _get_config()
        product = "" if pc_product in ("(新建)", None) else (pc_product or "")
        result = run_union_detection(
            registry, image_path, prompt=prompt, threshold=threshold,
            config=cfg, product_name=product)

        if result.get("error"):
            yield (result.get("image"), "ERROR", "—", "—", [],
                   f"⚠ {result['error']}", log(f"✗ {result['error']}"))
            return

        # 缓存供 "AI 缺陷解释" 复用 (不重复检测)
        _last_union = {"image_path": image_path, "result": result}

        verdict = result["verdict"]
        sources = result.get("ng_sources", [])
        pc = result.get("patchcore")
        dino = result.get("dino")
        yolo = result.get("yolo")
        dv = result.get("dinov2")

        # 统一明细表 (对齐表头 #/缺陷类型/置信度/位置) + detections (落库用)
        # 注: max_score 仅统计 dino/yolo (0~1概率语义); patchcore 距离分与
        #     dinov2 NLL 分为无界量纲, 混入会破坏百分比显示
        table = []
        detections = []
        max_score = 0.0
        row_no = 0
        if pc:
            row_no += 1
            table.append([str(row_no), "[PatchCore] 表面异常",
                          f"{pc.get('score', 0):.4f}", "热力图"])
            detections.append({"source": "patchcore", "label": "表面异常",
                               "score": pc.get("score", 0)})
        if dv:
            row_no += 1
            table.append([str(row_no), "[DINOv2] 表面异常",
                          f"{dv.get('score', 0):.4f}", "热力图"])
            detections.append({"source": "dinov2", "label": "表面异常",
                               "score": dv.get("score", 0)})
        if dino:
            for det in dino.get("detections", []):
                box = det["box"]
                row_no += 1
                table.append([str(row_no), f"[DINO] {det['label']}",
                              f"{det['score']:.2%}",
                              f"({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})"])
                detections.append({"source": "dino", **det})
            max_score = max(max_score, float(dino.get("max_score", 0)))
        if yolo:
            for b, l, s in zip(yolo.get("boxes", []),
                               yolo.get("labels", []),
                               yolo.get("scores", [])):
                row_no += 1
                table.append([str(row_no), f"[YOLO] {l}", f"{s:.2%}",
                              f"({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})"])
                detections.append({"source": "yolo", "box": b,
                                   "label": l, "score": s})
            max_score = max(max_score, float(yolo.get("max_score", 0)))

        if verdict == "OK":
            verdict_str = "✓ OK (合格)"
        elif verdict == "REVIEW":
            verdict_str = (f"◐ REVIEW 待人工复核 "
                           f"(触发源: {'+'.join(sources)}; 单源孤证不自主判NG)")
        else:
            verdict_str = f"✗ NG (触发源: {'+'.join(sources)})"

        # 落库
        try:
            data_dir = cfg.get("data_dir", "data")
            conn = get_conn(data_dir)
            save_qc_result(
                conn,
                persist_qc_image(image_path, Path(data_dir) / "qc_images"),
                verdict, detections, max_score, f"[Union] {prompt}")
            conn.close()
        except Exception as e:
            logger.warning("QC 结果落库失败: %s", e)

        active = [s for s, r in (("PatchCore", pc), ("DINO", dino),
                                 ("YOLO", yolo), ("DINOv2", dv))
                  if r]
        _fused = result.get("fusion", {})
        if (_fused.get("mode") or "staged") == "or":
            _finfo = "融合: 纯OR (v1.3.0)"
        else:
            _ncal = _fused.get("n_cal")
            _finfo = (f"融合: 阶段{_fused.get('stage', '?')} "
                      f"(n_cal={_ncal if _ncal is not None else '—'})")
        status = (f"Union 零漏检 · 产品: {product or '默认'} · "
                  f"激活源: {'+'.join(active) or '无'} · {_finfo}")
        yield (result.get("image"), verdict_str, f"{max_score:.2%}",
               str(len(detections)), table, status,
               log(f"✓ {verdict_str} · 最高分 {max_score:.2%}"))
        return

    # ─── Grounding DINO 零样本模式 (默认) ─────────────────
    # 检查模型是否已加载, 给出下载提示
    engine = registry.get("grounding_dino")
    if engine is None:
        yield (None, "ERROR", "—", "—", [],
               "⚠ Grounding DINO 引擎未注册", log("✗ 引擎未注册"))
        return

    if not engine.is_ready():
        yield _EMPTY[:6] + (
            log("▶ 首次加载 Grounding DINO 模型...\n"
                "  (首次需从 HuggingFace 下载 ~2.5GB 权重, 请耐心等待;\n"
                "   若网络不通, 日志将显示超时错误。后续启动为离线加载, 仅需数秒)"),)
        try:
            registry.ensure_loaded("grounding_dino")
        except Exception as e:
            yield (None, "ERROR", "—", "—", [],
                   f"⚠ 模型加载失败: {e}",
                   log(f"✗ 模型加载失败: {e}\n"
                       "  排查: 1) 检查网络能否访问 huggingface.co\n"
                       "  2) 或设置 HF_ENDPOINT=https://hf-mirror.com\n"
                       "  3) 或手动下载模型到 models/ 目录"))
            return
        if not engine.is_ready():
            yield (None, "ERROR", "—", "—", [],
                   "⚠ 模型加载后仍未就绪", log("✗ 模型状态异常"))
            return
        yield _EMPTY[:6] + (log("✓ 模型加载完成"),)

    yield _EMPTY[:6] + (log(f"▶ 推理中 (提示词: {prompt[:40]}..., 阈值: {threshold})..."),)
    result = run_detection(registry, image_path, prompt=prompt, threshold=threshold)

    if result.get("error"):
        yield (result.get("image"), "ERROR", "—", "—", [],
               f"⚠ {result['error']}", log(f"✗ 检测失败: {result['error']}"))
        return

    verdict = result["verdict"]
    max_score = result["max_score"]
    count = result["count"]

    if verdict == "OK":
        verdict_str = "✓ OK (合格)"
    else:
        verdict_str = f"✗ NG (不合格 · {count}处缺陷)"

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
        data_dir = cfg.get("data_dir", "data")
        conn = get_conn(data_dir)
        save_qc_result(
            conn,
            persist_qc_image(image_path, Path(data_dir) / "qc_images"),
            verdict, result["detections"], max_score, prompt)
        conn.close()
    except Exception as e:
        logger.warning("QC 结果落库失败: %s", e)

    status = f"检测完成 · 提示词: {prompt[:60]}... · 阈值: {threshold}"
    yield (result["image"], verdict_str, f"{max_score:.2%}",
           str(count), table, status,
           log(f"✓ {verdict_str} · 最高置信度 {max_score:.2%} · {count} 处"))


def _explain_union():
    """AI 缺陷解释: 复用最近一次 Union 结果, ROI 裁切 → 本地 VLM 识读。"""
    if _last_union is None:
        return [], "⚠ 请先运行一次 **Union 零漏检** 检测。"
    if _registry is None:
        return [], "⚠ 引擎未初始化。"

    from core.vlm_explain import explain_union
    try:
        out = explain_union(_registry, _last_union["image_path"],
                            _last_union["result"], _get_config())
    except Exception as e:  # noqa: BLE001
        return [], f"⚠ 解释过程异常: {e}"

    if out.get("error"):
        return out.get("crops", []), f"⚠ {out['error']}"
    crops = out.get("crops", [])
    summary = out.get("summary", "") or "(VLM 未返回内容)"
    n = len(out.get("rois", []))
    return crops, f"**已分析 {n} 个候选区域**\n\n{summary}"


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
        data_dir = cfg.get("data_dir", "data")
        conn = get_conn(data_dir)
        save_qc_result(
            conn,
            persist_qc_image(image_path, Path(data_dir) / "qc_images"),
            verdict, fused["fused_defects"],
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
    if result.get("dinov2"):
        msg += (f"\n- DINOv2 特征库: {result['dinov2'].get('n_etalons', 0)} "
                f"个原型 (Union 第4源)\n"
                f"  保存位置: {result.get('dinov2_saved_to', '')}")
    elif result.get("dinov2_error"):
        msg += f"\n- DINOv2 特征库: 建立失败 ({result['dinov2_error']}), 不影响 PatchCore"

    # NP 校准状态 + 融合阶段提示 (§6.2 校准协议入口)
    n_cals = [calibrated_n_samples(registry.get(n))
              for n in ("anomalib", "dinov2_anomaly")]
    n_cals = [n for n in n_cals if n]
    if n_cals:
        n_min = min(n_cals)
        stage = fusion_stage(n_min, (_get_config().get("qc", {}) or {})
                             .get("union", {}).get("fusion"))
        msg += f"\n- NP 校准: n_cal={n_min} → 融合 Stage {stage}"
        if stage < 2:
            msg += (f"\n\n⚠ n_cal<10: 融合处于 Stage 1 (纯 OR, 误报偏高)。"
                    f"请执行下方「校准协议」: 补采 ≥{MIN_CAL_RECOMMENDED} 张"
                    f"独立 OK 图 (变换光照/角度), 升级双源互证降误报。")
    return msg, gr.update(choices=banks, value=product)


def _register_subspace_bank(sa_product, sa_product_name, files):
    """注册 OK 样本, 构建 SubspaceAD 子空间库 (辅助通道)。"""
    if sa_product and sa_product != "(新建)":
        product = sa_product
    elif sa_product_name and sa_product_name.strip():
        product = sa_product_name.strip()
    else:
        return "⚠ 请选择已有产品或输入新产品名称。", gr.update()

    if not files:
        return "⚠ 请上传 OK 样本图片 (快速换线 1~4 张, 标准建库建议 ≥10 张)。", gr.update()

    paths = []
    for f in files:
        p = f.name if hasattr(f, "name") else str(f)
        if os.path.isfile(p):
            paths.append(p)
    if not paths:
        return "⚠ 无有效图片。", gr.update()

    registry = _registry
    if registry is None:
        return "⚠ 引擎未初始化。", gr.update()

    result = register_subspace_bank(registry, product, paths)
    if result.get("error"):
        return f"⚠ 建库失败: {result['error']}", gr.update()

    banks = ["(新建)"] + list_banks_subspace()
    fast = result.get("mode") == "fast"
    mode_txt = ("快速换线模式 (旋转增广)" if fast else "标准模式")
    msg = (f"✓ 产品「{product}」SubspaceAD 特征库已建立 ({mode_txt})\n\n"
           f"- OK 样本: {result.get('n_images', 0)} 张\n"
           f"- 增广视图入池: {result.get('n_augmented', 0)} 个\n"
           f"- PCA 子空间: {result.get('pca_k', 0)} 维 "
           f"(累计解释方差 {result.get('pca_ev_achieved', 0):.3f})\n"
           f"- 保存位置: {result.get('saved_to', '')}")
    if fast:
        msg += ("\n\n⚠ 快速换线模式仅为辅助提示: 自校准偏乐观, "
                "检测结果显示\"仅供参考\", 须人工复核; "
                "正式量产请补足 ≥10 张 OK 图并改用 PatchCore 主判。")
    return msg, gr.update(choices=banks, value=product)


def _run_calibration(cal_product, cal_files, ng_files):
    """执行校准协议 (§6.2): 独立校准图重标定 NP 阈值 + 验收报告。"""
    if not cal_product or cal_product == "(新建)":
        return "⚠ 请选择待校准的产品 (须已建库)。"

    def _paths(files):
        out = []
        for f in (files or []):
            p = f.name if hasattr(f, "name") else str(f)
            if os.path.isfile(p):
                out.append(p)
        return out

    cal_paths = _paths(cal_files)
    if len(cal_paths) < 3:
        return (f"⚠ 有效校准图仅 {len(cal_paths)} 张, 至少 3 张 "
                f"(建议 ≥{MIN_CAL_RECOMMENDED} 张, 变换光照/角度拍 3 组)。")

    registry = _registry
    if registry is None:
        return "⚠ 引擎未初始化。"

    fusion_cfg = (_get_config().get("qc", {}) or {}).get(
        "union", {}).get("fusion")
    try:
        result = recalibrate_product(
            registry, cal_product, cal_paths,
            ng_image_paths=_paths(ng_files), fusion_cfg=fusion_cfg)
    except Exception as e:  # noqa: BLE001 — UI 兜底
        return f"⚠ 校准协议执行异常: {e}"
    return format_report_md(result)
