"""缺陷检测流程编排模块 (v1.5.0 拆分后仅保留检测流程)

职责:
- 调用 Grounding DINO 引擎执行零样本检测 (run_detection)
- PatchCore 异常检测 (run_anomaly_detection)
- 四源 Union OR 检测 (run_union_detection)

拆分说明 (v1.5.0):
- 产品配方/提示词翻译 → core.recipes
- 检测框标注/印章/热力图/尺寸过滤 → core.qc_drawing
- 图片持久化/结果落库 → core.qc_persist
本模块 re-export 上述符号, 保持既有 import 向后兼容。
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from core.fusion import (calibrated_n_samples, get_drift_monitor,
                         staged_fusion)
from core.imutils import imread_unicode
# ─── 向后兼容 re-export (v1.5.0 拆分, 勿在本文件重复实现) ─────
from core.recipes import (  # noqa: F401
    DEFAULT_PROMPT, _RECIPES_DIR, _recipe_path, _safe_name,
    delete_recipe, list_recipes, load_recipe, save_recipe,
    translate_prompt)
from core.qc_drawing import (  # noqa: F401
    _DEFECT_COLORS, _bbox_area, _draw_detections, _filter_by_size,
    _overlay_heatmap, _pick_color, draw_verdict_badge)
from core.qc_persist import persist_qc_image, save_qc_result  # noqa: F401

logger = logging.getLogger("visionocr.defect")


# ─── 检测 + 标注 ────────────────────────────────────────────
def run_detection(registry, image_path: str, prompt: str = "",
                  threshold: float = 0.3,
                  size_cfg: dict | None = None) -> dict:
    """执行缺陷检测并返回标注结果。

    Args:
        registry: EngineRegistry 实例
        image_path: 图像文件路径
        prompt: 缺陷描述词 (点分隔)
        threshold: 置信度阈值
        size_cfg: 瑕疵尺寸过滤配置 (config.yaml 中 qc.defect_size 段)

    Returns:
        {"image": np.ndarray (BGR标注后), "verdict": "OK"/"NG",
         "detections": [...], "count": int, "max_score": float,
         "rejected_by_size": int}
    """
    # 读取图像
    img = imread_unicode(image_path)
    if img is None:
        return {"image": None, "verdict": "ERROR", "detections": [],
                "count": 0, "max_score": 0, "error": "无法读取图像"}

    # 获取引擎
    engine = registry.get("grounding_dino")
    if engine is None:
        return {"image": None, "verdict": "ERROR", "detections": [],
                "count": 0, "max_score": 0, "error": "Grounding DINO 引擎未注册"}

    # 确保加载
    if not engine.is_ready():
        registry.ensure_loaded("grounding_dino")
    if not engine.is_ready():
        return {"image": None, "verdict": "ERROR", "detections": [],
                "count": 0, "max_score": 0, "error": "模型加载失败"}

    # 推理 (中文提示词自动翻译为英文)
    if not prompt.strip():
        prompt = DEFAULT_PROMPT
    en_prompt = translate_prompt(prompt)
    from core.infer_stats import Timer
    # infer 租约: 推理期间引擎不会被 LRU 驱逐/空闲卸载 (v1.5.0)
    # ndarray 直通: 避免引擎内部重复解码磁盘文件 (v1.5.0)
    with registry.lease("grounding_dino"):
        with Timer("grounding_dino"):
            result = engine.infer(img, prompt=en_prompt,
                                  threshold=threshold)

    if result.get("error"):
        return {"image": img, "verdict": "ERROR", "detections": [],
                "count": 0, "max_score": 0, "error": result["error"]}

    boxes = result["boxes"]
    labels = result["labels"]
    scores = result["scores"]

    # 瑕疵尺寸过滤 (过滤噪点/背景误检)
    boxes, labels, scores, rejected_by_size = _filter_by_size(
        boxes, labels, scores, size_cfg)

    # 标注图像
    annotated = _draw_detections(img, boxes, labels, scores)

    # 判定: 有检测框 → NG
    verdict = "NG" if len(boxes) > 0 else "OK"
    max_score = max(scores) if scores else 0.0

    detections = [
        {"box": b, "label": l, "score": s, "area_px": round(_bbox_area(b), 1)}
        for b, l, s in zip(boxes, labels, scores)
    ]

    return {
        "image": annotated,
        "verdict": verdict,
        "detections": detections,
        "count": len(boxes),
        "max_score": round(max_score, 4),
        "rejected_by_size": rejected_by_size,
    }


# ─── PatchCore 异常检测 (生产主力路径) ─────────────────────────
def run_anomaly_detection(registry, image_path: str,
                          threshold: float | None = None) -> dict:
    """执行 PatchCore 异常检测, 返回热力图标注结果。

    与 run_detection (Grounding DINO bbox) 不同, 本函数基于正常样本记忆库,
    检测任何偏离正常模式的区域。适用于表面缺陷 (划痕/凹陷/色差等)。

    前置条件: anomalib 引擎已加载且记忆库已建立 (train 或 load_bank)。

    Args:
        registry: EngineRegistry 实例
        image_path: 图像文件路径
        threshold: 异常分数阈值 (None=使用配置, 保守模式自动减半)

    Returns:
        {"image": np.ndarray (BGR+热力图), "verdict": "OK"/"NG",
         "score": float, "anomaly_map": np.ndarray, "grid_size": int,
         "threshold_used": float}
    """
    img = imread_unicode(image_path)
    if img is None:
        return {"image": None, "verdict": "ERROR", "score": 0,
                "anomaly_map": None, "error": "无法读取图像"}

    engine = registry.get("anomalib")
    if engine is None:
        return {"image": None, "verdict": "ERROR", "score": 0,
                "anomaly_map": None, "error": "PatchCore 引擎未注册"}

    if not engine.is_ready():
        registry.ensure_loaded("anomalib")
    if not engine.is_ready():
        return {"image": None, "verdict": "ERROR", "score": 0,
                "anomaly_map": None, "error": "PatchCore 模型加载失败"}

    if not engine.has_bank:
        return {"image": None, "verdict": "ERROR", "score": 0,
                "anomaly_map": None,
                "error": "记忆库为空, 请先注册OK样本 (train/load_bank)"}

    # 推理 (infer 租约保护, v1.5.0; ndarray 直通避免重复解码)
    kwargs = {}
    if threshold is not None:
        kwargs["threshold"] = threshold
    with registry.lease("anomalib"):
        result = engine.infer(img, **kwargs)

    if result.get("error"):
        return {"image": img, "verdict": "ERROR", "score": 0,
                "anomaly_map": None, "error": result["error"]}

    score = result["score"]
    anomaly_map = result["anomaly_map"]
    pred = result["pred_label"]
    thresh_used = result.get("threshold_used", 0)

    # 叠加热力图到原图
    annotated = _overlay_heatmap(img, anomaly_map)

    # 大印章
    draw_verdict_badge(annotated, pred)

    return {
        "image": annotated,
        "verdict": pred,
        "score": score,
        "anomaly_map": anomaly_map,
        "grid_size": result.get("grid_size", 0),
        "threshold_used": thresh_used,
        "calibrated_score": result.get("calibrated_score"),
        "np_p_value": result.get("np_p_value"),
    }


# ─── 四源 Union 检测 (零漏检架构) ─────────────────────────────
def run_union_detection(registry, image_path: str,
                        prompt: str = "",
                        threshold: float | None = None,
                        size_cfg: dict | None = None,
                        config: dict | None = None,
                        product_name: str = "") -> dict:
    """四源 Union OR 检测: PatchCore + Grounding DINO + YOLO + DINOv2。

    零漏检架构:
    - PatchCore (表面异常): 划痕/凹陷/色差/污渍等, 基于正常样本记忆库
    - Grounding DINO (结构缺陷): 缺件/错位/标签歪等, 基于文本提示词
    - YOLO (结构缺陷): 缺孔/短路/毛刺等, 少样本微调; 受产品门控,
      仅当前产品有专属权重 (models/yolo/{product}.pt) 时激活, 防跨域误报
    - DINOv2 (表面异常): 自监督 ViT 特征 + GMM 分布建模, 与 PatchCore
      特征空间互补, 提升漏检覆盖; 需 OK 样本建库
    - Union OR: 任一模型判定 NG → 最终 NG (宁可误报, 不可漏检)
    - 误报由人工复核兜底 (产线标准流程)
    - v1.3.0 P0-6: PatchCore 与 DINOv2 推理并行执行 (ThreadPoolExecutor),
      结构源 (GDINO/YOLO) 保持串行, 热路径提速

    Args:
        registry: EngineRegistry 实例
        image_path: 图像文件路径
        prompt: Grounding DINO 缺陷描述词 (空=跳过 DINO)
        threshold: 异常分数阈值 (None=使用配置)
        size_cfg: 瑕疵尺寸过滤配置
        config: 全局配置字典 (读取 qc.union 段)
        product_name: 当前产品名 (YOLO 门控 + PatchCore/DINOv2 特征库
            自动加载用; 空/占位符=跳过 YOLO)

    Returns:
        {"image": np.ndarray, "verdict": "OK"/"NG",
         "patchcore": {...} | None, "dino": {...} | None,
         "yolo": {...} | None, "dinov2": {...} | None,
         "ng_sources": [...]}
        ng_sources 可含 "patchcore"/"dino"/"yolo"/"dinov2"
    """
    import cv2

    img = imread_unicode(image_path)
    if img is None:
        return {"image": None, "verdict": "ERROR",
                "error": "无法读取图像", "ng_sources": []}

    qc_cfg = (config or {}).get("qc", {})
    union_cfg = qc_cfg.get("union", {})
    enable_patchcore = union_cfg.get("enable_patchcore", True)
    enable_dino = union_cfg.get("enable_dino", True)
    enable_yolo = union_cfg.get("enable_yolo", True)
    enable_dinov2 = union_cfg.get("enable_dinov2", True)

    pc_result = None
    dino_result = None
    yolo_result = None
    dv_result = None
    ng_sources = []

    # ── 1) PatchCore 表面异常检测 (准备阶段; 推理与 DINOv2 并行, P0-6) ──
    pc_engine = None
    pc_kwargs = {}
    if enable_patchcore:
        engine = registry.get("anomalib")
        if engine is not None:
            if not engine.is_ready():
                registry.ensure_loaded("anomalib")
            # 自动加载产品特征库 (持久化 bank 在 Union 模式下直接生效)
            if engine.is_ready() and not engine.has_bank:
                from core.anomaly_bank import (load_product_bank,
                                               list_banks)
                if product_name:
                    load_product_bank(registry, product_name)
                else:
                    # 无产品上下文时, 尝试自动发现唯一特征库
                    available = list_banks()
                    if len(available) == 1:
                        logger.info("Union: PatchCore 自动加载唯一特征库「%s」",
                                    available[0])
                        load_product_bank(registry, available[0])
                    elif len(available) > 1:
                        logger.warning(
                            "Union: PatchCore 存在 %d 个特征库 [%s], "
                            "无法自动选择, 请在工程师面板指定产品",
                            len(available), ", ".join(available))
            if engine.is_ready() and engine.has_bank:
                pc_engine = engine
                if threshold is not None:
                    pc_kwargs["threshold"] = threshold
            else:
                logger.debug("Union: PatchCore 跳过 (未就绪或无记忆库)")

    # ── 1c) DINOv2 表面异常检测 (准备阶段; 上提与 PatchCore 配对并行, P0-6) ──
    dv_engine = None
    dv_kwargs = {}
    if enable_dinov2:
        dv_eng = registry.get("dinov2_anomaly")
        if dv_eng is not None:
            if not dv_eng.is_ready():
                registry.ensure_loaded("dinov2_anomaly")
            if dv_eng.is_ready() and not dv_eng.has_bank:
                from core.anomaly_bank import (load_product_bank_dinov2,
                                               list_banks_dinov2)
                if product_name:
                    load_product_bank_dinov2(registry, product_name)
                else:
                    # 无产品上下文时, 尝试自动发现唯一 DINOv2 特征库
                    available_dv = list_banks_dinov2()
                    if len(available_dv) == 1:
                        logger.info("Union: DINOv2 自动加载唯一特征库「%s」",
                                    available_dv[0])
                        load_product_bank_dinov2(registry, available_dv[0])
                    elif len(available_dv) > 1:
                        logger.warning(
                            "Union: DINOv2 存在 %d 个特征库 [%s], "
                            "无法自动选择, 请在工程师面板指定产品",
                            len(available_dv), ", ".join(available_dv))
            if dv_eng.is_ready() and dv_eng.has_bank:
                dv_engine = dv_eng
                if threshold is not None:
                    dv_kwargs["threshold"] = threshold
            else:
                logger.debug("Union: DINOv2 跳过 (未就绪或无特征库)")

    # ── 1x) 表面双源并行推理 (P0-6: PatchCore ∥ DINOv2, 热路径提速) ──
    # 准备阶段 (ensure_loaded/特征库加载) 已在主线程串行完成,
    # 推理本身无 registry 状态变更, 两引擎相互独立, 线程安全。
    # infer 租约: 推理窗口内禁止驱逐/卸载参与引擎 (v1.5.0)
    # ndarray 直通: 传入已解码 img, 避免各引擎重复读盘 (v1.5.0)
    _leased: list[str] = []
    if pc_engine is not None:
        registry.acquire_lease("anomalib")
        _leased.append("anomalib")
    if dv_engine is not None:
        registry.acquire_lease("dinov2_anomaly")
        _leased.append("dinov2_anomaly")
    try:
        if pc_engine is not None and dv_engine is not None:
            with ThreadPoolExecutor(max_workers=2,
                                    thread_name_prefix="union") as ex:
                f_pc = ex.submit(pc_engine.infer, img, **pc_kwargs)
                f_dv = ex.submit(dv_engine.infer, img, **dv_kwargs)
                # 单源异常不得拖垮整体: 置 error 结果, 下游统一降级处理
                try:
                    pc_result = f_pc.result()
                except Exception as e:
                    logger.warning("Union/PatchCore 推理异常: %s", e)
                    pc_result = {"error": str(e)}
                try:
                    dv_result = f_dv.result()
                except Exception as e:
                    logger.warning("Union/DINOv2 推理异常: %s", e)
                    dv_result = {"error": str(e)}
        elif pc_engine is not None:
            try:
                pc_result = pc_engine.infer(img, **pc_kwargs)
            except Exception as e:
                logger.warning("Union/PatchCore 推理异常: %s", e)
                pc_result = {"error": str(e)}
        elif dv_engine is not None:
            try:
                dv_result = dv_engine.infer(img, **dv_kwargs)
            except Exception as e:
                logger.warning("Union/DINOv2 推理异常: %s", e)
                dv_result = {"error": str(e)}
    finally:
        for _n in _leased:
            registry.release_lease(_n)

    # PatchCore 后处理 (保持原 ng_sources 追加顺序)
    if pc_result is not None:
        if pc_result.get("error"):
            logger.warning("Union/PatchCore 错误: %s", pc_result["error"])
            pc_result = None
        elif pc_result.get("pred_label") == "NG":
            ng_sources.append("patchcore")

    # ── 2) Grounding DINO 结构缺陷检测 ──
    if enable_dino and prompt.strip():
        dino_result = run_detection(registry, image_path,
                                    prompt=prompt,
                                    threshold=qc_cfg.get(
                                        "confidence_threshold", 0.3),
                                    size_cfg=size_cfg)
        if dino_result.get("error"):
            logger.warning("Union/DINO 错误: %s", dino_result.get("error"))
            dino_result = None
        elif dino_result.get("verdict") == "NG":
            ng_sources.append("dino")

    # ── 2b) YOLO 结构缺陷检测 (产品门控: 仅当前产品有专属权重时激活) ──
    # 跨域误报防护: YOLO 检测训练集标注的缺陷类别, 跨产品会大量误报,
    # 故无产品上下文或该产品未训练 YOLO 时, 跳过本检测源。
    if enable_yolo:
        yolo_eng = registry.get("yolo_defect")
        if yolo_eng is not None and yolo_eng.load_for_product(product_name):
            from core.infer_stats import Timer
            with registry.lease("yolo_defect"):
                with Timer("yolo_defect"):
                    try:
                        yolo_result = yolo_eng.infer(img)
                    except Exception as e:
                        logger.warning("Union/YOLO 推理异常: %s", e)
                        yolo_result = {"error": str(e)}
            if yolo_result.get("error"):
                logger.warning("Union/YOLO 错误: %s", yolo_result["error"])
                yolo_result = None
            elif yolo_result.get("count", 0) > 0:
                ng_sources.append("yolo")
        else:
            logger.debug("Union: YOLO 跳过 (产品「%s」无专属权重)",
                         product_name or "<无>")

    # ── 2c) DINOv2 后处理 (推理已在 1x 并行完成, 保持原 ng_sources 顺序) ──
    if dv_result is not None:
        if dv_result.get("error"):
            logger.warning("Union/DINOv2 错误: %s", dv_result["error"])
            dv_result = None
        elif dv_result.get("pred_label") == "NG":
            ng_sources.append("dinov2")

    # ── 3) 分阶段融合判定 (v1.4.0 §5.5; qc.union.fusion.mode=or 回退纯OR) ──
    # 安全守卫: 如果所有引擎都被跳过, 警告用户 OK 不可信
    _any_participated = (pc_result is not None or
                         dino_result is not None or
                         yolo_result is not None or
                         dv_result is not None)
    if not _any_participated and not ng_sources:
        logger.warning(
            "Union: 所有检测引擎均被跳过, verdict=OK 不可信! "
            "请确认: (1) 已选择产品 (2) 已建 OK 样本特征库 "
            "(3) GroundingDINO prompt 非空")

    # 各校准源的 NP 校准样本数 (决定融合阶段; 未参与/未拟合 → None)
    n_cal_by_source = {
        "patchcore": (calibrated_n_samples(pc_engine)
                      if pc_result is not None else None),
        "dinov2": (calibrated_n_samples(dv_engine)
                   if dv_result is not None else None),
    }
    fusion_cfg = union_cfg.get("fusion", {})
    fused = staged_fusion(ng_sources, n_cal_by_source, fusion_cfg)
    verdict = fused["verdict"]
    if verdict == "REVIEW":
        logger.info("Union: REVIEW 黄牌待复核 (%s)",
                    "; ".join(fused["review_reasons"]) or "单源孤证")
    if fused.get("fallback_or"):
        logger.info("Union: 融合回退 OR (%s)",
                    "; ".join(fused["review_reasons"]))

    # Stage 3 漂移监控: 对每张图观测校准分数 (不只 NG 图), 仅预警不改判决
    if fused["stage"] == 3:
        _mon = get_drift_monitor()
        for _src, _res, _eng in (("patchcore", pc_result, pc_engine),
                                 ("dinov2", dv_result, dv_engine)):
            if _res is None or _eng is None:
                continue
            _cal = getattr(_eng, "_np_calibrator", None)
            if _cal is None or not getattr(_cal, "is_fitted", False):
                continue
            _warn = _mon.observe(
                f"{product_name or '默认'}/{_src}",
                float(_res.get("score", 0.0)),
                float(_cal.threshold),
                float(getattr(_cal, "epsilon", 0.10)))
            if _warn:
                logger.warning("Union 漂移监控: %s", _warn)

    # ── 4) 合成标注图 ──
    annotated = img.copy()

    # 叠加热力图 (PatchCore + DINOv2: 归一化后逐像素取max, 一次叠加)
    # merged_map 同时暴露给下游 (VLM ROI 裁切定位)
    h0, w0 = img.shape[:2]
    merged_map = None
    heat_maps = []
    if pc_result and pc_result.get("anomaly_map") is not None:
        heat_maps.append(pc_result["anomaly_map"])
    if dv_result and dv_result.get("anomaly_map") is not None:
        heat_maps.append(dv_result["anomaly_map"])
    if len(heat_maps) == 1:
        merged_map = cv2.resize(heat_maps[0].astype(np.float32), (w0, h0))
        annotated = _overlay_heatmap(annotated, merged_map, alpha=0.35)
    elif len(heat_maps) == 2:
        merged_map = np.maximum(
            cv2.resize(heat_maps[0].astype(np.float32), (w0, h0)),
            cv2.resize(heat_maps[1].astype(np.float32), (w0, h0)))
        annotated = _overlay_heatmap(annotated, merged_map, alpha=0.35)

    # 叠加检测框 (Grounding DINO)
    if dino_result and dino_result.get("detections"):
        boxes = [d["box"] for d in dino_result["detections"]]
        labels = [d["label"] for d in dino_result["detections"]]
        scores = [d["score"] for d in dino_result["detections"]]
        annotated = _draw_detections(annotated, boxes, labels, scores)

    # 叠加检测框 (YOLO 结构缺陷)
    if yolo_result and yolo_result.get("boxes"):
        annotated = _draw_detections(annotated,
                                     yolo_result["boxes"],
                                     yolo_result["labels"],
                                     yolo_result["scores"])

    # 统一印章 (覆盖 _draw_detections 内部的印章)
    det_count = (dino_result.get("count", 0) if dino_result else 0) + \
                (yolo_result.get("count", 0) if yolo_result else 0)
    draw_verdict_badge(annotated, verdict, count=det_count)

    # ── 5) 组装结果 ──
    result = {
        "image": annotated,
        "verdict": verdict,
        "ng_sources": ng_sources,
        "patchcore": None,
        "dino": None,
        "yolo": None,
        "dinov2": None,
        # 融合热力图 (全分辨率 float32, 无表面源时为 None) —
        # 供 VLM ROI 裁切等下游定位使用
        "anomaly_map": merged_map,
        # 分阶段融合元数据 (v1.4.0 §5.5)
        "fusion": {
            "stage": fused["stage"],
            "mode": fused["mode"],
            "n_cal": fused["n_cal"],
            "review_required": fused["review_required"],
            "review_reasons": fused["review_reasons"],
            "fallback_or": fused["fallback_or"],
        },
    }

    if pc_result:
        result["patchcore"] = {
            "score": pc_result.get("score", 0),
            "pred_label": pc_result.get("pred_label", "?"),
            "threshold_used": pc_result.get("threshold_used", 0),
            "calibrated_score": pc_result.get("calibrated_score"),
        }

    if dino_result:
        result["dino"] = {
            "verdict": dino_result.get("verdict", "?"),
            "count": dino_result.get("count", 0),
            "max_score": dino_result.get("max_score", 0),
            "detections": dino_result.get("detections", []),
            "rejected_by_size": dino_result.get("rejected_by_size", 0),
        }

    if yolo_result:
        result["yolo"] = {
            "count": yolo_result.get("count", 0),
            "max_score": yolo_result.get("max_score", 0),
            "boxes": yolo_result.get("boxes", []),
            "labels": yolo_result.get("labels", []),
            "scores": yolo_result.get("scores", []),
        }

    if dv_result:
        result["dinov2"] = {
            "score": dv_result.get("score", 0),
            "pred_label": dv_result.get("pred_label", "?"),
            "threshold_used": dv_result.get("threshold_used", 0),
            "calibrated_score": dv_result.get("calibrated_score"),
        }

    logger.info("Union 检测: verdict=%s, sources=%s, pc_score=%.4f, "
                "dino_count=%d, yolo_count=%d, dv_score=%.4f",
                verdict, ng_sources,
                pc_result.get("score", 0) if pc_result else -1,
                dino_result.get("count", 0) if dino_result else -1,
                yolo_result.get("count", 0) if yolo_result else -1,
                dv_result.get("score", 0) if dv_result else -1)

    return result
