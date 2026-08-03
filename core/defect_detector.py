"""缺陷检测高层逻辑 (Phase 4A)

职责:
- 调用 Grounding DINO 引擎执行零样本检测
- 在图像上绘制检测框 + 标签
- OK/NG 判定 (有缺陷 → NG, 无缺陷 → OK)
- 产品配方管理 (每个产品保存缺陷描述词, 一键切换)
- 检测结果落库 (qc_results 表)
- 中文提示词自动翻译为英文 (Grounding DINO 仅支持英文 BERT)
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from core.fusion import (calibrated_n_samples, get_drift_monitor,
                         staged_fusion)
from core.imutils import imread_unicode

logger = logging.getLogger("visionocr.defect")

# ─── 中英缺陷词对照表 (工业外观常见) ─────────────────────────
_ZH_EN_MAP = {
    "划痕": "scratch", "刮伤": "scratch", "划伤": "scratch",
    "凹陷": "dent", "凹坑": "dent", "压痕": "dent",
    "裂纹": "crack", "裂缝": "crack", "开裂": "crack",
    "污渍": "stain", "脏污": "stain", "污点": "stain",
    "毛刺": "burr", "飞边": "burr",
    "色差": "color difference", "变色": "discoloration",
    "缺件": "missing part", "缺失": "missing", "漏装": "missing component",
    "变形": "deformation", "翘曲": "warp", "弯曲": "bend",
    "气泡": "bubble", "气孔": "porosity", "砂眼": "blowhole",
    "锈": "rust", "锈蚀": "rust", "氧化": "oxidation",
    "磨损": "wear", "磨伤": "abrasion",
    "异物": "foreign object", "杂质": "impurity",
    "错位": "misalignment", "偏移": "offset", "倾斜": "tilt",
    "破损": "damage", "断裂": "fracture", "缺口": "notch",
    "溢胶": "glue overflow", "胶渍": "glue residue",
    "焊渣": "solder spatter", "虚焊": "cold solder joint",
    "短路": "short circuit", "断路": "open circuit",
    "标签歪": "misaligned label", "贴歪": "crooked label",
    "印刷不良": "print defect", "漏印": "missing print",
    "缩水": "shrinkage", "飞料": "flash",
    "缺陷": "defect", "不良": "defect", "异常": "anomaly",
}

# 默认提示词 (中文界面, 内部自动翻译为英文)
DEFAULT_PROMPT = "划痕.凹陷.裂纹.污渍.毛刺.色差.缺件.变形"

# 产品配方存储路径
_RECIPES_DIR = Path("data/recipes")


def _safe_name(name: str) -> str:
    """清洗产品配方名，防止路径穿越攻击。"""
    import re
    s = re.sub(r'[\\/:*?"<>|.]', '_', str(name).strip())
    return s or '_'


def _recipe_path(name: str) -> Path:
    """构造并校验配方路径: 优先清洗名, 旧文件用原始名回退兼容 (均不越界)。"""
    root = _RECIPES_DIR.resolve()
    p = (_RECIPES_DIR / f"{_safe_name(name)}.json").resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"路径越界: {name} 非法")
    if p.exists():
        return p
    # 回退兼容 v1.4.1 前已存在的旧文件 (名称含 . 等特殊字符)
    legacy = (_RECIPES_DIR / f"{str(name).strip()}.json").resolve()
    if legacy.is_relative_to(root) and legacy.exists():
        return legacy
    return p


def translate_prompt(prompt: str) -> str:
    """将中文缺陷提示词翻译为英文 (点号分隔)。

    规则:
    - 逐词查表, 命中则替换为英文
    - 未命中且含中文 → 保留原文 (模型可能部分识别)
    - 已是英文 → 原样保留
    """
    terms = [t.strip() for t in prompt.replace("。", ".").split(".") if t.strip()]
    translated = []
    for term in terms:
        en = _ZH_EN_MAP.get(term)
        if en:
            translated.append(en)
        else:
            translated.append(term)
    return ".".join(translated)


# ─── 产品配方 ────────────────────────────────────────────────
def list_recipes() -> list[str]:
    """列出所有已保存的产品配方名。"""
    if not _RECIPES_DIR.exists():
        return []
    return sorted(p.stem for p in _RECIPES_DIR.glob("*.json"))


def load_recipe(name: str) -> Optional[dict]:
    """加载产品配方。"""
    try:
        p = _recipe_path(name)
    except ValueError:
        return None
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_recipe(name: str, prompt: str, threshold: float = 0.3,
                note: str = "",
                min_area_px: int = 0, max_area_px: int = 0,
                pixels_per_mm: float = 0.0) -> None:
    """保存产品配方 (含瑕疵尺寸阈值)。"""
    _RECIPES_DIR.mkdir(parents=True, exist_ok=True)
    p = _recipe_path(name)
    data = {
        "name": p.stem,
        "prompt": prompt,
        "threshold": threshold,
        "note": note,
        "defect_size": {
            "min_area_px": min_area_px,
            "max_area_px": max_area_px,
            "pixels_per_mm": pixels_per_mm,
        },
    }
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_recipe(name: str) -> bool:
    """删除产品配方。"""
    try:
        p = _recipe_path(name)
    except ValueError:
        return False
    if p.exists():
        p.unlink()
        return True
    return False


# ─── 瑕疵尺寸过滤 ─────────────────────────────────────────────
def _bbox_area(box) -> float:
    """计算检测框面积 (像素²)。box = [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1) * (y2 - y1))


def _filter_by_size(boxes: list, labels: list, scores: list,
                    size_cfg: dict | None = None) -> tuple[list, list, list, int]:
    """按面积阈值过滤检测结果。

    Args:
        boxes/labels/scores: Grounding DINO 原始输出
        size_cfg: defect_size 配置字典, None 或 enabled=False 时不过滤

    Returns:
        (filtered_boxes, filtered_labels, filtered_scores, rejected_count)
    """
    if not size_cfg or not size_cfg.get("enabled", False):
        return boxes, labels, scores, 0

    min_px = size_cfg.get("min_area_px", 0)
    max_px = size_cfg.get("max_area_px", float("inf"))
    min_mm2 = size_cfg.get("min_area_mm2", 0.0)
    max_mm2 = size_cfg.get("max_area_mm2", 0.0)
    px_per_mm = size_cfg.get("pixels_per_mm", 0.0)

    # 物理面积阈值转换为像素面积 (需要标定系数)
    if px_per_mm > 0:
        px2_per_mm2 = px_per_mm * px_per_mm
        if min_mm2 > 0:
            min_px = max(min_px, min_mm2 * px2_per_mm2)
        if max_mm2 > 0:
            max_px = min(max_px, max_mm2 * px2_per_mm2)

    kept_boxes, kept_labels, kept_scores = [], [], []
    rejected = 0
    for b, l, s in zip(boxes, labels, scores):
        area = _bbox_area(b)
        if area < min_px or area > max_px:
            rejected += 1
            logger.debug("尺寸过滤: area=%.0f px² (范围 %.0f~%.0f), label=%s",
                         area, min_px, max_px, l)
        else:
            kept_boxes.append(b)
            kept_labels.append(l)
            kept_scores.append(s)

    if rejected:
        logger.info("尺寸过滤: %d/%d 个检测被过滤 (面积不在阈值内)",
                    rejected, len(boxes))
    return kept_boxes, kept_labels, kept_scores, rejected


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
    import cv2

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
    with Timer("grounding_dino"):
        result = engine.infer(image_path, prompt=en_prompt, threshold=threshold)

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
    import cv2

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

    # 推理
    kwargs = {}
    if threshold is not None:
        kwargs["threshold"] = threshold
    result = engine.infer(image_path, **kwargs)

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


def _overlay_heatmap(img: np.ndarray, anomaly_map: np.ndarray,
                     alpha: float = 0.4) -> np.ndarray:
    """将异常热力图叠加到原图上 (JET colormap)。"""
    import cv2

    h, w = img.shape[:2]
    # 上采样热力图到原图尺寸
    heat = cv2.resize(anomaly_map.astype(np.float32), (w, h),
                      interpolation=cv2.INTER_CUBIC)
    # 归一化到 0~255
    heat = np.clip(heat * 255, 0, 255).astype(np.uint8)
    # JET colormap
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    # 叠加
    blended = cv2.addWeighted(img, 1 - alpha, heat_color, alpha, 0)
    return blended


# ─── 双模型 Union 检测 (零漏检架构) ─────────────────────────────
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
    if pc_engine is not None and dv_engine is not None:
        with ThreadPoolExecutor(max_workers=2,
                                thread_name_prefix="union") as ex:
            f_pc = ex.submit(pc_engine.infer, image_path, **pc_kwargs)
            f_dv = ex.submit(dv_engine.infer, image_path, **dv_kwargs)
            pc_result = f_pc.result()
            dv_result = f_dv.result()
    elif pc_engine is not None:
        pc_result = pc_engine.infer(image_path, **pc_kwargs)
    elif dv_engine is not None:
        dv_result = dv_engine.infer(image_path, **dv_kwargs)

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
            with Timer("yolo_defect"):
                yolo_result = yolo_eng.infer(image_path)
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
    total_ng = len(ng_sources)
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


def draw_verdict_badge(img: np.ndarray, verdict: str, count: int = 0,
                       alpha: float = 0.75) -> np.ndarray:
    """在图像右上角绘制大面积 OK/NG 印章, 供工人一眼判定。

    Args:
        img:     BGR 图像 (会被原地修改)。
        verdict: "OK" / "NG" / "REVIEW" (黄牌待复核, v1.4.0 分阶段融合)。
        count:   缺陷数量 (NG 时显示)。
        alpha:   印章背景不透明度。

    Returns:
        修改后的图像 (同一引用)。
    """
    import cv2

    h, w = img.shape[:2]
    # 印章尺寸随图像等比缩放
    badge_h = max(60, h // 8)
    font_scale = badge_h / 55.0
    thickness_txt = max(3, int(font_scale * 2.5))

    _v = verdict.upper()
    if "REVIEW" in _v:
        # 黄牌: 单源孤证等可疑图 — 非 NG 拦截, 强制人工复核 (ASCII 避免
        # cv2.putText 中文乱码)
        text = "REVIEW"
        bg_color = (0, 200, 255)     # BGR 黄橙 (黄牌)
    elif "OK" in _v and "NG" not in _v:
        text = "OK"
        bg_color = (0, 180, 0)       # BGR 绿
    else:
        text = f"NG  x{count}" if count > 0 else "NG"
        bg_color = (0, 0, 220)       # BGR 红

    # 计算文字尺寸
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, thickness_txt)
    pad_x, pad_y = int(badge_h * 0.4), int(badge_h * 0.25)
    bw, bh = tw + pad_x * 2, th + pad_y * 2 + baseline

    # 右上角定位
    x0 = w - bw - max(10, w // 50)
    y0 = max(10, h // 50)
    x1, y1 = x0 + bw, y0 + bh

    # 半透明背景叠加
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, -1)
    # 边框加粗突出
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), max(3, badge_h // 20))
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # 文字 (白色粗体)
    tx = x0 + pad_x
    ty = y0 + pad_y + th
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness_txt, cv2.LINE_AA)
    return img


# 缺陷框配色 (BGR): 按常见类型区分颜色, 未匹配则用红色
_DEFECT_COLORS = {
    "scratch":   (0, 165, 255),   # 橙
    "dent":      (0, 0, 255),     # 红
    "crack":     (0, 0, 200),     # 深红
    "stain":     (255, 180, 0),   # 蓝
    "burr":      (0, 200, 200),   # 黄
    "missing":   (180, 0, 255),   # 紫
    "deform":    (0, 100, 255),   # 橙红
}


def _pick_color(label: str) -> tuple:
    """按缺陷类型关键词选颜色。"""
    low = label.lower()
    for key, color in _DEFECT_COLORS.items():
        if key in low:
            return color
    return (0, 0, 255)  # 默认红


def _draw_detections(img: np.ndarray, boxes: list, labels: list,
                 scores: list) -> np.ndarray:
    """在图像上绘制检测框、编号标签和 OK/NG 大印章。"""
    import cv2

    annotated = img.copy()
    h, w = annotated.shape[:2]

    base_thickness = max(3, min(w, h) // 200)
    font_scale = max(0.6, min(w, h) / 1200)
    circle_r = max(14, min(w, h) // 45)

    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores), 1):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = _pick_color(label)

        # 检测框 (加粗)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, base_thickness)

        # 编号圆圈 (左上角, 与明细表行号对应)
        cx, cy = x1, y1
        cv2.circle(annotated, (cx, cy), circle_r, color, -1)
        cv2.circle(annotated, (cx, cy), circle_r, (255, 255, 255), 2)
        num_text = str(idx)
        (ntw, nth), _ = cv2.getTextSize(num_text, cv2.FONT_HERSHEY_SIMPLEX,
                                         font_scale * 0.9, 2)
        cv2.putText(annotated, num_text,
                    (cx - ntw // 2, cy + nth // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # 标签 (编号 + 类型 + 分数, 带背景)
        text = f"#{idx} {label} {score:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, 2)
        label_y = y1 - th - 10
        if label_y < 0:
            label_y = y2 + th + 10  # 框上方放不下则放下方
        cv2.rectangle(annotated, (x1, label_y - th - 4),
                      (x1 + tw + 8, label_y + 4), color, -1)
        cv2.putText(annotated, text, (x1 + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # 大印章: OK / NG
    verdict = "NG" if boxes else "OK"
    draw_verdict_badge(annotated, verdict, count=len(boxes))

    return annotated


# ─── 结果落库 ────────────────────────────────────────────────
def persist_qc_image(image_path: str, dest_dir: Path | str) -> str:
    """入库前把图片复制到稳定目录, 返回持久化路径。

    Gradio 上传会产生临时文件, 清理后看板图片直链 404。
    入库前复制一份到 data/qc_images/, 文件名 = 内容 sha1[:16] + 原扩展名
    (同图重复检测天然去重, 不产生冗余副本)。

    降级策略: 源文件不存在 / 已在目标目录 / 复制失败时原样返回,
    持久化失败不阻断检测落库。
    """
    src = Path(image_path)
    dest_dir = Path(dest_dir)
    if not src.is_file():
        return image_path
    try:
        if src.resolve().parent == dest_dir.resolve():
            return image_path
    except OSError:
        pass
    try:
        data = src.read_bytes()
        name = hashlib.sha1(data).hexdigest()[:16] + (src.suffix.lower() or ".png")
        dest = dest_dir / name
        if not dest.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        return str(dest)
    except OSError:
        logger.warning("[落库] 图片持久化失败, 使用原路径: %s", image_path)
        return image_path


def save_qc_result(conn: sqlite3.Connection, image_path: str,
                   verdict: str, detections: list[dict],
                   max_score: float = 0.0, prompt: str = "") -> int:
    """将检测结果写入 qc_results 表。"""
    defect_json = json.dumps(detections, ensure_ascii=False)
    cur = conn.execute(
        """INSERT INTO qc_results
           (image_path, verdict, anomaly_score, defect_json, barcode_content)
           VALUES (?, ?, ?, ?, ?)""",
        (image_path, verdict, max_score, defect_json[:5000], prompt[:200]),
    )
    conn.commit()
    return int(cur.lastrowid)
