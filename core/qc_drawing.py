"""质检结果可视化: 检测框标注/印章/热力图/尺寸过滤
(自 defect_detector.py 拆分, v1.5.0)
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("visionocr.defect")


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


# ─── 热力图叠加 ───────────────────────────────────────────────
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


# ─── OK/NG 大印章 ─────────────────────────────────────────────
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
