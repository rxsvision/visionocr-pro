"""3D 深度融合缺陷检测 (Phase 4C)
====================================

将 Sizector 结构光相机的 **深度图 (mm)** 与 **2D 彩色检测** 融合,
实现"几何形变 + 表面纹理"双通道缺陷判定, 降低单一通道的误报/漏报。

检测原理:
    1. 稳健基准面拟合: 用深度图的中值/分位数平面作为"合格基准",
       避免缺陷点本身污染基准 (相比最小二乘更鲁棒)。
    2. 高度偏差图: deviation = depth - baseline, 单位 mm。
    3. 异常掩膜: |deviation| > depth_threshold_mm 且深度有效的像素。
    4. 连通域分析: 过滤小于 min_area_px 的噪点, 输出每个异常区域
       (位置 / 面积 / 峰值偏差 / 凸起或凹坑)。
    5. 融合判定: 深度异常 OR 2D 缺陷 -> NG (高召回); 同时标注
       两通道空间重合的区域 (高置信度缺陷)。

对外暴露:
    - detect_depth_anomaly(frame, ...)   仅深度通道检测
    - fuse_detection(frame, result_2d, ...)  深度 + 2D 融合
    - annotate_depth(frame, anomaly)     生成深度伪彩 + RGB 叠加标注图
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 稳健基准面拟合
# =============================================================================
def _fit_baseline(depth_mm: np.ndarray, method: str = "median") -> np.ndarray:
    """估计"合格表面"基准深度, 返回与 depth_mm 同形的基准面。

    Args:
        depth_mm: (H, W) float32, 无效点为 nan。
        method:   "median" 全局中值 (最快) |
                  "percentile" 低分位平面 (偏向最低表面, 适合检测凸起) |
                  "plane" 稳健倾斜平面拟合 (适合表面本身有斜率的场景)。

    设计取舍:
        工业件表面常带缓慢倾斜, 纯全局中值会把倾斜误判为异常;
        "plane" 用迭代中值拟合一次平面, 兼顾倾斜与鲁棒性 (默认推荐)。
    """
    d = depth_mm.astype(np.float32)
    valid = np.isfinite(d)
    if not valid.any():
        return np.full_like(d, np.nan)

    if method == "median":
        return np.full_like(d, float(np.nanmedian(d)))

    if method == "percentile":
        # 用 30 分位作为基准 (偏向较低表面, 凸起更易被检出)
        return np.full_like(d, float(np.nanpercentile(d, 30)))

    # method == "plane": 稳健一次平面拟合 z = a*x + b*y + c
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xs, ys, zs = xx[valid], yy[valid], d[valid]

    # 两轮迭代: 第一轮普通最小二乘, 剔除残差 >2*MAD 的点后重拟合
    A = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    resid = zs - A @ coef
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
    inlier = np.abs(resid - np.median(resid)) < 2.5 * 1.4826 * mad
    if inlier.sum() >= 3:
        coef, *_ = np.linalg.lstsq(A[inlier], zs[inlier], rcond=None)

    baseline = coef[0] * xx + coef[1] * yy + coef[2]
    return baseline.astype(np.float32)


# =============================================================================
# 深度异常检测
# =============================================================================
def detect_depth_anomaly(
    frame,
    depth_threshold_mm: float = 0.5,
    min_area_px: int = 30,
    baseline_method: str = "plane",
) -> dict:
    """对深度帧做几何形变检测, 返回异常区域列表。

    Args:
        frame:               DepthFrame (含 depth_mm)。
        depth_threshold_mm:  高度偏差阈值 (mm), 超过即视为异常。
                             建议按相机 Z 精度与被检件公差设定 (如 0.3~1.0)。
        min_area_px:         最小连通域面积 (像素), 滤除噪点。
        baseline_method:     基准面拟合方法 (median/percentile/plane)。

    Returns:
        dict: {
            "deviation_mm": (H,W) float32 偏差图,
            "mask":         (H,W) bool 异常掩膜,
            "regions":      [ {bbox, area_px, peak_mm, polarity, centroid} ],
            "n_regions":    int,
            "max_abs_dev":  float,
            "baseline_method": str,
        }
    """
    import cv2

    depth = frame.depth_mm
    valid = np.isfinite(depth)

    if not valid.any():
        return {"deviation_mm": np.zeros_like(depth), "mask": np.zeros_like(depth, bool),
                "regions": [], "n_regions": 0, "max_abs_dev": 0.0,
                "baseline_method": baseline_method, "error": "深度图无有效点"}

    baseline = _fit_baseline(depth, method=baseline_method)
    deviation = depth - baseline
    deviation[~valid] = 0.0

    mask = valid & (np.abs(deviation) > depth_threshold_mm)

    # 连通域分析 (8-邻域)
    regions = []
    mask_u8 = mask.astype(np.uint8) * 255
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8)

    for lab in range(1, n_labels):  # 0 是背景
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])

        region_mask = labels == lab
        region_dev = deviation[region_mask]
        peak = float(region_dev[np.argmax(np.abs(region_dev))])
        polarity = "凸起" if peak > 0 else "凹坑"

        regions.append({
            "bbox": (x, y, x + bw, y + bh),
            "area_px": area,
            "peak_mm": peak,
            "abs_peak_mm": abs(peak),
            "polarity": polarity,
            "centroid": (float(centroids[lab, 0]), float(centroids[lab, 1])),
        })

    # 按峰值偏差降序排列
    regions.sort(key=lambda r: r["abs_peak_mm"], reverse=True)

    return {
        "deviation_mm": deviation,
        "mask": mask,
        "regions": regions,
        "n_regions": len(regions),
        "max_abs_dev": float(np.abs(deviation[valid]).max()) if valid.any() else 0.0,
        "baseline_method": baseline_method,
    }


# =============================================================================
# 深度 + 2D 融合判定
# =============================================================================
def _bbox_overlap(box_a, box_b) -> float:
    """计算两个 bbox (x1,y1,x2,y2) 的 IoU。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def fuse_detection(
    frame,
    result_2d: dict | None = None,
    depth_threshold_mm: float = 0.5,
    min_area_px: int = 30,
    baseline_method: str = "plane",
    fusion_mode: str = "or",
) -> dict:
    """融合深度异常与 2D 检测结果, 输出统一判定。

    Args:
        frame:          DepthFrame。
        result_2d:      2D 检测结果 (来自 run_detection), 含 detections/verdict;
                        为 None 时仅用深度通道。
        fusion_mode:    "or"  深度或2D任一报NG即NG (高召回, 默认) |
                        "and" 两通道都报NG才NG (高精确, 易漏报) |
                        "depth_only" 仅深度通道。

    Returns:
        dict: {
            "verdict": "OK"/"NG",
            "depth_anomaly": {...},       # 深度检测结果
            "detections_2d": [...],       # 2D 检测明细
            "fused_defects": [...],       # 融合后的缺陷列表
            "count": int,
            "reason": str,
        }
    """
    depth_anomaly = detect_depth_anomaly(
        frame,
        depth_threshold_mm=depth_threshold_mm,
        min_area_px=min_area_px,
        baseline_method=baseline_method,
    )

    depth_regions = depth_anomaly["regions"]
    detections_2d = []
    if result_2d and not result_2d.get("error"):
        detections_2d = result_2d.get("detections", [])

    # 标注每个深度区域是否与 2D 检测空间重合 (重合 = 高置信度)
    for region in depth_regions:
        region["confirmed_by_2d"] = False
        for det in detections_2d:
            if _bbox_overlap(region["bbox"], det.get("box", (0, 0, 0, 0))) > 0.1:
                region["confirmed_by_2d"] = True
                region["matched_2d_label"] = det.get("label", "")
                break

    depth_ng = len(depth_regions) > 0
    two_d_ng = len(detections_2d) > 0

    # 融合判定
    if fusion_mode == "depth_only":
        verdict_ng = depth_ng
        reason = f"深度通道: {len(depth_regions)} 处几何异常"
    elif fusion_mode == "and":
        verdict_ng = depth_ng and two_d_ng
        reason = f"深度 {len(depth_regions)} 处 ∧ 2D {len(detections_2d)} 处"
    else:  # "or"
        verdict_ng = depth_ng or two_d_ng
        reason = f"深度 {len(depth_regions)} 处 ∨ 2D {len(detections_2d)} 处"

    # 汇总融合缺陷列表 (深度区域 + 未与深度重合的 2D 检测)
    fused_defects = []
    for region in depth_regions:
        fused_defects.append({
            "source": "3D+2D" if region["confirmed_by_2d"] else "3D",
            "type": f"{region['polarity']} ({region['abs_peak_mm']:.2f}mm)",
            "bbox": region["bbox"],
            "confidence": 0.95 if region["confirmed_by_2d"] else 0.7,
            "peak_mm": region["peak_mm"],
        })

    # 仅 2D 检出但深度无异常的区域 (表面纹理缺陷, 如划痕/色差)
    matched_2d_boxes = [d.get("box") for d in detections_2d
                        if any(_bbox_overlap(r["bbox"], d.get("box", (0, 0, 0, 0))) > 0.1
                               for r in depth_regions)]
    for det in detections_2d:
        box = det.get("box", (0, 0, 0, 0))
        if box not in matched_2d_boxes:
            fused_defects.append({
                "source": "2D",
                "type": det.get("label", "缺陷"),
                "bbox": box,
                "confidence": det.get("score", 0.5),
                "peak_mm": 0.0,
            })

    return {
        "verdict": "NG" if verdict_ng else "OK",
        "depth_anomaly": depth_anomaly,
        "detections_2d": detections_2d,
        "fused_defects": fused_defects,
        "count": len(fused_defects),
        "reason": reason,
        "fusion_mode": fusion_mode,
    }


# =============================================================================
# 可视化标注
# =============================================================================
def annotate_depth(frame, fused: dict, alpha: float = 0.5) -> np.ndarray:
    """生成融合标注图: RGB 底图 + 深度异常伪彩叠加 + 缺陷框。

    Args:
        frame:  DepthFrame。
        fused:  fuse_detection 的返回值。
        alpha:  深度异常区域叠加透明度。

    Returns:
        (H, W, 3) uint8 BGR 标注图。
    """
    import cv2

    h, w = frame.height, frame.width

    # 底图: 优先 RGB, 否则灰度转 BGR, 再否则深度伪彩
    if frame.rgb is not None:
        canvas = frame.rgb.copy()
    elif frame.gray is not None:
        canvas = cv2.cvtColor(frame.gray, cv2.COLOR_GRAY2BGR)
    else:
        canvas = frame.depth_colormap()

    # 深度异常掩膜叠加 (红色调, 区分凸起/凹坑)
    anomaly = fused.get("depth_anomaly", {})
    deviation = anomaly.get("deviation_mm")
    mask = anomaly.get("mask")
    if deviation is not None and mask is not None and mask.any():
        overlay = canvas.copy()
        # 凸起偏红, 凹坑偏蓝
        bump = mask & (deviation > 0)
        pit = mask & (deviation < 0)
        overlay[bump] = [0, 0, 220]     # BGR 红
        overlay[pit] = [220, 120, 0]    # BGR 蓝
        canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    # 绘制缺陷框 + 标签
    for defect in fused.get("fused_defects", []):
        x1, y1, x2, y2 = [int(v) for v in defect["bbox"]]
        src = defect["source"]
        if src == "3D+2D":
            color = (0, 0, 255)      # 红: 双通道确认 (最高置信)
        elif src == "3D":
            color = (0, 140, 255)    # 橙: 仅深度
        else:
            color = (0, 220, 220)    # 黄: 仅2D
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"[{src}] {defect['type']}"
        # 文字背景框提升可读性
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas
