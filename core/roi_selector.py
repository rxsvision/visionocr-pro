"""智能 ROI 选择 — 异常热力图 / 检测框 → 缺陷候选区域 (Phase 3)

用途: VLM 整图解释在高分辨率工业图上既慢又易失焦。本模块把
异常证据 (PatchCore/DINOv2 融合热力图 + DINO/YOLO 检测框) 收敛为
少量候选区域, 供 VLM 局部放大解释。

设计取舍 (Simple is best):
- 纯 numpy + cv2, 不引入新依赖。
- 热力图路径: 相对阈值 (相对自身峰值) → 连通域 → bbox → 过滤/合并/
  top-k。不依赖分数量纲, 跨引擎通用。
- 检测框路径: 直接转 ROI (检测器已给出定位, 无需再分析)。
- 输出为原图坐标系 {x, y, w, h, score, source}, 裁切由 crop_rois 负责。
"""
from __future__ import annotations

import numpy as np

__all__ = ["select_rois", "crop_rois"]


def _rois_from_map(anomaly_map: np.ndarray, rel_thresh: float,
                   min_area: float) -> list[dict]:
    """热力图 → 候选 ROI (连通域分析)。"""
    import cv2

    m = np.asarray(anomaly_map, dtype=np.float32)
    if m.size == 0:
        return []
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-9:  # 平坦图: 无定位信息
        return []
    norm = (m - lo) / (hi - lo)
    mask = (norm >= rel_thresh).astype(np.uint8)
    if not mask.any():
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = m.shape[:2]
    rois = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        seg = mask[y:y + bh, x:x + bw] > 0
        score = float(norm[y:y + bh, x:x + bw][seg].mean())
        rois.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                     "score": score, "source": "heatmap"})
    return rois


def _rois_from_boxes(boxes: list, scores: list | None) -> list[dict]:
    """检测框 (x1,y1,x2,y2) → ROI。"""
    rois = []
    n = len(boxes)
    scores = list(scores) if scores else [1.0] * n
    for box, sc in zip(boxes, scores):
        try:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        rois.append({"x": int(round(x1)), "y": int(round(y1)),
                     "w": int(round(x2 - x1)), "h": int(round(y2 - y1)),
                     "score": float(sc), "source": "det"})
    return rois


def _pad_and_clip(roi: dict, img_h: int, img_w: int,
                  pad_frac: float) -> dict:
    pw = int(round(roi["w"] * pad_frac))
    ph = int(round(roi["h"] * pad_frac))
    x1 = max(0, roi["x"] - pw)
    y1 = max(0, roi["y"] - ph)
    x2 = min(img_w, roi["x"] + roi["w"] + pw)
    y2 = min(img_h, roi["y"] + roi["h"] + ph)
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
            "score": roi["score"], "source": roi["source"]}


def _contained_frac(a: dict, b: dict) -> float:
    """a 被 b 覆盖的面积比例 (用于重叠归并)。"""
    ix1 = max(a["x"], b["x"])
    iy1 = max(a["y"], b["y"])
    ix2 = min(a["x"] + a["w"], b["x"] + b["w"])
    iy2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, a["w"] * a["h"])
    return inter / area_a


def select_rois(image_shape, anomaly_map: np.ndarray | None = None,
                boxes: list | None = None,
                box_scores: list | None = None,
                max_rois: int = 3, min_area_frac: float = 0.0005,
                pad_frac: float = 0.25, rel_thresh: float = 0.45,
                merge_frac: float = 0.6) -> list[dict]:
    """融合热力图与检测框, 输出 top-k 候选 ROI。

    Args:
        image_shape:  原图 (H, W[, C])。
        anomaly_map:  融合异常热力图 (任意尺度, 内部归一化), 可为 None。
        boxes:        检测框列表 [(x1,y1,x2,y2), ...], 可为 None。
        box_scores:   各框置信度 (缺省按 1.0)。
        max_rois:     最多返回 ROI 数。
        min_area_frac: 连通域最小面积占全图比例 (滤噪点)。
        pad_frac:     ROI 外扩比例 (给 VLM 留上下文)。
        rel_thresh:   热力图相对阈值 (相对峰值)。
        merge_frac:   一个 ROI 被另一个覆盖超过该比例则归并 (保留高分者)。

    Returns:
        [{x, y, w, h, score, source}], 按 score 降序, 长度 <= max_rois。
    """
    if image_shape is None or len(image_shape) < 2:
        return []
    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    if img_h < 8 or img_w < 8:
        return []

    rois = []
    if anomaly_map is not None:
        rois += _rois_from_map(anomaly_map, rel_thresh,
                               min_area_frac * img_h * img_w)
    if boxes:
        rois += _rois_from_boxes(boxes, box_scores)
    if not rois:
        return []

    rois = [_pad_and_clip(r, img_h, img_w, pad_frac) for r in rois]
    rois = [r for r in rois if r["w"] >= 4 and r["h"] >= 4]
    rois.sort(key=lambda r: -r["score"])

    # 重叠归并: 低分 ROI 被高分 ROI 覆盖超过 merge_frac → 丢弃
    kept: list[dict] = []
    for r in rois:
        if any(_contained_frac(r, k) >= merge_frac for k in kept):
            continue
        kept.append(r)
        if len(kept) >= max_rois:
            break
    return kept


def crop_rois(image: np.ndarray, rois: list[dict],
              min_side: int = 256, max_side: int = 1024
              ) -> list[tuple[dict, np.ndarray]]:
    """按 ROI 裁切图像; 过小放大便于 VLM 识读, 过大缩小控制传输量。

    缩放的语义: 短边 < min_side 时放大至 min_side, 但放大倍率上限 4
    (纯插值无法创造细节, 无限放大只增加传输量); 长边 > max_side 时
    等比缩小。

    Returns:
        [(roi, crop_bgr), ...] 与输入 rois 同序 (无效 ROI 跳过)。
    """
    import cv2

    out = []
    if image is None:
        return out
    img_h, img_w = image.shape[:2]
    for r in rois or []:
        x1 = max(0, min(r["x"], img_w - 1))
        y1 = max(0, min(r["y"], img_h - 1))
        x2 = max(x1 + 1, min(r["x"] + r["w"], img_w))
        y2 = max(y1 + 1, min(r["y"] + r["h"], img_h))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        long_side = max(ch, cw)
        short_side = min(ch, cw)
        if long_side > max_side:
            f = max_side / long_side
            crop = cv2.resize(crop, (max(1, int(cw * f)),
                                     max(1, int(ch * f))),
                              interpolation=cv2.INTER_AREA)
        elif short_side < min_side:
            f = min(4.0, min_side / max(1, short_side))
            crop = cv2.resize(crop, (min(2048, int(cw * f)),
                                     min(2048, int(ch * f))),
                              interpolation=cv2.INTER_CUBIC)
        out.append((r, crop))
    return out
