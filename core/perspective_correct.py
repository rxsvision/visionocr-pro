"""透视/倾斜矫正模块 - 工业 OCR 预处理核心环节

解决问题:
- 拍照角度倾斜 → 字符畸变 → OCR 置信度下降
- 产品表面与相机不平行 → 梯形畸变 → 字符宽高比失真

算法流程:
1. 自动倾斜校正 (Deskew): Hough 变换检测文本行主角度 → 旋转
2. 透视校正 (Perspective): 检测产品表面四边形轮廓 → 单应性变换 → 正面视图

设计原则:
- 保守策略: 检测不到明确几何特征时不做变换, 避免引入畸变
- 角度阈值: 倾斜 < 1.5° 不矫正 (避免过度处理)
- 四边形检测: 面积需占图像 > 20% 才视为产品表面
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# 矫正配置
DEFAULT_CORRECT_CFG = {
    "enabled": True,
    "deskew": True,           # 倾斜校正
    "perspective": True,      # 透视校正
    "min_skew_angle": 1.5,    # 最小矫正角度 (度), 低于此不矫正
    "max_skew_angle": 30.0,   # 最大可信角度, 超过视为检测失败
    "min_quad_area_ratio": 0.2,  # 四边形面积占图像比例阈值
    "border_value": 255,      # 旋转填充值 (白色背景)
}


def correct_perspective(image_path: str,
                        cfg: dict | None = None,
                        ) -> tuple[str, dict]:
    """对图像执行透视/倾斜矫正。

    Args:
        image_path: 输入图像路径
        cfg: 矫正配置 (None 则使用默认)

    Returns:
        (output_path, meta) - 矫正后图像路径 + 元数据
        如果无需矫正或检测失败, output_path == image_path
    """
    if cfg is None:
        cfg = DEFAULT_CORRECT_CFG.copy()
    else:
        merged = DEFAULT_CORRECT_CFG.copy()
        merged.update(cfg)
        cfg = merged

    if not cfg.get("enabled", True):
        return image_path, {"corrected": False, "reason": "disabled"}

    img = cv2.imread(image_path)
    if img is None:
        return image_path, {"corrected": False, "reason": "imread_failed"}

    h, w = img.shape[:2]
    meta = {"corrected": False, "original_size": f"{w}x{h}"}
    result = img.copy()

    # ─── Step 1: 透视校正 (检测产品表面四边形) ────────────────
    if cfg.get("perspective", True):
        quad = _detect_surface_quad(img, cfg)
        if quad is not None:
            warped = _warp_quad(img, quad)
            if warped is not None:
                result = warped
                meta["perspective"] = True
                meta["quad_corners"] = quad.tolist()
                meta["corrected"] = True
                h, w = result.shape[:2]
                meta["warped_size"] = f"{w}x{h}"

    # ─── Step 2: 倾斜校正 (Hough 文本行角度) ─────────────────
    if cfg.get("deskew", True):
        angle = _detect_skew_angle(result)
        if angle is not None:
            min_a = cfg.get("min_skew_angle", 1.5)
            max_a = cfg.get("max_skew_angle", 30.0)
            if abs(angle) >= min_a and abs(angle) <= max_a:
                rotated = _rotate_image(result, angle, cfg.get("border_value", 255))
                result = rotated
                meta["deskew_angle"] = round(angle, 2)
                meta["corrected"] = True
            else:
                meta["deskew_angle"] = round(angle, 2)
                meta["deskew_skipped"] = abs(angle) < min_a
        else:
            meta["deskew_angle"] = None

    # ─── 输出 ────────────────────────────────────────────────
    if not meta["corrected"]:
        meta["reason"] = "no_correction_needed"
        return image_path, meta

    uid = uuid.uuid4().hex[:8]
    out_path = str(Path(tempfile.gettempdir()) / f"visionocr_corrected_{uid}.png")
    cv2.imwrite(out_path, result)
    meta["output_size"] = f"{result.shape[1]}x{result.shape[0]}"
    return out_path, meta


def _detect_surface_quad(img: np.ndarray, cfg: dict) -> Optional[np.ndarray]:
    """检测图像中最大的四边形轮廓 (产品表面)。

    Returns:
        4x2 数组 (四个角点, 按 左上→右上→右下→左下 排序) 或 None
    """
    h, w = img.shape[:2]
    img_area = h * w
    min_area = img_area * cfg.get("min_quad_area_ratio", 0.2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 自适应阈值 + 边缘检测组合
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    # 膨胀连接断裂边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 按面积排序, 找最大的四边形
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:  # 只检查前 5 个最大轮廓
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # 多边形逼近
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return _order_points(pts)

    return None


def _order_points(pts: np.ndarray) -> np.ndarray:
    """将 4 个点排序为: 左上, 右上, 右下, 左下。"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上: x+y 最小
    rect[2] = pts[np.argmax(s)]   # 右下: x+y 最大
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # 右上: y-x 最小
    rect[3] = pts[np.argmax(d)]   # 左下: y-x 最大
    return rect


def _warp_quad(img: np.ndarray, quad: np.ndarray) -> Optional[np.ndarray]:
    """将四边形区域透视变换为矩形。"""
    tl, tr, br, bl = quad
    # 计算目标矩形尺寸
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    if max_width < 50 or max_height < 20:
        return None

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height),
                                  borderValue=(255, 255, 255))
    return warped


def _detect_skew_angle(img: np.ndarray) -> Optional[float]:
    """用 Hough 变换检测文本行主倾斜角度。

    Returns:
        角度 (度), 正值=逆时针倾斜, None=检测失败
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # 反转: 使文字为白色 (Hough 检测亮线)
    if gray.mean() > 128:
        gray = cv2.bitwise_not(gray)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=img.shape[1] // 8,
                            maxLineGap=10)
    if lines is None or len(lines) < 3:
        return None

    # 统计所有线段角度
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # 只保留接近水平的线 (文本行)
        if abs(angle) < 45:
            angles.append(angle)

    if len(angles) < 3:
        return None

    # 取中位数 (比均值更抗离群值)
    median_angle = float(np.median(angles))
    return median_angle


def _rotate_image(img: np.ndarray, angle: float,
                  border_value: int = 255) -> np.ndarray:
    """绕中心旋转图像。"""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 计算旋转后边界
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(border_value,) * 3 if img.ndim == 3 else border_value)
    return rotated
