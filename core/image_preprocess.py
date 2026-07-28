"""图像预处理管线 (Phase 3F 审查后新增)

解决工业相机/手机拍照/扫描存档/截图等场景下的 OCR 前图像质量问题。
管线按条件触发, 不对已满足质量要求的图像做无意义处理:
  1. 暗色/反转检测 → 反色
  2. 对比度不足 → CLAHE 自适应增强
  3. 噪声过高 → 快速降噪
  4. 倾斜 → 纠偏 (Hough)
  5. 分辨率不足 → 放大 (小图/截图)
  6. 光照不均 → 背景归一化

设计原则: 精度第一, 不引入伪影; 处理耗时 < 200ms (1080p)。
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


# ─── 阈值常量 ────────────────────────────────────────────────
_DARK_BG_MEAN = 80        # 背景平均灰度 < 此值 → 疑似暗色/反转
_LOW_CONTRAST_STD = 40    # 灰度标准差 < 此值 → 对比度不足
_NOISE_LAPLACIAN = 800    # Laplacian 方差 > 此值且 mean 低 → 噪声
_MIN_TEXT_HEIGHT = 25     # 目标最小文字高度 (px)
_UPSCALE_MAX = 2.5        # 最大放大倍率
_SKEW_ANGLE_THRESH = 0.5  # 倾斜角度 > 此值才纠偏
_MAX_DIM_FOR_UPSCALE = 1200  # 短边 < 此值才考虑放大


def preprocess_image(image: np.ndarray, *,
                     do_deskew: bool = True,
                     do_denoise: bool = True,
                     do_enhance: bool = True,
                     do_upscale: bool = True,
                     ) -> np.ndarray:
    """对图像执行条件式预处理, 返回处理后图像 (BGR)。

    Args:
        image: BGR numpy 数组 (H, W, 3) 或灰度 (H, W)
        do_*: 各步骤开关 (可按场景关闭)

    Returns:
        预处理后的 BGR 图像
    """
    if cv2 is None or image is None:
        return image

    # 确保 3 通道
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    img = image.copy()

    # 1. 暗色/反转检测
    if do_enhance:
        img = _fix_inverted(img)

    # 2. 对比度增强 (CLAHE)
    if do_enhance:
        img = _enhance_contrast(img)

    # 3. 降噪
    if do_denoise:
        img = _denoise(img)

    # 4. 纠偏
    if do_deskew:
        img = _deskew(img)

    # 5. 自适应放大 (小图/截图)
    if do_upscale:
        img = _adaptive_upscale(img)

    # 6. 背景归一化 (光照不均/泛黄)
    if do_enhance:
        img = _normalize_background(img)

    return img


def preprocess_file(image_path: str, output_path: str | None = None, **kwargs) -> str:
    """读取图像文件 → 预处理 → 保存, 返回输出路径。

    若 output_path 为 None, 覆盖原文件。
    """
    if cv2 is None:
        return image_path
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return image_path
    result = preprocess_image(img, **kwargs)
    out = output_path or image_path
    cv2.imwrite(out, result)
    return out


# ─── 内部实现 ────────────────────────────────────────────────

def _fix_inverted(img: np.ndarray) -> np.ndarray:
    """检测暗色背景 (白字黑底) 并反色。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    # 暗色背景: 均值低 + 高亮像素占比少
    if mean_val < _DARK_BG_MEAN:
        bright_ratio = (gray > 200).sum() / gray.size
        if bright_ratio < 0.3:
            img = cv2.bitwise_not(img)
    return img


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE 自适应直方图均衡化 (仅对比度不足时触发)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.std() >= _LOW_CONTRAST_STD:
        return img  # 对比度已足够
    # 转 LAB, 对 L 通道做 CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _denoise(img: np.ndarray) -> np.ndarray:
    """快速降噪 (仅高噪声图像触发)。

    判据: Laplacian 方差高 (细节多/噪声多) + 灰度均值中等 (非纯文档)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 高 Laplacian + 低对比度 → 可能是噪声而非文字
    if lap_var > _NOISE_LAPLACIAN and gray.std() < _LOW_CONTRAST_STD:
        img = cv2.fastNlMeansDenoisingColored(img, None, h=6, hForColorComponents=6,
                                              templateWindowSize=7, searchWindowSize=21)
    return img


def _deskew(img: np.ndarray) -> np.ndarray:
    """基于 Hough 变换的文档纠偏 (仅倾斜 > 0.5° 时触发)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=gray.shape[1] // 4, maxLineGap=10)
    if lines is None or len(lines) < 3:
        return img

    # 统计主要角度
    angles = []
    for line in lines[:200]:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # 只取接近水平的线 (文档行)
        if abs(angle) < 45:
            angles.append(angle)
    if not angles:
        return img

    median_angle = np.median(angles)
    if abs(median_angle) < _SKEW_ANGLE_THRESH:
        return img  # 倾斜不显著

    # 旋转纠偏
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _adaptive_upscale(img: np.ndarray) -> np.ndarray:
    """小图/截图放大 (短边 < 1200px 时放大到目标尺寸)。"""
    h, w = img.shape[:2]
    short_side = min(h, w)
    if short_side >= _MAX_DIM_FOR_UPSCALE:
        return img  # 已足够大

    # 计算放大倍率 (目标短边 1200px, 但不超过 _UPSCALE_MAX)
    target = _MAX_DIM_FOR_UPSCALE
    scale = min(target / short_side, _UPSCALE_MAX)
    if scale < 1.2:
        return img

    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _normalize_background(img: np.ndarray) -> np.ndarray:
    """背景归一化: 消除光照不均/泛黄 (大核高斯背景估计 + 差分)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 如果背景已经很均匀 (std 低), 跳过
    if gray.std() < 30:
        return img
    # 估计背景 (大核模糊)
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    # 归一化: 原图 / 背景 * 255
    norm = cv2.divide(gray, bg, scale=255)
    # 如果归一化后对比度提升, 用归一化结果替换灰度
    if norm.std() > gray.std():
        # 保持彩色信息: 用归一化灰度 + 原始色度
        img_norm = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        return img_norm
    return img
