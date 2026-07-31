"""图像预处理管线 - 提升 OCR 识别精度

针对工业场景 (零件标记、铭牌、激光刻字) 的预处理策略:
1. CLAHE 对比度增强 (解决低对比度/光照不均)
2. 超分辨率放大 (小字符 -> 放大到引擎最佳识别尺寸)
3. 自适应二值化 (解决背景干扰)
4. 去噪 (高斯/中值, 解决颗粒噪声)

设计原则:
- 不改变原始图像, 输出预处理后的临时文件
- 可通过 config 开关控制各步骤
- 对已经是高质量文档图的输入, 预处理应无副作用或轻微正增益
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.imutils import imread_unicode

# 输入尺寸保护: 超过此尺寸的图像先缩小再处理 (防 OOM)
MAX_INPUT_SIZE = 4096


# 预处理配置默认值
DEFAULT_PREPROCESS_CFG = {
    "enabled": True,
    "clahe": True,
    "clahe_clip": 2.0,
    "clahe_grid": 8,
    "upscale": True,
    "upscale_min_height": 800,
    "upscale_factor": 2.0,
    "upscale_max_height": 2400,
    "denoise": True,
    "denoise_strength": 3,
    "binarize": False,
    "binarize_block": 11,
    "binarize_c": 2,
    "sharpen": True,
    "sharpen_amount": 0.5,
}


def preprocess_for_ocr(image_path: str,
                       cfg: Optional[dict] = None,
                       ) -> tuple[str, dict]:
    """对输入图像执行 OCR 预处理, 返回 (处理后路径, 元信息)。"""
    p = {**DEFAULT_PREPROCESS_CFG, **(cfg or {})}
    meta = {"original": image_path, "steps": [], "size_before": None, "size_after": None}

    if not p.get("enabled", True):
        return image_path, meta

    img = imread_unicode(image_path, flags=cv2.IMREAD_UNCHANGED)
    if img is None:
        return image_path, meta

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 输入尺寸保护: 超大图先缩小 (防 OOM)
    h0, w0 = img.shape[:2]
    if max(h0, w0) > MAX_INPUT_SIZE:
        scale = MAX_INPUT_SIZE / max(h0, w0)
        img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)),
                         interpolation=cv2.INTER_AREA)
        meta["steps"].append(f"downscale_{max(h0,w0)}→{MAX_INPUT_SIZE}")

    h, w = img.shape[:2]
    meta["size_before"] = f"{w}x{h}"

    # 1. 去噪
    if p.get("denoise"):
        k = int(p.get("denoise_strength", 3))
        if k % 2 == 0:
            k += 1
        if img.ndim == 2:
            img = cv2.medianBlur(img, k)
        else:
            img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
        meta["steps"].append("denoise")

    # 2. CLAHE 对比度增强
    if p.get("clahe"):
        clip = float(p.get("clahe_clip", 2.0))
        grid = int(p.get("clahe_grid", 4))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        if img.ndim == 2:
            img = clahe.apply(img)
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        meta["steps"].append("clahe")

    # 3. 锐化 (Unsharp Mask)
    if p.get("sharpen"):
        amount = float(p.get("sharpen_amount", 0.5))
        blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        meta["steps"].append("sharpen")

    # 4. 小图放大
    if p.get("upscale"):
        min_h = int(p.get("upscale_min_height", 800))
        max_h = int(p.get("upscale_max_height", 2400))
        factor = float(p.get("upscale_factor", 2.0))
        cur_h, cur_w = img.shape[:2]
        if cur_h < min_h:
            actual_factor = min(factor, max_h / cur_h)
            new_w = int(cur_w * actual_factor)
            new_h = int(cur_h * actual_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            meta["steps"].append(f"upscale_x{actual_factor:.1f}")

    # 5. 自适应二值化 (可选, 默认关)
    if p.get("binarize"):
        block = int(p.get("binarize_block", 11))
        c = int(p.get("binarize_c", 2))
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, c
        )
        meta["steps"].append("binarize")

    # 输出
    h2, w2 = img.shape[:2]
    meta["size_after"] = f"{w2}x{h2}"

    if not meta["steps"]:
        return image_path, meta

    suffix = Path(image_path).suffix.lower()
    out_suffix = suffix if suffix in ('.png', '.bmp', '.tiff') else '.png'
    uid = uuid.uuid4().hex[:8]
    tmp = Path(tempfile.gettempdir()) / f"visionocr_pp_{uid}{out_suffix}"
    cv2.imwrite(str(tmp), img)
    meta["output"] = str(tmp)
    return str(tmp), meta


def check_image_quality(image_path: str) -> dict:
    """图像质量预检: 模糊/过曝/全黑检测。

    Returns:
        {"blur": bool, "blur_score": float, "exposure": str,
         "ok": bool, "warning": str}
    """
    img = imread_unicode(image_path)
    if img is None:
        return {"blur": False, "blur_score": 0, "exposure": "unreadable",
                "ok": False, "warning": "无法读取图像文件"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # 模糊检测: Laplacian 方差
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blur = lap_var < 100

    # 曝光检测: 均值
    mean_val = gray.mean()
    if mean_val < 20:
        exposure = "underexposed"
    elif mean_val > 240:
        exposure = "overexposed"
    else:
        exposure = "normal"

    # 综合判定
    warnings = []
    if is_blur:
        warnings.append(f"图像模糊 (清晰度 {lap_var:.0f} < 100), 建议重拍")
    if exposure == "underexposed":
        warnings.append(f"图像过暗 (均值 {mean_val:.0f}), 建议增加光照")
    elif exposure == "overexposed":
        warnings.append(f"图像过曝 (均值 {mean_val:.0f}), 建议降低曝光")

    return {
        "blur": is_blur,
        "blur_score": round(lap_var, 1),
        "exposure": exposure,
        "ok": len(warnings) == 0,
        "warning": "; ".join(warnings),
    }
