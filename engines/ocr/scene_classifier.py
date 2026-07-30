"""场景分类器 v1 - 基于规则的图像场景分类 (无 CNN)

通过 EXIF / 分辨率 / 边缘锐度 (Laplacian 方差) / 手写启发式特征,
把输入图片分类为:
    "document"     -> 路由到 OvisOCR2 (印刷文档)
    "camera"       -> 路由到 PaddleOCR-VL (相机照片, 扭曲/倾斜)
    "handwriting"  -> 路由到 HunyuanOCR (手写体)
    "cpu_fallback" -> 路由到 RapidOCR (兜底)

规则:
    - EXIF 含手机厂商/拍摄软件 -> 倾向 camera
    - 高分辨率 + 手机 EXIF -> camera
    - Laplacian 方差低 (模糊) -> camera; 高 (锐利) -> document
    - 高频笔画 + 基线不规则 -> handwriting
    - 综合置信度 < 0.7 -> 返回 camera (最安全的兜底, 走 PaddleOCR-VL)

输出:
    {"scene": str, "confidence": float, "rules_triggered": [str, ...]}
"""
from __future__ import annotations

import logging
import os
from typing import Any

from engines.base import BaseEngine, EngineMeta, EngineState

logger = logging.getLogger("visionocr.scene_classifier")

# 常见手机/相机厂商关键字 (EXIF Make/Model/Software)
_CAMERA_MARKERS = (
    "iphone", "apple", "samsung", "huawei", "xiaomi", "oppo", "vivo",
    "honor", "redmi", "pixel", "google", "oneplus", "meizu", "realme",
    "android", "camera", "mi ", "redmi note", "iphone os",
)


class SceneClassifierEngine(BaseEngine):
    """规则版场景分类器 (v1, 无模型)"""

    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="scene_classifier",
            display_name="场景分类器",
            category="ocr",
            vram_gb=0.1,
            license="Apache-2.0",
            description="图片场景分类, 自动路由到最佳 OCR 引擎",
            tags=["分类", "路由", "轻量"],
        )

    # ─── 生命周期 ────────────────────────────────────────────
    def load(self) -> None:
        """规则版无需加载模型, 仅校验依赖"""
        self.state = EngineState.LOADING
        try:
            import cv2  # type: ignore  # noqa: F401
            import numpy as np  # type: ignore  # noqa: F401
            from PIL import Image  # type: ignore  # noqa: F401
        except ImportError as e:
            self.state = EngineState.ERROR
            logger.error(
                "依赖缺失: 请执行 "
                "`pip install opencv-python pillow numpy`。原始错误: %s", e
            )
            return
        self.state = EngineState.READY
        logger.info("就绪 (规则版 v1)")

    def unload(self) -> None:
        self._model = None
        self.state = EngineState.UNLOADED

    # ─── 推理 ────────────────────────────────────────────────
    def infer(self, image_path: str, **kwargs: Any) -> dict:
        if not self.is_ready():
            return self._result("cpu_fallback", 0.0, ["引擎未就绪"])
        if not image_path or not os.path.isfile(image_path):
            return self._result("cpu_fallback", 0.0, [f"图片不存在: {image_path}"])

        try:
            import cv2
            import numpy as np
            from PIL import Image, ExifTags
        except ImportError as e:
            return self._result("cpu_fallback", 0.0, [f"依赖缺失: {e}"])

        rules: list[str] = []
        # 各场景累加得分
        scores = {"document": 0.0, "camera": 0.0, "handwriting": 0.0}

        # ── 读图 (cv2 用于分析, PIL 用于 EXIF) ──
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return self._result("cpu_fallback", 0.0, ["无法解码图片"])
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # ── 规则 1: EXIF 相机来源 ──
        exif_text = self._read_exif(image_path, Image, ExifTags)
        is_phone = any(m in exif_text for m in _CAMERA_MARKERS)
        if is_phone:
            scores["camera"] += 2.0
            rules.append("EXIF 含手机/相机来源")

        # ── 规则 2: 分辨率 ──
        megapixels = (w * h) / 1e6
        if megapixels > 8 and is_phone:
            scores["camera"] += 1.5
            rules.append(f"高分辨率 ({megapixels:.1f}MP) + 手机 EXIF")
        elif megapixels > 8:
            scores["camera"] += 0.5
            rules.append(f"高分辨率 ({megapixels:.1f}MP)")

        # ── 规则 3: 边缘锐度 (Laplacian 方差) ──
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 100:
            scores["camera"] += 1.5
            rules.append(f"图像模糊 (Laplacian={lap_var:.0f})")
        elif lap_var > 500:
            scores["document"] += 1.5
            rules.append(f"图像锐利 (Laplacian={lap_var:.0f})")

        # ── 规则 4: 背景白度 (扫描件通常大面积白底) ──
        white_ratio = float(np.mean(gray > 200))
        if white_ratio > 0.7:
            scores["document"] += 1.0
            rules.append(f"大面积白底 ({white_ratio:.0%})")

        # ── 规则 5: 手写启发式 (高频笔画 + 基线不规则) ──
        hw_score = self._handwriting_heuristic(gray, cv2, np)
        if hw_score > 0.5:
            scores["handwriting"] += 2.0 * hw_score
            rules.append(f"手写特征 (score={hw_score:.2f})")

        # ── 决策 ──
        scene, confidence = self._decide(scores, rules)
        return self._result(scene, confidence, rules)

    # ─── 内部工具 ────────────────────────────────────────────
    @staticmethod
    def _read_exif(image_path: str, Image: Any, ExifTags: Any) -> str:
        """提取 EXIF 文本 (Make/Model/Software), 失败返回空串"""
        try:
            with Image.open(image_path) as im:
                exif = im.getexif()
                if not exif:
                    return ""
                parts = []
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if tag in ("Make", "Model", "Software", "HostComputer"):
                        parts.append(str(value).lower())
                return " ".join(parts)
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _handwriting_heuristic(gray: Any, cv2: Any, np: Any) -> float:
        """手写体启发式打分 (0~1)

        思路:
          - 二值化后做边缘, 统计边缘像素密度 (手写笔画细碎, 密度中等偏高)
          - 按行投影, 计算每行重心的波动 (手写基线不规则, 波动大)
        """
        try:
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            # Otsu 二值化
            _, binary = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            ink_ratio = float(np.mean(binary > 0))
            # 墨迹过少 (空白) 或过多 (纯图) 都不像手写
            if ink_ratio < 0.01 or ink_ratio > 0.4:
                return 0.0

            # 行投影: 每行的墨迹重心 x 坐标
            row_mass = binary.sum(axis=1).astype(np.float64)
            valid = row_mass > row_mass.max() * 0.05
            if valid.sum() < 10:
                return 0.0
            # 用每行墨迹列加权的重心衡量基线漂移
            cols = np.arange(binary.shape[1])
            centroids = []
            for r in np.where(valid)[0]:
                row = binary[r].astype(np.float64)
                if row.sum() > 0:
                    centroids.append((row * cols).sum() / row.sum())
            if len(centroids) < 5:
                return 0.0
            centroids = np.array(centroids)
            # 基线不规则度: 重心标准差归一化
            irregularity = float(centroids.std() / (binary.shape[1] + 1e-6))

            # 笔画细碎度: 连通域数量 / 面积
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            num_labels = max(num_labels - 1, 1)  # 去掉背景
            fragility = min(num_labels / (ink_ratio * binary.size + 1e-6), 1.0)

            score = 0.6 * min(irregularity * 8, 1.0) + 0.4 * fragility
            return float(min(max(score, 0.0), 1.0))
        except Exception:  # noqa: BLE001
            return 0.0

    @staticmethod
    def _decide(scores: dict, rules: list) -> tuple[str, float]:
        """根据累加得分决策, 置信度 < 0.7 时回退到 camera"""
        total = sum(scores.values())
        if total <= 0:
            # 无任何规则触发 -> 最安全兜底
            rules.append("无规则触发, 使用安全兜底")
            return "camera", 0.5

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = scores[best] / total

        # 置信度不足 -> camera (走 PaddleOCR-VL, 对扭曲/手写都较鲁棒)
        if confidence < 0.7:
            rules.append(f"置信度 {confidence:.2f} < 0.7, 回退 camera")
            return "camera", round(confidence, 4)

        return best, round(confidence, 4)

    @staticmethod
    def _result(scene: str, confidence: float, rules: list) -> dict:
        return {
            "scene": scene,
            "confidence": round(float(confidence), 4),
            "rules_triggered": rules,
            "engine": "scene_classifier",
        }
