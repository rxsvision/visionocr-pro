"""条码识别引擎 — pyzbar (ZBar) 后端

支持: EAN-13/8, UPC-A/E, Code 128/39/93, QR, DataMatrix, PDF417, Aztec 等。
无 GPU 依赖, 单张推理 < 10ms。LGPL 动态链接, 商业友好。

产品设计:
- 工业场景条码是刚需 (工件追溯、批次号、产线路由)
- 与 OCR 互补: OCR 读文字, Barcode 读编码
- 支持多图预处理策略: 原图 → 灰度 → 自适应阈值 (提升低对比度场景识别率)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from engines.base import BaseEngine, EngineMeta, EngineState

logger = logging.getLogger("visionocr.barcode")


class BarcodeEngine(BaseEngine):
    """ZBar 条码/二维码识别 (pyzbar)。"""

    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="barcode",
            display_name="条码识别 (ZBar)",
            category="vision",
            vram_gb=0.0,
            license="LGPL-2.1",
            description="一维码/二维码识别, 支持 EAN/UPC/Code128/QR/DataMatrix 等, <10ms/张",
            tags=["条码", "二维码", "QR", "DataMatrix", "轻量", "追溯"],
        )

    def load(self) -> None:
        """验证 pyzbar 可用性。"""
        try:
            from pyzbar import pyzbar  # noqa: F401
            self._pyzbar = pyzbar
            self.state = EngineState.READY
            logger.info("条码引擎就绪 (pyzbar/ZBar)")
        except ImportError as e:
            logger.error("pyzbar 不可用: %s (pip install pyzbar)", e)
            self.state = EngineState.ERROR
        except Exception as e:
            logger.error("ZBar 初始化失败: %s", e)
            self.state = EngineState.ERROR

    def infer(self, image: Any, **kwargs) -> dict:
        """识别图像中所有条码/二维码。

        Args:
            image: 图像路径 (str/Path) 或 numpy array (BGR/RGB)

        Returns:
            {
                "codes": [{"content": str, "type": str, "rect": [x,y,w,h],
                           "polygon": [[x,y],...]}],
                "count": int,
                "raw_count": int,  # 预处理前原图识别数
            }
        """
        if not self.is_ready():
            return {"codes": [], "count": 0, "error": "引擎未就绪"}

        img = self._load_image(image)
        if img is None:
            return {"codes": [], "count": 0, "error": "无法读取图像"}

        # 策略1: 原图直识 (最快, 适合高对比度工业场景)
        codes = self._decode(img)
        raw_count = len(codes)

        # 策略2: 若原图无结果, 尝试灰度 + 自适应阈值 (低对比度/反光场景)
        if not codes:
            codes = self._decode_enhanced(img)

        return {
            "codes": codes,
            "count": len(codes),
            "raw_count": raw_count,
        }

    def unload(self) -> None:
        self._pyzbar = None
        self.state = EngineState.UNLOADED

    # ─── 内部实现 ────────────────────────────────────────────

    def _load_image(self, image: Any) -> np.ndarray | None:
        """统一输入为 numpy BGR array。"""
        if isinstance(image, (str, Path)):
            from core.imutils import imread_unicode
            img = imread_unicode(str(image))
            return img
        elif isinstance(image, np.ndarray):
            return image
        elif hasattr(image, "convert"):  # PIL
            return np.array(image.convert("RGB"))[:, :, ::-1]
        return None

    def _decode(self, img: np.ndarray) -> list[dict]:
        """核心解码: numpy BGR → pyzbar decode。"""
        from pyzbar import pyzbar

        # pyzbar 接受灰度或 RGB, 不接受 BGR → 转换
        if img.ndim == 3 and img.shape[2] == 3:
            rgb = img[:, :, ::-1]
        else:
            rgb = img

        try:
            results = pyzbar.decode(rgb)
        except Exception as e:
            logger.warning("pyzbar decode 异常: %s", e)
            return []

        codes = []
        for r in results:
            content = r.data.decode("utf-8", errors="replace")
            codes.append({
                "content": content,
                "type": r.type,
                "rect": [r.rect.left, r.rect.top, r.rect.width, r.rect.height],
                "polygon": [[p.x, p.y] for p in r.polygon] if r.polygon else [],
            })
        return codes

    def _decode_enhanced(self, img: np.ndarray) -> list[dict]:
        """增强解码: 灰度 → 自适应阈值 → 解码 (低对比度/反光场景)。"""
        import cv2

        # 灰度
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 尝试多种预处理
        strategies = [
            ("adaptive_thresh", lambda g: cv2.adaptiveThreshold(
                g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 51, 10)),
            ("otsu", lambda g: cv2.threshold(
                g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("clahe", lambda g: cv2.createCLAHE(
                clipLimit=2.0, tileGridSize=(8, 8)).apply(g)),
        ]

        for name, transform in strategies:
            try:
                processed = transform(gray)
                codes = self._decode(processed)
                if codes:
                    logger.debug("增强解码成功 (%s): %d codes", name, len(codes))
                    return codes
            except Exception:
                continue

        return []
