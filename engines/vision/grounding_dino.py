"""Grounding DINO - 开放词汇零样本检测 (Phase 4A)

基于 HuggingFace transformers 的 zero-shot-object-detection pipeline。
文本提示驱动, 无需训练即可检测任意描述的缺陷/目标。

模型选择:
- grounding-dino-tiny: ~1.2GB VRAM, 速度快, 适合产线节拍
- grounding-dino-base: ~2.5GB VRAM, 精度更高 (默认)

离线使用: 首次加载自动下载并缓存到 HF_HOME, 之后全离线运行。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from engines.base import BaseEngine, EngineMeta, EngineState

logger = logging.getLogger("visionocr.grounding_dino")

# 模型映射 (短名 → HF model id)
_MODEL_MAP = {
    "tiny": "IDEA-Research/grounding-dino-tiny",
    "base": "IDEA-Research/grounding-dino-base",
}


class GroundingDINOEngine(BaseEngine):
    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="grounding_dino",
            display_name="Grounding DINO (开放词汇零样本检测)",
            category="vision",
            vram_gb=2.5,
            license="Apache-2.0",
            description="文本提示驱动的零样本目标检测, 无需训练即可定位任意缺陷",
            tags=["检测", "开放词汇", "零样本", "文本驱动", "缺陷检测"],
            resident=True,
        )

    def load(self) -> None:
        """加载 Grounding DINO 模型 (transformers pipeline)。"""
        try:
            from transformers import pipeline as hf_pipeline
            import torch
        except ImportError as e:
            logger.error("依赖缺失: %s (需要 transformers + torch)", e)
            self.state = EngineState.ERROR
            return

        # 从配置选择模型规格
        qc_cfg = self.config.get("qc", {}) or {}
        model_size = qc_cfg.get("grounding_dino_model", "base")
        model_id = _MODEL_MAP.get(model_size, _MODEL_MAP["base"])

        device = self._resolve_device()

        try:
            logger.info("加载 Grounding DINO (%s) → %s ...", model_id, device)
            pipe_kwargs = dict(
                task="zero-shot-object-detection",
                model=model_id,
                device=device,
            )
            # FP16 可选: 默认 FP32 保证兼容性 (实测仅 ~1GB VRAM, 无需省显存)
            # transformers 5.x 的 image_processor 输出 FP32, 强制 FP16 会导致
            # "expected scalar type Half but found Float" 推理全部失败
            use_fp16 = qc_cfg.get("grounding_dino_fp16", False)
            if device >= 0 and use_fp16:
                import torch
                pipe_kwargs["dtype"] = torch.float16
            self._model = hf_pipeline(**pipe_kwargs)
            self.state = EngineState.READY
            logger.info("Grounding DINO 就绪 (%s)",
                        "FP16" if (device >= 0 and use_fp16) else "FP32")
        except Exception as e:
            logger.error("Grounding DINO 加载失败: %s", e)
            self.state = EngineState.ERROR

    def infer(self, image: Any, prompt: str = "",
              threshold: float = 0.3, **kwargs) -> dict:
        """执行开放词汇检测。

        Args:
            image: PIL.Image / np.ndarray (BGR) / 文件路径
            prompt: 点分隔的检测目标, 如 "scratch.dent.crack.missing screw"
            threshold: 置信度阈值 (低于此值的框丢弃)

        Returns:
            {"boxes": [[x1,y1,x2,y2],...], "labels": [...], "scores": [...],
             "count": int}
        """
        if not self.is_ready():
            return {"boxes": [], "labels": [], "scores": [], "count": 0,
                    "error": "模型未加载"}

        from PIL import Image

        # 统一输入为 PIL Image
        pil_img = self._to_pil(image)
        if pil_img is None:
            return {"boxes": [], "labels": [], "scores": [], "count": 0,
                    "error": "无法解析图像"}

        if not prompt.strip():
            prompt = "defect.scratch.dent.crack"

        # Grounding DINO 要求候选词用点号分隔
        candidate_labels = [p.strip() for p in prompt.replace("。", ".").split(".")
                            if p.strip()]
        if not candidate_labels:
            candidate_labels = ["defect"]

        try:
            results = self._model(
                pil_img,
                candidate_labels=candidate_labels,
                threshold=threshold,
            )
        except Exception as e:
            logger.error("推理失败: %s", e)
            return {"boxes": [], "labels": [], "scores": [], "count": 0,
                    "error": str(e)}

        boxes, labels, scores = [], [], []
        for det in results:
            box = det["box"]  # {"xmin","ymin","xmax","ymax"}
            boxes.append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
            labels.append(det["label"])
            scores.append(round(det["score"], 4))

        return {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "count": len(boxes),
        }

    def unload(self) -> None:
        self._model = None
        self.state = EngineState.UNLOADED
        # 释放 CUDA 缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ─── 内部工具 ────────────────────────────────────────────
    def _resolve_device(self) -> int:
        """根据配置和硬件选择设备。"""
        try:
            import torch
            device_cfg = self.config.get("device", "auto")
            if device_cfg == "cuda" or (device_cfg == "auto" and torch.cuda.is_available()):
                return 0
        except ImportError:
            pass
        return -1  # CPU

    @staticmethod
    def _to_pil(image: Any):
        """将多种输入格式统一为 PIL.Image (RGB)。"""
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            try:
                return Image.open(str(image)).convert("RGB")
            except Exception:
                return None
        if isinstance(image, np.ndarray):
            # 假设 BGR (OpenCV 格式) → RGB
            if image.ndim == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]
            return Image.fromarray(image)
        return None
