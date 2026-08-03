"""YOLO 缺陷检测引擎 — 少样本微调的结构缺陷检测

定位:
    - PatchCore 擅长表面异常 (划痕/色差), 但微观结构缺陷 (缺孔/短路/毛刺) 失效
    - YOLO 少样本微调补位: PCB 6 类结构缺陷, ~693 标注图即可收敛
    - 与 PatchCore 互补, 可经 Union 检测 OR 合并 (零漏检策略)

权重发现优先级:
    1. config["yolo_defect"]["weights"]
    2. finetune/output_yolo/pcb_defect/weights/best.pt
    3. models/yolo_defect.pt

输出格式 (与 grounding_dino 对齐, 供 defect_detector 复用):
    {"boxes": [[x1,y1,x2,y2]], "labels": [str], "scores": [float],
     "count": int, "max_score": float}
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from engines.base import BaseEngine, EngineMeta, EngineState

logger = logging.getLogger("visionocr.yolo_defect")

_ROOT = Path(__file__).resolve().parents[2]
_WEIGHT_CANDIDATES = [
    _ROOT / "finetune" / "output_yolo" / "pcb_defect" / "weights" / "best.pt",
    _ROOT / "models" / "yolo_defect.pt",
]

# PCB 缺陷类别中文显示名
_LABEL_ZH = {
    "missing_hole": "缺孔",
    "mouse_bite": "鼠咬",
    "open_circuit": "开路",
    "short": "短路",
    "spur": "毛刺",
    "spurious_copper": "杂铜",
}


class YOLODefectEngine(BaseEngine):
    def __init__(self, config: dict):
        super().__init__(config)
        ycfg = (config.get("yolo_defect", {}) or {})
        self._weights_cfg = ycfg.get("weights", "")
        self._conf = float(ycfg.get("confidence_threshold", 0.25))
        self._imgsz = int(ycfg.get("imgsz", 1280))
        self._names: dict[int, str] = {}
        self._loaded_product: str | None = None

    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="yolo_defect",
            display_name="YOLO 结构缺陷检测",
            category="vision",
            vram_gb=2.0,
            license="AGPL-3.0 (ultralytics)",
            description="YOLO 少样本微调 (YOLO11 基线, 兼容 YOLOv8 权重), "
                        "检测微观结构缺陷 (缺孔/短路/毛刺等)",
            tags=["检测", "缺陷", "YOLO", "结构缺陷", "PCB"],
        )

    def _resolve_weights(self) -> Path | None:
        if self._weights_cfg:
            p = Path(self._weights_cfg)
            if p.exists():
                return p
            # 显式配置即用户意图, 缺失视为配置错误 (不静默回退)
            logger.warning("配置权重不存在: %s", p)
            return None
        for cand in _WEIGHT_CANDIDATES:
            if cand.exists():
                return cand
        return None

    def load(self) -> None:
        self._load_weights(self._resolve_weights())

    def _load_weights(self, weights: Path | None) -> None:
        if weights is None:
            logger.error("未找到 YOLO 权重, 请先训练: python finetune/train_yolo.py "
                         "(默认 yolo11n 基线), 训练后将 best.pt 复制到 "
                         "models/yolo/{产品名}.pt 启用产品门控")
            self.state = EngineState.ERROR
            return
        try:
            from ultralytics import YOLO
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = YOLO(str(weights))
            self._device = device
            # 读取类别名
            self._names = getattr(self._model, "names", {}) or {}
            logger.info("YOLO 权重加载: %s (device=%s, %d 类)",
                        weights.name, device, len(self._names))
            self.state = EngineState.READY
        except Exception as e:
            logger.error("YOLO 加载失败: %s", e)
            self.state = EngineState.ERROR

    def load_for_product(self, product_name: str | None) -> bool:
        """按产品加载专属权重 (跨域误报防护门控)。

        Returns:
            True  — 该产品有专属权重且已就绪, 可参与检测
            False — 无产品上下文或该产品未训练 YOLO, Union 应跳过本检测源

        不改变「无权重」时的状态语义: 解析不到权重直接返回 False,
        不触发 ERROR (区别于 load() 的显式配置缺失)。
        """
        from core.yolo_products import resolve_yolo_weights
        weights = resolve_yolo_weights(product_name)
        if weights is None:
            return False
        # 已加载同一产品权重 → 复用
        if self.is_ready() and self._loaded_product == product_name:
            return True
        self._load_weights(weights)
        if self.is_ready():
            self._loaded_product = product_name
        return self.is_ready()

    def infer(self, image_path: str, **kwargs) -> Any:
        if not self.is_ready():
            return {"boxes": [], "labels": [], "scores": [], "count": 0,
                    "error": "YOLO 引擎未就绪"}
        conf = float(kwargs.get("confidence_threshold", self._conf))
        imgsz = int(kwargs.get("imgsz", self._imgsz))

        try:
            results = self._predict(image_path, conf, imgsz)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self._device == "cuda":
                logger.warning("YOLO GPU OOM, 降级 CPU 重试")
                self._device = "cpu"
                try:
                    results = self._predict(image_path, conf, imgsz)
                except Exception as e2:
                    return {"boxes": [], "labels": [], "scores": [], "count": 0,
                            "error": f"CPU 重试失败: {e2}"}
            else:
                return {"boxes": [], "labels": [], "scores": [], "count": 0,
                        "error": str(e)}
        except Exception as e:
            return {"boxes": [], "labels": [], "scores": [], "count": 0,
                    "error": str(e)}

        return self._format(results)

    def _predict(self, image_path: str, conf: float, imgsz: int):
        # imread_unicode 兼容中文路径: ultralytics 支持 numpy 数组输入
        from core.imutils import imread_unicode
        img = imread_unicode(image_path)
        if img is None:
            raise ValueError(f"图像读取失败: {image_path}")
        return self._model.predict(
            img, conf=conf, imgsz=imgsz, device=self._device, verbose=False)

    def _format(self, results) -> dict:
        boxes, labels, scores = [], [], []
        for r in results:
            b = getattr(r, "boxes", None)
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            for box, sc, ci in zip(xyxy, conf, cls):
                name = self._names.get(int(ci), str(ci))
                boxes.append([round(float(v), 1) for v in box])
                labels.append(_LABEL_ZH.get(name, name))
                scores.append(round(float(sc), 3))
        return {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "count": len(boxes),
            "max_score": max(scores) if scores else 0.0,
        }

    def unload(self) -> None:
        self._model = None
        self.state = EngineState.UNLOADED
