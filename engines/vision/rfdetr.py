"""RF-DETR - 目标检测"""
from typing import Any
from engines.base import BaseEngine, EngineMeta, EngineState


class RFDETREngine(BaseEngine):
    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="rfdetr",
            display_name="RF-DETR Large (检测)",
            category="vision",
            vram_gb=8.0,
            license="Apache-2.0",
            description="[stub] 实时目标检测, COCO mAP 54.3, 支持自定义数据集",
            tags=["检测", "DETR", "实时"],
        )

    def load(self) -> None:
        # TODO: Phase 1 - 加载 RF-DETR 模型
        self.state = EngineState.READY

    def infer(self, image_path: str, **kwargs) -> Any:
        # TODO: Phase 1 - 执行目标检测
        return {"boxes": [], "labels": [], "scores": [], "note": "[stub] RF-DETR 未接入"}

    def unload(self) -> None:
        self._model = None
        self.state = EngineState.UNLOADED
