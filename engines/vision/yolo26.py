"""YOLO26 - 边缘实时检测"""
from typing import Any
from engines.base import BaseEngine, EngineMeta, EngineState


class YOLO26Engine(BaseEngine):
    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="yolo26",
            display_name="YOLO26m (边缘实时)",
            category="vision",
            vram_gb=4.0,
            license="AGPL-3.0",
            description="[stub] 边缘设备实时检测, 低延迟高吞吐",
            tags=["检测", "边缘", "实时", "轻量"],
        )

    def load(self) -> None:
        # TODO: Phase 1 - 加载 YOLO26 模型
        self.state = EngineState.READY

    def infer(self, image_path: str, **kwargs) -> Any:
        # TODO: Phase 1 - 执行目标检测
        return {"boxes": [], "labels": [], "scores": [], "note": "[stub] YOLO26 未接入"}

    def unload(self) -> None:
        self._model = None
        self.state = EngineState.UNLOADED
