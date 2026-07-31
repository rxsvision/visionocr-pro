"""轻量实时检测引擎 (stub)

预留边缘设备实时目标检测能力, 低延迟高吞吐。
待选型确定后接入具体推理后端 (Apache-2.0 优先)。
"""
from typing import Any
from engines.base import BaseEngine, EngineMeta, EngineState


class YOLO26Engine(BaseEngine):
    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="yolo26",
            display_name="轻量实时检测 (边缘)",
            category="vision",
            vram_gb=4.0,
            license="TBD (stub)",
            description="[stub] 边缘设备实时目标检测, 低延迟高吞吐",
            tags=["检测", "边缘", "实时", "轻量"],
        )

    def load(self) -> None:
        # TODO: 接入轻量检测模型后实现
        self.state = EngineState.READY

    def infer(self, image_path: str, **kwargs) -> Any:
        # TODO: 接入轻量检测模型后实现
        return {"boxes": [], "labels": [], "scores": [], "note": "[stub] 轻量检测引擎未接入"}

    def unload(self) -> None:
        self._model = None
        self.state = EngineState.UNLOADED
