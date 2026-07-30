"""RTMPose - 人体姿态估计"""
from typing import Any
from engines.base import BaseEngine, EngineMeta, EngineState


class RTMPoseEngine(BaseEngine):
    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="rtmpose",
            display_name="RTMPose-m (姿态)",
            category="pose",
            vram_gb=2.0,
            license="Apache-2.0",
            description="[stub] 实时人体姿态估计, 17/133 关键点, 支持多人",
            tags=["姿态", "关键点", "实时", "多人"],
        )

    def load(self) -> None:
        # TODO: Phase 1 - 加载 RTMPose 模型
        self.state = EngineState.READY

    def infer(self, image_path: str, **kwargs) -> Any:
        # TODO: Phase 1 - 执行姿态估计
        return {"keypoints": [], "scores": [], "note": "[stub] RTMPose 未接入"}

    def unload(self) -> None:
        self._model = None
        self.state = EngineState.UNLOADED
