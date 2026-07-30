"""CTR-GCN - 动作识别"""
from typing import Any
from engines.base import BaseEngine, EngineMeta, EngineState


class CTRGCNEngine(BaseEngine):
    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="ctrgcn",
            display_name="CTR-GCN (动作识别)",
            category="pose",
            vram_gb=0.5,
            license="MIT",
            description="[stub] 基于骨架的动作识别, 时空图卷积网络",
            tags=["动作识别", "骨架", "GCN", "时序"],
        )

    def load(self) -> None:
        # TODO: Phase 1 - 加载 CTR-GCN 模型
        self.state = EngineState.READY

    def infer(self, image_path: str, **kwargs) -> Any:
        # TODO: Phase 1 - 执行动作识别
        return {"action": "[stub] CTR-GCN 未接入", "confidence": 0.0}

    def unload(self) -> None:
        self._model = None
        self.state = EngineState.UNLOADED
