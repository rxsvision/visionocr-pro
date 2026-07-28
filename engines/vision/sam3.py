"""SAM 3 - 通用分割"""
from typing import Any
from engines.base import BaseEngine, EngineMeta, EngineState


class SAM3Engine(BaseEngine):
    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="sam3",
            display_name="SAM 3 (分割)",
            category="vision",
            vram_gb=12.0,
            license="Apache-2.0",
            description="通用图像分割, 支持点/框/文本提示, 零样本迁移",
            tags=["分割", "SAM", "交互式", "零样本"],
        )

    def load(self) -> None:
        # TODO: Phase 1 - 加载 SAM 3 模型
        self.state = EngineState.READY

    def infer(self, image_path: str, **kwargs) -> Any:
        # TODO: Phase 1 - 执行分割推理
        return {"masks": [], "scores": [], "note": "[stub] SAM 3 未接入"}

    def unload(self) -> None:
        self._model = None
        self.state = EngineState.UNLOADED
