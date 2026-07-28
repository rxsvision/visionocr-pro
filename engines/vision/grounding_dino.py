"""Grounding DINO - 开放词汇检测"""
from typing import Any
from engines.base import BaseEngine, EngineMeta, EngineState


class GroundingDINOEngine(BaseEngine):
    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="grounding_dino",
            display_name="Grounding DINO (开放词汇)",
            category="vision",
            vram_gb=6.0,
            license="Apache-2.0",
            description="开放词汇目标检测, 文本提示驱动, 零样本泛化",
            tags=["检测", "开放词汇", "零样本", "文本驱动"],
        )

    def load(self) -> None:
        # TODO: Phase 1 - 加载 Grounding DINO 模型
        self.state = EngineState.READY

    def infer(self, image_path: str, **kwargs) -> Any:
        # TODO: Phase 1 - 执行开放词汇检测
        return {"boxes": [], "labels": [], "scores": [], "note": "[stub] Grounding DINO 未接入"}

    def unload(self) -> None:
        self._model = None
        self.state = EngineState.UNLOADED
