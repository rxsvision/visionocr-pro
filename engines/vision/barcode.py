"""条码识别 - 一维码/二维码"""
from typing import Any
from engines.base import BaseEngine, EngineMeta, EngineState


class BarcodeEngine(BaseEngine):
    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="barcode",
            display_name="条码识别",
            category="vision",
            vram_gb=0.1,
            license="LGPL/Apache",
            description="[stub] 一维码/二维码识别, 支持 EAN/UPC/QR/DataMatrix 等",
            tags=["条码", "二维码", "QR", "轻量"],
        )

    def load(self) -> None:
        # TODO: Phase 1 - 加载条码识别引擎
        self.state = EngineState.READY

    def infer(self, image_path: str, **kwargs) -> Any:
        # TODO: Phase 1 - 执行条码识别
        return {"codes": [], "note": "[stub] 条码识别未接入"}

    def unload(self) -> None:
        self._model = None
        self.state = EngineState.UNLOADED
