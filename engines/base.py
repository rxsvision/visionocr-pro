"""引擎基类 - 所有能力引擎的统一接口"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EngineState(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


@dataclass
class EngineMeta:
    """引擎元信息"""
    name: str                       # 唯一标识, e.g. "ovisocr2"
    display_name: str               # 显示名, e.g. "OvisOCR2 (印刷文档)"
    category: str                   # ocr | vision | pose | llm
    vram_gb: float = 0.0           # 预估显存占用
    license: str = "unknown"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    resident: bool = False         # 常驻引擎: 不参与 LRU 驱逐/空闲卸载 (v1.3.0)


class BaseEngine(ABC):
    """所有引擎继承此类"""

    def __init__(self, config: dict):
        self.config = config
        self.state = EngineState.UNLOADED
        self._model = None

    @property
    @abstractmethod
    def meta(self) -> EngineMeta:
        """返回引擎元信息"""
        ...

    @abstractmethod
    def load(self) -> None:
        """加载模型到显存/内存。完成后设置 self.state = READY"""
        ...

    @abstractmethod
    def infer(self, *args, **kwargs) -> Any:
        """执行推理。调用前确保 state == READY"""
        ...

    @abstractmethod
    def unload(self) -> None:
        """释放显存/内存。完成后设置 self.state = UNLOADED"""
        ...

    def is_ready(self) -> bool:
        return self.state == EngineState.READY

    def __repr__(self):
        return f"<{self.__class__.__name__} state={self.state.value}>"
