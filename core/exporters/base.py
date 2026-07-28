"""导出器基类 (Phase 3E)"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExportResult:
    """单次导出结果。"""
    exporter: str
    success: bool
    message: str = ""
    path: str = ""          # 本地文件路径 (文件类导出)
    record_count: int = 0
    errors: list[str] = field(default_factory=list)


class BaseExporter(ABC):
    """导出器抽象基类。

    子类需实现:
    - name: 导出器标识
    - export(contracts, receivables) -> ExportResult
    """

    name: str = "base"

    def __init__(self, config: dict):
        self.config = config
        self.export_cfg = config.get("export", {}) or {}

    @abstractmethod
    def export(self, contracts: list[dict],
               receivables: list[dict]) -> ExportResult:
        """执行导出, 返回结果。"""
        ...

    def _export_dir(self) -> str:
        """获取导出目录 (绝对路径)。"""
        import os
        d = self.export_cfg.get("dir", "exports")
        os.makedirs(d, exist_ok=True)
        return d
