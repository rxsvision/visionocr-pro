"""导出插件包 (Phase 3E)

插件架构: 所有导出器继承 BaseExporter, 通过 config.export.enabled 按需启用。
内置: ExcelExporter (双Sheet), CSVExporter (通用)
骨架: YonyouExporter (用友U8/T+), KingdeeExporter (金蝶K3/云星空)
"""
from core.exporters.base import BaseExporter, ExportResult
from core.exporters.excel_exporter import ExcelExporter
from core.exporters.csv_exporter import CSVExporter
from core.exporters.yonyou_connector import YonyouExporter
from core.exporters.kingdee_connector import KingdeeExporter

_REGISTRY: dict[str, type[BaseExporter]] = {
    "excel": ExcelExporter,
    "csv": CSVExporter,
    "yonyou": YonyouExporter,
    "kingdee": KingdeeExporter,
}


def get_enabled_exporters(config: dict) -> list[BaseExporter]:
    """根据配置返回已启用的导出器实例列表。"""
    ecfg = config.get("export", {}) or {}
    enabled = ecfg.get("enabled", ["excel"])
    exporters = []
    for name in enabled:
        cls = _REGISTRY.get(name)
        if cls:
            exporters.append(cls(config))
    return exporters


def run_export(config: dict, contracts: list[dict],
               receivables: list[dict]) -> list[ExportResult]:
    """执行所有已启用导出器, 返回结果列表。"""
    results = []
    for exp in get_enabled_exporters(config):
        try:
            r = exp.export(contracts, receivables)
            results.append(r)
        except Exception as e:  # noqa: BLE001
            results.append(ExportResult(
                exporter=exp.name, success=False, message=f"异常: {e}"))
    return results
