"""CSV 通用导出器 (Phase 3E)

输出: {export_dir}/contracts_{date}.csv + receivables_{date}.csv
适用于: 无 API 的 ERP/MES 系统, 人工导入或第三方中间件消费。
"""
from __future__ import annotations

import csv
import os
from datetime import date

from core.exporters.base import BaseExporter, ExportResult


class CSVExporter(BaseExporter):
    name = "csv"

    def export(self, contracts: list[dict],
               receivables: list[dict]) -> ExportResult:
        out_dir = self._export_dir()
        today = date.today().isoformat()
        paths = []
        errors = []

        # 合同主表
        c_path = os.path.join(out_dir, f"contracts_{today}.csv")
        try:
            with open(c_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["id", "contract_no", "title", "our_party", "counterparty",
                            "signer", "start_date", "end_date", "total_amount",
                            "currency", "direction", "confidence", "reviewed", "status"])
                for r in contracts:
                    w.writerow([
                        r.get("id"), r.get("contract_no"), r.get("title"),
                        r.get("our_party"), r.get("counterparty"), r.get("signer"),
                        r.get("start_date"), r.get("end_date"), r.get("total_amount"),
                        r.get("currency"), r.get("direction"), r.get("confidence"),
                        r.get("reviewed"), r.get("status"),
                    ])
            paths.append(c_path)
        except Exception as e:  # noqa: BLE001
            errors.append(f"contracts.csv: {e}")

        # 应收明细
        r_path = os.path.join(out_dir, f"receivables_{today}.csv")
        try:
            with open(r_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["id", "contract_id", "due_date", "amount", "currency",
                            "condition_text", "penalty", "direction", "status", "source"])
                for r in receivables:
                    w.writerow([
                        r.get("id"), r.get("contract_id"), r.get("due_date"),
                        r.get("amount"), r.get("currency"), r.get("condition_text"),
                        r.get("penalty"), r.get("direction"), r.get("status"),
                        r.get("source"),
                    ])
            paths.append(r_path)
        except Exception as e:  # noqa: BLE001
            errors.append(f"receivables.csv: {e}")

        success = len(errors) == 0
        msg = f"导出 {len(paths)} 个文件" + (f", {len(errors)} 个错误" if errors else "")
        return ExportResult(self.name, success, msg,
                            path="; ".join(paths), record_count=len(receivables),
                            errors=errors)
