"""Excel 双 Sheet 导出器 (Phase 3E)

输出: {export_dir}/receivables_{date}.xlsx
Sheet1: 合同总览 (含未收余额/复核状态)
Sheet2: 应收明细 (含签单人/条件/违约条款)
"""
from __future__ import annotations

import os
from datetime import date

from core.exporters.base import BaseExporter, ExportResult


class ExcelExporter(BaseExporter):
    name = "excel"

    def export(self, contracts: list[dict],
               receivables: list[dict]) -> ExportResult:
        try:
            from openpyxl import Workbook
        except ImportError:
            return ExportResult(self.name, False, "openpyxl 未安装")

        out_dir = self._export_dir()
        out_path = os.path.join(out_dir, f"receivables_{date.today().isoformat()}.xlsx")

        wb = Workbook()
        # Sheet 1: 合同总览
        ws1 = wb.active
        ws1.title = "合同总览"
        ws1.append(["合同", "编号", "我方主体", "对方主体", "签单人",
                    "起始日", "终止日", "总额", "已收", "未收", "方向", "置信度", "已复核"])
        for r in contracts:
            title = r.get("title") or os.path.basename(r.get("file_path") or "")
            ws1.append([
                title, r.get("contract_no") or "", r.get("our_party") or "",
                r.get("counterparty") or "", r.get("signer") or "",
                r.get("start_date") or "", r.get("end_date") or "",
                r.get("total_amount") or "", r.get("collected_sum") or 0,
                r.get("outstanding") or 0,
                r.get("direction") or "", f"{r.get('confidence', 0):.0%}",
                "是" if r.get("reviewed") else "否",
            ])

        # Sheet 2: 应收明细
        ws2 = wb.create_sheet("应收明细")
        ws2.append(["合同", "签单人", "到期日", "金额", "币种", "方向",
                    "条件", "违约条款", "来源", "状态"])
        for r in receivables:
            title = r.get("contract_title") or os.path.basename(r.get("file_path") or "")
            ws2.append([
                title, r.get("signer") or "", r.get("due_date") or "待定",
                r.get("amount") or "", r.get("currency") or "CNY",
                r.get("direction") or "", r.get("condition_text") or "",
                r.get("penalty") or "", r.get("source") or "",
                r.get("status") or "pending",
            ])

        wb.save(out_path)
        return ExportResult(
            self.name, True, f"已导出 {len(contracts)} 份合同 + {len(receivables)} 条应收",
            path=out_path, record_count=len(receivables))
