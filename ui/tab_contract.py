"""合同自动化 Tab (Phase 3A)

管线: 上传合同 → 文档读取(PDF文本/OCR) → 分级LLM抽取(本地优先/云端兜底/规则兜底)
      → 方向判定 + 金额勾稽 → 落库(contracts + receivables) → 应收计划表
提醒: 扫描 pending 应收, 逾期/7/3/1 天四级桌面通知。
"""
from __future__ import annotations

import os
from datetime import date

import gradio as gr

from core.config import load_config
from core.database import get_conn
from core.document_reader import read_document
from core.contract_extractor import extract_contract
from core.payment_store import (
    save_contract, save_receivables, list_receivables,
    list_contracts, check_reminders, log_error,
)
from engines.llm.router import route_extract, get_llm

_registry = None
_config = None


def set_registry(registry) -> None:
    global _registry
    _registry = registry


def _get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def create_tab_contract(config: dict, registry):
    set_registry(registry)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 合同上传与提取")
            file_upload = gr.File(
                label="上传合同 (PDF/图片, 支持批量)",
                file_count="multiple",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff"],
            )
            extract_btn = gr.Button("提取应收条款", variant="primary")
            status_box = gr.Markdown("")

        with gr.Column(scale=2):
            gr.Markdown("### 应收回款计划")
            result_table = gr.Dataframe(
                headers=["合同", "签单人", "到期日", "金额", "币种",
                         "方向", "条件", "来源", "状态"],
                label="应收计划",
                wrap=True,
            )
            with gr.Row():
                refresh_btn = gr.Button("刷新列表")
                export_btn = gr.Button("导出 Excel")

    with gr.Accordion("合同总览 (含未收余额)", open=False):
        contract_table = gr.Dataframe(
            headers=["合同", "编号", "我方", "对方", "总额", "已收", "未收", "方向", "置信度"],
            label="合同总览",
            wrap=True,
        )
        contract_refresh_btn = gr.Button("刷新合同总览")

    with gr.Accordion("日程提醒 (逾期/7/3/1 天)", open=False):
        reminder_status = gr.Textbox(label="提醒状态", lines=6, interactive=False)
        check_btn = gr.Button("检查到期提醒")

    extract_btn.click(
        fn=_extract_contracts,
        inputs=[file_upload],
        outputs=[result_table, status_box],
    )
    refresh_btn.click(fn=_refresh_table, outputs=[result_table])
    contract_refresh_btn.click(fn=_refresh_contracts, outputs=[contract_table])
    check_btn.click(fn=_check_reminders, outputs=[reminder_status])
    export_btn.click(fn=_export_excel, outputs=[status_box])


# ─── 回调 ────────────────────────────────────────────────────
def _extract_contracts(files) -> tuple[list[list], str]:
    if not files:
        return [], "请先上传合同文件。"

    registry = _registry
    cfg = _get_config()
    data_dir = cfg.get("data_dir", "data")
    company = cfg.get("company", {})
    conn = get_conn(data_dir)

    statuses: list[str] = []
    total_items = 0
    ok_count = 0
    fail_count = 0
    tiers_used: dict[str, int] = {}

    try:
        for f in files:
            path = f.name if hasattr(f, "name") else str(f)
            name = os.path.basename(path)
            try:
                if not os.path.isfile(path):
                    statuses.append(f"⚠ 跳过(文件不存在): {name}")
                    fail_count += 1
                    log_error(conn, "read", "FILE_NOT_FOUND",
                              f"文件不存在: {path}", file_path=path)
                    continue

                doc = read_document(path, registry)
                if doc.get("error"):
                    statuses.append(f"⚠ {name}: {doc['error']}")
                    fail_count += 1
                    log_error(conn, "ocr", "DOC_READ_FAIL",
                              doc["error"], file_path=path)
                    continue
                text = doc.get("text", "")
                if not text.strip():
                    statuses.append(f"⚠ {name}: 未识别到文本")
                    fail_count += 1
                    log_error(conn, "ocr", "EMPTY_TEXT",
                              "OCR/文本提取结果为空", file_path=path)
                    continue

                # 分级路由抽取
                result, tier = _routed_extract(
                    registry, cfg, text, company)
                tiers_used[tier] = tiers_used.get(tier, 0) + 1

                cid = save_contract(conn, path, result, text)
                n = save_receivables(conn, cid, result.get("payments", []))
                total_items += n
                ok_count += 1

                # 校验告警
                warn_str = ""
                if result.get("warnings"):
                    warn_str = " · ⚠ " + "; ".join(result["warnings"][:2])
                statuses.append(
                    f"✓ {name} · {doc['pages']}页 · {doc['source']} · "
                    f"[{tier}] 抽取 {n} 条应收 · "
                    f"置信度 {result.get('confidence', 0):.0%}"
                    f"{warn_str}"
                )
            except Exception as e:  # noqa: BLE001
                fail_count += 1
                statuses.append(f"✗ {name}: 处理异常 - {e}")
                log_error(conn, "extract", "EXCEPTION", str(e), file_path=path)
    finally:
        conn.close()

    tier_summary = ", ".join(f"{k}:{v}" for k, v in tiers_used.items()) or "无"
    summary = (
        f"**路由: {tier_summary}** · 共 {len(files)} 份, 成功 {ok_count}, "
        f"失败 {fail_count}, 入库 {total_items} 条应收\n\n" + "\n".join(statuses)
    )
    return _refresh_table(), summary


def _routed_extract(registry, cfg: dict, text: str, company: dict) -> tuple[dict, str]:
    """使用分级路由执行抽取; registry 不可用时降级为纯规则。"""
    if registry is None:
        result = extract_contract(text, llm=None, company=company)
        return result, "regex"

    def _fn(llm):
        return extract_contract(text, llm=llm, company=company)

    try:
        result, tier = route_extract(registry, cfg, _fn)
        if not result:
            result = extract_contract(text, llm=None, company=company)
            tier = "regex"
        return result, tier
    except Exception:  # noqa: BLE001
        result = extract_contract(text, llm=None, company=company)
        return result, "regex"


def _source_label(source: str | None) -> str:
    return {"llm": "LLM", "regex": "规则"}.get(source or "", "—")


def _direction_label(d: str | None) -> str:
    return {"receivable": "应收", "payable": "应付"}.get(d or "", "—")


def _refresh_table() -> list[list]:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    rows = list_receivables(conn)
    conn.close()
    table = []
    for r in rows:
        title = r.get("contract_title") or os.path.basename(r.get("file_path") or "")
        amount = r.get("amount")
        amount_str = f"{amount:,.2f}" if isinstance(amount, (int, float)) else "—"
        table.append([
            title,
            r.get("signer") or "—",
            r.get("due_date") or "待定",
            amount_str,
            r.get("currency") or "CNY",
            _direction_label(r.get("direction")),
            (r.get("condition_text") or "")[:40],
            _source_label(r.get("source")),
            r.get("status") or "pending",
        ])
    return table


def _refresh_contracts() -> list[list]:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    rows = list_contracts(conn)
    conn.close()
    table = []
    for r in rows:
        title = r.get("title") or os.path.basename(r.get("file_path") or "")
        total = r.get("total_amount")
        table.append([
            title,
            r.get("contract_no") or "—",
            r.get("our_party") or "—",
            r.get("counterparty") or "—",
            f"{total:,.2f}" if isinstance(total, (int, float)) else "—",
            f"{r.get('collected_sum', 0):,.2f}",
            f"{r.get('outstanding', 0):,.2f}",
            _direction_label(r.get("direction")),
            f"{r.get('confidence', 0):.0%}",
        ])
    return table


def _check_reminders() -> str:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    fired = check_reminders(conn, today=date.today(), do_notify=True)
    conn.close()
    if not fired:
        return f"[{date.today().isoformat()}] 暂无逾期或 7 天内到期的应收, 或已全部提醒。"
    lines = [f"[{date.today().isoformat()}] 触发 {len(fired)} 条提醒:"]
    for item in fired:
        lines.append(f"  · [{item['level']}] {item['message']}")
    return "\n".join(lines)


def _export_excel() -> str:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    contracts = list_contracts(conn)
    receivables = list_receivables(conn)
    conn.close()
    if not receivables:
        return "无数据可导出。"
    try:
        from openpyxl import Workbook

        export_dir = cfg.get("export", {}).get("dir", cfg.get("data_dir", "data"))
        os.makedirs(export_dir, exist_ok=True)
        out_path = os.path.join(export_dir, f"receivables_{date.today().isoformat()}.xlsx")

        wb = Workbook()
        # Sheet 1: 合同总览
        ws1 = wb.active
        ws1.title = "合同总览"
        ws1.append(["合同", "编号", "我方主体", "对方主体", "签单人",
                    "起始日", "终止日", "总额", "已收", "未收", "方向", "置信度"])
        for r in contracts:
            title = r.get("title") or os.path.basename(r.get("file_path") or "")
            ws1.append([
                title, r.get("contract_no") or "", r.get("our_party") or "",
                r.get("counterparty") or "", r.get("signer") or "",
                r.get("start_date") or "", r.get("end_date") or "",
                r.get("total_amount") or "", r.get("collected_sum") or 0,
                r.get("outstanding") or 0,
                _direction_label(r.get("direction")),
                f"{r.get('confidence', 0):.0%}",
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
                _direction_label(r.get("direction")),
                r.get("condition_text") or "", r.get("penalty") or "",
                _source_label(r.get("source")), r.get("status") or "pending",
            ])

        wb.save(out_path)
        return f"已导出: {out_path}"
    except Exception as e:  # noqa: BLE001
        return f"导出失败: {e}"
