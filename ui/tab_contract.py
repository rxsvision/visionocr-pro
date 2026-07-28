"""合同自动化 Tab (Phase 2 实现)

管线: 上传合同 → 文档读取(PDF文本/OCR) → LLM抽取(规则兜底) → 落库 → 付款计划表。
提醒: 扫描 pending 付款, 7/3/1 天三级桌面通知。
"""
from __future__ import annotations

import os
from datetime import date

import gradio as gr

from core.config import load_config
from core.database import get_conn
from core.document_reader import read_document
from core.contract_extractor import extract_payments
from core.payment_store import (
    save_contract, save_payments, list_payments, check_reminders,
)
from engines.llm.router import get_llm

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
            extract_btn = gr.Button("提取付款条款", variant="primary")
            status_box = gr.Markdown("")

        with gr.Column(scale=2):
            gr.Markdown("### 付款计划")
            result_table = gr.Dataframe(
                headers=["合同", "付款日期", "金额", "币种", "条件", "来源", "状态"],
                label="付款计划",
                wrap=True,
            )
            with gr.Row():
                refresh_btn = gr.Button("刷新列表")
                export_btn = gr.Button("导出 Excel")

    with gr.Accordion("日程提醒 (7/3/1 天)", open=False):
        reminder_status = gr.Textbox(label="提醒状态", lines=6, interactive=False)
        check_btn = gr.Button("检查到期提醒")

    extract_btn.click(
        fn=_extract_payments,
        inputs=[file_upload],
        outputs=[result_table, status_box],
    )
    refresh_btn.click(fn=_refresh_table, outputs=[result_table])
    check_btn.click(fn=_check_reminders, outputs=[reminder_status])
    export_btn.click(fn=_export_excel, outputs=[status_box])


# ─── 回调 ────────────────────────────────────────────────────
def _extract_payments(files) -> tuple[list[list], str]:
    if not files:
        return [], "请先上传合同文件。"

    registry = _registry
    cfg = _get_config()
    data_dir = cfg.get("data_dir", "data")
    conn = get_conn(data_dir)

    llm = get_llm(registry, cfg) if registry else None
    use_llm = llm is not None
    statuses: list[str] = []
    total_payments = 0
    ok_count = 0
    fail_count = 0

    try:
        for f in files:
            path = f.name if hasattr(f, "name") else str(f)
            name = os.path.basename(path)
            try:
                if not os.path.isfile(path):
                    statuses.append(f"⚠ 跳过(文件不存在): {name}")
                    fail_count += 1
                    continue

                doc = read_document(path, registry)
                if doc.get("error"):
                    statuses.append(f"⚠ {name}: {doc['error']}")
                    fail_count += 1
                    continue
                text = doc.get("text", "")
                if not text.strip():
                    statuses.append(f"⚠ {name}: 未识别到文本")
                    fail_count += 1
                    continue

                result = extract_payments(text, llm=llm)
                cid = save_contract(conn, path, result.get("title", ""),
                                    result.get("parties", ""), text, result)
                n = save_payments(conn, cid, result.get("payments", []))
                total_payments += n
                ok_count += 1
                statuses.append(
                    f"✓ {name} · {doc['pages']}页 · "
                    f"{doc['source']} · 抽取 {n} 条付款"
                )
            except Exception as e:  # noqa: BLE001 单份失败不中断整批
                fail_count += 1
                statuses.append(f"✗ {name}: 处理异常 - {e}")
    finally:
        conn.close()

    mode = "LLM结构化抽取" if use_llm else "规则兜底(LLM不可用)"
    summary = (
        f"**模式: {mode}** · 共 {len(files)} 份, 成功 {ok_count}, "
        f"失败 {fail_count}, 入库 {total_payments} 条付款\n\n" + "\n".join(statuses)
    )
    return _refresh_table(), summary


def _source_label(source: str | None) -> str:
    return {"llm": "LLM", "regex": "规则"}.get(source or "", "—")


def _refresh_table() -> list[list]:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    rows = list_payments(conn)
    conn.close()
    table = []
    for r in rows:
        title = r.get("contract_title") or os.path.basename(r.get("file_path") or "")
        amount = r.get("amount")
        amount_str = f"{amount:,.2f}" if isinstance(amount, (int, float)) else "—"
        table.append([
            title,
            r.get("due_date") or "待定",
            amount_str,
            r.get("currency") or "CNY",
            (r.get("condition_text") or "")[:40],
            _source_label(r.get("source")),
            r.get("status") or "pending",
        ])
    return table


def _check_reminders() -> str:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    fired = check_reminders(conn, today=date.today(), do_notify=True)
    conn.close()
    if not fired:
        return f"[{date.today().isoformat()}] 暂无 7 天内到期的付款, 或已全部提醒。"
    lines = [f"[{date.today().isoformat()}] 触发 {len(fired)} 条提醒:"]
    for item in fired:
        lines.append(f"  · [{item['level']}] {item['message']}")
    return "\n".join(lines)


def _export_excel() -> str:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    rows = list_payments(conn)
    conn.close()
    if not rows:
        return "无数据可导出。"
    try:
        import openpyxl
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "付款计划"
        ws.append(["合同", "付款日期", "金额", "币种", "条件", "来源", "状态"])
        for r in rows:
            title = r.get("contract_title") or os.path.basename(r.get("file_path") or "")
            ws.append([
                title, r.get("due_date") or "待定", r.get("amount") or "",
                r.get("currency") or "CNY", r.get("condition_text") or "",
                _source_label(r.get("source")), r.get("status") or "pending",
            ])
        out_dir = cfg.get("data_dir", "data")
        out_path = os.path.join(out_dir, f"payments_{date.today().isoformat()}.xlsx")
        wb.save(out_path)
        return f"已导出: {out_path}"
    except Exception as e:  # noqa: BLE001
        return f"导出失败: {e}"
