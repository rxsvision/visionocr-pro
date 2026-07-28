"""合同自动化 Tab (Phase 3B)

管线: 上传合同 → 文档读取(PDF文本/OCR) → 分级LLM抽取(本地优先/云端兜底/规则兜底)
      → 方向判定 + 金额勾稽 → 落库(contracts + receivables) → 人工复核门控 → 应收计划表
复核: 低置信度标红, 原文对照 + 字段可编辑 + 确认/驳回, reviewed=1 后才进入回款日程。
提醒: 扫描已复核合同的 pending 应收, 逾期/7/3/1 天四级桌面通知。
错误: 结构化错误定位面板, 按阶段/错误码/文件筛选。
"""
from __future__ import annotations

import json
import os
from datetime import date

import gradio as gr

from core.config import load_config
from core.database import get_conn
from core.document_reader import read_document
from core.contract_extractor import extract_contract
from core.payment_store import (
    save_contract, save_receivables, list_receivables,
    list_contracts, check_reminders, log_error, list_errors,
    list_pending_review, get_contract_detail,
    update_contract_fields, update_receivable_fields,
    mark_reviewed, reject_contract,
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

    # ─── 人工复核面板 ────────────────────────────────────────
    with gr.Accordion("人工复核 (低置信度标红, 确认后才进入回款日程)", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                review_list = gr.Dataframe(
                    headers=["ID", "合同", "编号", "置信度", "来源", "状态"],
                    label="待复核列表",
                    wrap=True,
                )
                review_refresh_btn = gr.Button("刷新待复核")
            with gr.Column(scale=2):
                review_id_input = gr.Number(label="选择合同 ID", precision=0)
                load_detail_btn = gr.Button("加载详情")
                with gr.Row():
                    raw_text_box = gr.Textbox(label="合同原文 (前2000字)", lines=8,
                                              interactive=False)
                with gr.Row():
                    edit_title = gr.Textbox(label="标题")
                    edit_signer = gr.Textbox(label="签单人")
                with gr.Row():
                    edit_our_party = gr.Textbox(label="我方主体")
                    edit_counterparty = gr.Textbox(label="对方主体")
                with gr.Row():
                    edit_total = gr.Number(label="合同总额")
                    edit_direction = gr.Dropdown(label="方向",
                                                 choices=["receivable", "payable"])
                with gr.Row():
                    edit_start = gr.Textbox(label="起始日期 (YYYY-MM-DD)")
                    edit_end = gr.Textbox(label="终止日期 (YYYY-MM-DD)")
                recv_detail_box = gr.Dataframe(
                    headers=["ID", "到期日", "金额", "条件", "方向"],
                    label="应收条目 (只读, 修正在下方)",
                    wrap=True,
                )
                with gr.Row():
                    confirm_btn = gr.Button("✓ 确认通过", variant="primary")
                    reject_btn = gr.Button("✗ 驳回", variant="stop")
                review_msg = gr.Markdown("")

    # ─── 合同总览 ────────────────────────────────────────────
    with gr.Accordion("合同总览 (含未收余额)", open=False):
        contract_table = gr.Dataframe(
            headers=["合同", "编号", "我方", "对方", "总额", "已收", "未收", "方向", "置信度"],
            label="合同总览",
            wrap=True,
        )
        contract_refresh_btn = gr.Button("刷新合同总览")

    # ─── 日程提醒 ────────────────────────────────────────────
    with gr.Accordion("日程提醒 (逾期/7/3/1 天, 仅已复核)", open=False):
        reminder_status = gr.Textbox(label="提醒状态", lines=6, interactive=False)
        check_btn = gr.Button("检查到期提醒")

    # ─── 错误定位面板 ────────────────────────────────────────
    with gr.Accordion("错误定位 (按阶段/错误码/文件筛选)", open=False):
        with gr.Row():
            err_stage = gr.Dropdown(
                label="阶段", choices=["", "read", "ocr", "extract", "store",
                                       "review", "notify", "export"],
                value="",
            )
            err_code = gr.Textbox(label="错误码 (模糊)", placeholder="如 EMPTY_TEXT")
            err_file = gr.Textbox(label="文件名 (模糊)", placeholder="如 contract.pdf")
            err_search_btn = gr.Button("查询")
        error_table = gr.Dataframe(
            headers=["时间", "阶段", "错误码", "文件", "字段", "消息", "建议"],
            label="错误日志",
            wrap=True,
        )

    # ─── 事件绑定 ────────────────────────────────────────────
    extract_btn.click(
        fn=_extract_contracts, inputs=[file_upload],
        outputs=[result_table, status_box],
    )
    refresh_btn.click(fn=_refresh_table, outputs=[result_table])
    contract_refresh_btn.click(fn=_refresh_contracts, outputs=[contract_table])
    check_btn.click(fn=_check_reminders, outputs=[reminder_status])
    export_btn.click(fn=_export_excel, outputs=[status_box])

    # 复核
    review_refresh_btn.click(fn=_refresh_review_list, outputs=[review_list])
    load_detail_btn.click(
        fn=_load_review_detail, inputs=[review_id_input],
        outputs=[raw_text_box, edit_title, edit_signer, edit_our_party,
                 edit_counterparty, edit_total, edit_direction,
                 edit_start, edit_end, recv_detail_box],
    )
    confirm_btn.click(
        fn=_confirm_review,
        inputs=[review_id_input, edit_title, edit_signer, edit_our_party,
                edit_counterparty, edit_total, edit_direction, edit_start, edit_end],
        outputs=[review_msg, review_list],
    )
    reject_btn.click(
        fn=_reject_review, inputs=[review_id_input],
        outputs=[review_msg, review_list],
    )

    # 错误
    err_search_btn.click(
        fn=_search_errors, inputs=[err_stage, err_code, err_file],
        outputs=[error_table],
    )


# ─── 提取回调 ────────────────────────────────────────────────
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

                result, tier = _routed_extract(registry, cfg, text, company)
                tiers_used[tier] = tiers_used.get(tier, 0) + 1

                cid = save_contract(conn, path, result, text)
                n = save_receivables(conn, cid, result.get("payments", []))
                total_items += n
                ok_count += 1

                warn_str = ""
                if result.get("warnings"):
                    warn_str = " · ⚠ " + "; ".join(result["warnings"][:2])
                conf = result.get("confidence", 0)
                flag = "🔴" if conf < 0.6 else "✓"
                statuses.append(
                    f"{flag} {name} · {doc['pages']}页 · {doc['source']} · "
                    f"[{tier}] 抽取 {n} 条应收 · 置信度 {conf:.0%}{warn_str}"
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


# ─── 复核回调 ────────────────────────────────────────────────
def _refresh_review_list() -> list[list]:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    rows = list_pending_review(conn)
    conn.close()
    table = []
    for r in rows:
        title = r.get("title") or os.path.basename(r.get("file_path") or "")
        conf = r.get("confidence", 0)
        conf_str = f"{'🔴 ' if conf < 0.6 else ''}{conf:.0%}"
        table.append([
            r["id"], title, r.get("contract_no") or "—",
            conf_str, r.get("extract_source") or "—", "待复核",
        ])
    return table


def _load_review_detail(contract_id) -> tuple:
    empty = ("", "", "", "", "", None, "receivable", "", "", [])
    if not contract_id:
        return empty
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    detail = get_contract_detail(conn, int(contract_id))
    conn.close()
    if not detail:
        return empty

    raw = (detail.get("raw_text") or "")[:2000]
    recv_table = [
        [r["id"], r.get("due_date") or "待定", r.get("amount") or "—",
         (r.get("condition_text") or "")[:50], r.get("direction") or "—"]
        for r in detail.get("receivables", [])
    ]
    return (
        raw,
        detail.get("title") or "",
        detail.get("signer") or "",
        detail.get("our_party") or "",
        detail.get("counterparty") or "",
        detail.get("total_amount"),
        detail.get("direction") or "receivable",
        detail.get("start_date") or "",
        detail.get("end_date") or "",
        recv_table,
    )


def _confirm_review(contract_id, title, signer, our_party, counterparty,
                    total, direction, start_date, end_date) -> tuple[str, list[list]]:
    if not contract_id:
        return "请先选择合同 ID。", _refresh_review_list()
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    fields = {}
    if title is not None:
        fields["title"] = title
    if signer is not None:
        fields["signer"] = signer
    if our_party is not None:
        fields["our_party"] = our_party
    if counterparty is not None:
        fields["counterparty"] = counterparty
    if total is not None:
        fields["total_amount"] = total
    if direction:
        fields["direction"] = direction
    if start_date:
        fields["start_date"] = start_date
    if end_date:
        fields["end_date"] = end_date
    update_contract_fields(conn, int(contract_id), fields)
    mark_reviewed(conn, int(contract_id))
    conn.close()
    return f"✓ 合同 #{int(contract_id)} 已确认通过, 正式进入回款日程。", _refresh_review_list()


def _reject_review(contract_id) -> tuple[str, list[list]]:
    if not contract_id:
        return "请先选择合同 ID。", _refresh_review_list()
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    reject_contract(conn, int(contract_id), reason="人工复核驳回")
    conn.close()
    return f"✗ 合同 #{int(contract_id)} 已驳回 (标记为 terminated)。", _refresh_review_list()


# ─── 错误定位回调 ────────────────────────────────────────────
def _search_errors(stage, code, file) -> list[list]:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    rows = list_errors(conn, limit=100, stage=stage or "",
                       error_code=code or "", file_path=file or "")
    conn.close()
    table = []
    for r in rows:
        table.append([
            r.get("created_at") or "—",
            r.get("stage") or "—",
            r.get("error_code") or "—",
            os.path.basename(r.get("file_path") or "") or "—",
            r.get("field") or "—",
            (r.get("message") or "")[:80],
            (r.get("suggestion") or "")[:60],
        ])
    return table


# ─── 通用回调 ────────────────────────────────────────────────
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
        return f"[{date.today().isoformat()}] 暂无逾期或 7 天内到期的应收 (仅已复核合同), 或已全部提醒。"
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
                _direction_label(r.get("direction")),
                f"{r.get('confidence', 0):.0%}",
                "是" if r.get("reviewed") else "否",
            ])

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
