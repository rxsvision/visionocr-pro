"""合同自动化 Tab (Phase 3F)

管线: 上传合同 → 文档读取(PDF文本/OCR) → 分级LLM抽取(本地优先/云端兜底/规则兜底)
      → 方向判定 + 金额勾稽 → 落库(contracts + receivables) → 人工复核门控 → 应收计划表
复核: 低置信度标红, 原文对照 + 字段可编辑 + 确认/驳回, reviewed=1 后才进入回款日程。
提醒: 扫描已复核合同的 pending 应收, 逾期/7/3/1 天四级桌面通知。
错误: 结构化错误定位面板, 按阶段/错误码/文件筛选。
看板: KPI 卡片 + 多维筛选 + 月度趋势图 + 逾期排行 (Phase 3F)。
"""
from __future__ import annotations

from ui.safe_yield import safe_generator

import json
import os
from datetime import date

import gradio as gr

from core.config import load_config
from core.database import get_conn
from core.dedup import compute_sha256, check_duplicate, register_file
from core.document_reader import read_document
from core.contract_extractor import extract_contract
from core.payment_store import (
    save_contract, save_receivables, list_receivables,
    list_contracts, check_reminders, log_error, list_errors,
    list_pending_review, get_contract_detail,
    update_contract_fields, update_receivable_fields,
    mark_reviewed, reject_contract,
    upsert_signer, list_signers, delete_signer, outstanding_by_signer,
    save_risk_alerts, list_risk_alerts, list_contracts_with_risks,
    dashboard_kpi, monthly_trend, overdue_ranking, filter_contracts,
)
from core.risk_engine import scan_risks
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

    # ─── Phase 3F: 数据看板 ─────────────────────────────────
    with gr.Accordion("数据看板 (KPI · 趋势 · 逾期排行 · 多维筛选)", open=True):
        kpi_display = gr.Markdown("点击「刷新看板」加载数据")
        with gr.Row():
            dash_refresh_btn = gr.Button("刷新看板", variant="primary", scale=1)
            dash_months = gr.Slider(3, 24, value=12, step=1,
                                    label="趋势月数", scale=2)
        trend_plot = gr.Plot(label="月度应收 vs 实收趋势")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 逾期排行")
                rank_by = gr.Dropdown(choices=["按签单人", "按合同"],
                                      value="按签单人", label="维度")
                rank_table = gr.Dataframe(
                    headers=["签单人", "逾期金额", "逾期笔数", "合同数"],
                    label="逾期排行",
                    wrap=True,
                )
            with gr.Column(scale=2):
                gr.Markdown("#### 多维筛选")
                with gr.Row():
                    f_signer = gr.Textbox(label="签单人 (模糊)", scale=2)
                    f_direction = gr.Dropdown(
                        label="方向", choices=["", "receivable", "payable"],
                        value="", scale=1)
                    f_reviewed = gr.Dropdown(
                        label="复核", choices=["", "yes", "no"],
                        value="", scale=1)
                with gr.Row():
                    f_date_from = gr.Textbox(label="起始日 (YYYY-MM-DD)", scale=1)
                    f_date_to = gr.Textbox(label="截止日 (YYYY-MM-DD)", scale=1)
                    f_amt_min = gr.Number(label="最小金额", scale=1)
                    f_amt_max = gr.Number(label="最大金额", scale=1)
                filter_btn = gr.Button("筛选", variant="secondary")
                filter_table = gr.Dataframe(
                    headers=["合同", "编号", "签单人", "对方", "总额",
                             "已收", "未收", "方向", "复核", "置信度"],
                    label="筛选结果",
                    wrap=True,
                )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 合同上传与提取")
            file_upload = gr.File(
                label="上传合同 (PDF/图片/HEIC/TIFF, 支持批量)",
                file_count="multiple",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif",
                            ".bmp", ".webp", ".heic", ".heif"],
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
                gr.Markdown("---")
                batch_threshold = gr.Slider(
                    0.5, 1.0, value=0.85, step=0.05,
                    label="批量通过阈值 (置信度 ≥ 此值自动通过)")
                batch_approve_btn = gr.Button(
                    "批量通过 (高置信度)", variant="secondary")
                batch_msg = gr.Markdown("")
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
                                                 choices=["receivable", "payable", "unknown"])
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

    # ─── 签单人映射管理 ──────────────────────────────────────
    with gr.Accordion("签单人映射 (人名 → 飞书/企微账号)", open=False):
        signer_table = gr.Dataframe(
            headers=["姓名", "飞书ID", "企微ID", "手机", "备注"],
            label="签单人列表",
            wrap=True,
        )
        with gr.Row():
            s_name = gr.Textbox(label="姓名 (合同中的)", scale=2)
            s_feishu = gr.Textbox(label="飞书 open_id", scale=2)
            s_wecom = gr.Textbox(label="企微 userid", scale=2)
        with gr.Row():
            s_phone = gr.Textbox(label="手机号", scale=1)
            s_note = gr.Textbox(label="备注", scale=2)
            s_save_btn = gr.Button("保存/更新", variant="primary")
            s_del_btn = gr.Button("删除")
        signer_refresh_btn = gr.Button("刷新列表")
        signer_msg = gr.Markdown("")

    # ─── 回款日程 (按签单人聚合) ─────────────────────────────
    with gr.Accordion("回款日程 · 按签单人汇总未收", open=False):
        schedule_table = gr.Dataframe(
            headers=["签单人", "合同数", "应收总额", "已收总额", "未收余额"],
            label="未收回款汇总",
            wrap=True,
        )
        schedule_refresh_btn = gr.Button("刷新汇总")

    # ─── 风险预警面板 ────────────────────────────────────────
    with gr.Accordion("合同风险预警 (红=高风险 / 黄=需关注)", open=False):
        risk_summary_table = gr.Dataframe(
            headers=["合同", "风险数", "红色", "置信度", "已复核"],
            label="风险合同总览",
            wrap=True,
        )
        risk_detail_table = gr.Dataframe(
            headers=["级别", "规则", "消息", "证据"],
            label="风险明细 (红色优先)",
            wrap=True,
        )
        risk_refresh_btn = gr.Button("刷新风险预警")

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
    batch_approve_btn.click(
        fn=_batch_approve, inputs=[batch_threshold],
        outputs=[batch_msg, review_list],
    )

    # 错误
    err_search_btn.click(
        fn=_search_errors, inputs=[err_stage, err_code, err_file],
        outputs=[error_table],
    )

    # 签单人映射
    s_save_btn.click(
        fn=_save_signer, inputs=[s_name, s_feishu, s_wecom, s_phone, s_note],
        outputs=[signer_msg, signer_table],
    )
    s_del_btn.click(
        fn=_delete_signer, inputs=[s_name],
        outputs=[signer_msg, signer_table],
    )
    signer_refresh_btn.click(fn=_refresh_signers, outputs=[signer_table])

    # 回款日程
    schedule_refresh_btn.click(fn=_refresh_schedule, outputs=[schedule_table])

    # 风险预警
    risk_refresh_btn.click(
        fn=_refresh_risks, outputs=[risk_summary_table, risk_detail_table])

    # Phase 3F 看板
    dash_refresh_btn.click(
        fn=_refresh_dashboard, inputs=[dash_months, rank_by],
        outputs=[kpi_display, trend_plot, rank_table],
    )
    rank_by.change(
        fn=_refresh_rank, inputs=[rank_by], outputs=[rank_table],
    )
    filter_btn.click(
        fn=_filter_contracts_ui,
        inputs=[f_signer, f_direction, f_reviewed,
                f_date_from, f_date_to, f_amt_min, f_amt_max],
        outputs=[filter_table],
    )


# ─── 提取回调 (流式进度) ─────────────────────────────────────
@safe_generator(lambda e: ([], f"[ERROR] 提取异常: {e}"))
def _extract_contracts(files):
    """Generator: 逐文件处理并 yield 进度, Gradio 自动流式渲染。"""
    if not files:
        yield [], "请先上传合同文件。"
        return

    registry = _registry
    cfg = _get_config()
    data_dir = cfg.get("data_dir", "data")
    company = cfg.get("company", {})
    conn = get_conn(data_dir)

    statuses: list[str] = []
    total_items = 0
    ok_count = 0
    fail_count = 0
    dup_count = 0
    tiers_used: dict[str, int] = {}
    total_files = len(files)

    try:
        for idx, f in enumerate(files, 1):
            path = f.name if hasattr(f, "name") else str(f)
            name = os.path.basename(path)
            try:
                if not os.path.isfile(path):
                    statuses.append(f"⚠ 跳过(文件不存在): {name}")
                    fail_count += 1
                    log_error(conn, "read", "FILE_NOT_FOUND",
                              f"文件不存在: {path}", file_path=path)
                    yield _refresh_table(), _progress_summary(
                        idx, total_files, ok_count, dup_count, fail_count,
                        total_items, tiers_used, statuses)
                    continue

                # 重复检测: SHA-256 内容哈希
                sha = compute_sha256(path)
                existing = check_duplicate(conn, sha)
                if existing:
                    dup_count += 1
                    prev = existing.get("file_name") or existing.get("file_path", "")
                    statuses.append(
                        f"⚠ 跳过(重复): {name} — 与已入库文件相同 ({prev}, "
                        f"入库于 {existing.get('created_at', '?')})"
                    )
                    yield _refresh_table(), _progress_summary(
                        idx, total_files, ok_count, dup_count, fail_count,
                        total_items, tiers_used, statuses)
                    continue

                doc = read_document(path, registry)
                if doc.get("error"):
                    statuses.append(f"⚠ {name}: {doc['error']}")
                    fail_count += 1
                    log_error(conn, "ocr", "DOC_READ_FAIL",
                              doc["error"], file_path=path)
                    yield _refresh_table(), _progress_summary(
                        idx, total_files, ok_count, dup_count, fail_count,
                        total_items, tiers_used, statuses)
                    continue
                text = doc.get("text", "")
                if not text.strip():
                    statuses.append(f"⚠ {name}: 未识别到文本")
                    fail_count += 1
                    log_error(conn, "ocr", "EMPTY_TEXT",
                              "OCR/文本提取结果为空", file_path=path)
                    yield _refresh_table(), _progress_summary(
                        idx, total_files, ok_count, dup_count, fail_count,
                        total_items, tiers_used, statuses)
                    continue

                result, tier = _routed_extract(registry, cfg, text, company)
                tiers_used[tier] = tiers_used.get(tier, 0) + 1

                cid = save_contract(conn, path, result, text)
                n = save_receivables(conn, cid, result.get("payments", []))
                total_items += n
                ok_count += 1

                # 注册文件哈希 (重复检测)
                register_file(conn, sha, path, file_name=name,
                              file_size=os.path.getsize(path), contract_id=cid)

                # 风险扫描
                alerts = scan_risks(result, text)
                if alerts:
                    save_risk_alerts(conn, cid, alerts)

                warn_str = ""
                if result.get("warnings"):
                    warn_str = " · ⚠ " + "; ".join(result["warnings"][:2])
                risk_str = ""
                if alerts:
                    red_n = sum(1 for a in alerts if a["level"] == "red")
                    risk_str = f" · 🚨 {len(alerts)}项风险({red_n}红)"
                conf = result.get("confidence", 0)
                flag = "🔴" if conf < 0.6 else "✓"
                statuses.append(
                    f"{flag} {name} · {doc['pages']}页 · {doc['source']} · "
                    f"[{tier}] 抽取 {n} 条应收 · 置信度 {conf:.0%}{warn_str}{risk_str}"
                )
            except Exception as e:  # noqa: BLE001
                fail_count += 1
                statuses.append(f"✗ {name}: 处理异常 - {e}")
                log_error(conn, "extract", "EXCEPTION", str(e), file_path=path)

            # 每处理完一个文件 yield 一次进度
            yield _refresh_table(), _progress_summary(
                idx, total_files, ok_count, dup_count, fail_count,
                total_items, tiers_used, statuses)
    finally:
        conn.close()


def _progress_summary(idx: int, total: int, ok: int, dup: int, fail: int,
                      items: int, tiers: dict, statuses: list[str]) -> str:
    """生成带进度条的状态摘要。"""
    pct = int(idx / total * 100) if total else 100
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    tier_summary = ", ".join(f"{k}:{v}" for k, v in tiers.items()) or "..."
    header = (
        f"**[{bar}] {idx}/{total} ({pct}%)** · 路由: {tier_summary}\n\n"
        f"成功 {ok} · 重复跳过 {dup} · 失败 {fail} · 入库 {items} 条应收\n\n"
    )
    return header + "\n".join(statuses)


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
    # H7 修复: 日期格式校验
    import re as _re
    _DATE_FMT = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for label, val in [("起始日期", start_date), ("终止日期", end_date)]:
        if val and not _DATE_FMT.match(str(val)):
            return f"⚠ {label}格式无效 (需 YYYY-MM-DD): {val}", _refresh_review_list()
    # H7 修复: 金额校验
    if total is not None and total < 0:
        return "⚠ 合同总额不能为负数。", _refresh_review_list()

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


def _batch_approve(threshold: float) -> tuple[str, list[list]]:
    """批量通过: 将所有置信度 >= threshold 的待复核合同标记为已复核。"""
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    rows = list_pending_review(conn)
    approved = []
    for r in rows:
        if (r.get("confidence") or 0) >= threshold:
            mark_reviewed(conn, r["id"])
            title = r.get("title") or os.path.basename(r.get("file_path") or "")
            approved.append(f"#{r['id']} {title} ({r.get('confidence', 0):.0%})")
    conn.close()
    if approved:
        msg = f"✓ 批量通过 {len(approved)} 份合同 (阈值 {threshold:.0%}):\n\n" + "\n".join(approved)
    else:
        msg = f"无符合条件的合同 (阈值 {threshold:.0%}, 待复核 {len(rows)} 份均低于阈值)。"
    return msg, _refresh_review_list()


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
    fired = check_reminders(conn, today=date.today(), do_notify=True, config=cfg)
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

    from core.exporters import run_export
    results = run_export(cfg, contracts, receivables)
    lines = []
    for r in results:
        icon = "✓" if r.success else "⚠"
        line = f"{icon} [{r.exporter}] {r.message}"
        if r.path:
            line += f" → {r.path}"
        lines.append(line)
    return "\n".join(lines) if lines else "无已启用的导出器。"


# ─── 签单人映射回调 ──────────────────────────────────────────
def _save_signer(name, feishu, wecom, phone, note) -> tuple[str, list[list]]:
    if not name or not name.strip():
        return "请填写签单人姓名。", _refresh_signers()
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    upsert_signer(conn, name.strip(), feishu_id=feishu or "",
                  wecom_id=wecom or "", phone=phone or "", note=note or "")
    conn.close()
    return f"✓ 已保存: {name.strip()}", _refresh_signers()


def _delete_signer(name) -> tuple[str, list[list]]:
    if not name or not name.strip():
        return "请填写要删除的签单人姓名。", _refresh_signers()
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    delete_signer(conn, name.strip())
    conn.close()
    return f"已删除: {name.strip()}", _refresh_signers()


def _refresh_signers() -> list[list]:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    rows = list_signers(conn)
    conn.close()
    return [
        [r.get("name") or "", r.get("feishu_id") or "",
         r.get("wecom_id") or "", r.get("phone") or "", r.get("note") or ""]
        for r in rows
    ]


def _refresh_schedule() -> list[list]:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    rows = outstanding_by_signer(conn)
    conn.close()
    return [
        [r.get("signer") or "—", r.get("contract_count") or 0,
         f"{r.get('total_receivable', 0):,.2f}",
         f"{r.get('total_collected', 0):,.2f}",
         f"{r.get('total_outstanding', 0):,.2f}"]
        for r in rows
    ]


# ─── 风险预警回调 ────────────────────────────────────────────
def _refresh_risks() -> tuple[list[list], list[list]]:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    summary_rows = list_contracts_with_risks(conn)
    detail_rows = list_risk_alerts(conn)
    conn.close()

    summary = [
        [
            r.get("title") or os.path.basename(r.get("file_path") or ""),
            r.get("risk_count") or 0,
            f"🔴 {r.get('red_count') or 0}",
            f"{r.get('confidence', 0):.0%}",
            "是" if r.get("reviewed") else "否",
        ]
        for r in summary_rows
    ]
    detail = [
        [
            "🔴" if r.get("level") == "red" else "🟡",
            r.get("rule") or "—",
            (r.get("message") or "")[:80],
            (r.get("evidence") or "")[:60],
        ]
        for r in detail_rows
    ]
    return summary, detail


# ─── Phase 3F: 看板回调 ─────────────────────────────────────
def _refresh_dashboard(months, rank_dim) -> tuple:
    """刷新 KPI + 趋势图 + 逾期排行。"""
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    try:
        kpi = dashboard_kpi(conn)
        trend = monthly_trend(conn, months=int(months or 12))
        by = "contract" if rank_dim == "按合同" else "signer"
        ranks = overdue_ranking(conn, by=by)
    finally:
        conn.close()

    # KPI Markdown 卡片
    kpi_md = (
        f"| 指标 | 数值 | 指标 | 数值 |\n"
        f"|---|---|---|---|\n"
        f"| 合同总数 | **{kpi['contract_count']}** "
        f"| 待复核 | **{kpi['pending_review']}** |\n"
        f"| 应收总额 | **{kpi['total_receivable']:,.0f}** "
        f"| 已收总额 | **{kpi['total_collected']:,.0f}** |\n"
        f"| 未收余额 | **{kpi['total_outstanding']:,.0f}** "
        f"| 逾期笔数 | **{kpi['overdue_items']}** |\n"
        f"| 逾期金额 | **{kpi['overdue_amount']:,.0f}** "
        f"| 风险预警 | **{kpi['risk_count']}** (🔴{kpi['risk_red']}) |\n"
    )

    # 趋势图
    fig = _build_trend_chart(trend)

    # 逾期排行
    rank_table = _format_rank(ranks, by)

    return kpi_md, fig, rank_table


def _refresh_rank(rank_dim) -> list[list]:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    by = "contract" if rank_dim == "按合同" else "signer"
    ranks = overdue_ranking(conn, by=by)
    conn.close()
    return _format_rank(ranks, by)


def _format_rank(ranks: list[dict], by: str) -> list[list]:
    if not ranks:
        return [["暂无逾期数据", "—", "—", "—"]]
    table = []
    for r in ranks:
        if by == "contract":
            name = r.get("title") or os.path.basename(r.get("file_path") or "")
            table.append([
                name[:30],
                f"{r.get('overdue_amount', 0):,.2f}",
                r.get("overdue_items") or 0,
                r.get("signer") or "—",
            ])
        else:
            table.append([
                r.get("signer") or "未指定",
                f"{r.get('overdue_amount', 0):,.2f}",
                r.get("overdue_items") or 0,
                r.get("contract_count") or 0,
            ])
    return table


def _build_trend_chart(trend: list[dict]):
    """用 matplotlib 绘制月度应收 vs 实收柱状图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        if not trend:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "暂无趋势数据", ha="center", va="center", fontsize=14)
            ax.set_axis_off()
            return fig

        months = [t["month"] for t in trend]
        recv = [t["receivable"] / 10000 for t in trend]
        coll = [t["collected"] / 10000 for t in trend]

        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(months))
        width = 0.35
        ax.bar([i - width / 2 for i in x], recv, width, label="应收", color="#4A90D9")
        ax.bar([i + width / 2 for i in x], coll, width, label="实收", color="#67C23A")
        ax.set_xticks(list(x))
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("金额 (万元)")
        ax.set_title("月度应收 vs 实收")
        ax.legend()
        fig.tight_layout()
        return fig
    except Exception:  # noqa: BLE001
        return None


def _filter_contracts_ui(signer, direction, reviewed,
                         date_from, date_to, amt_min, amt_max) -> list[list]:
    cfg = _get_config()
    conn = get_conn(cfg.get("data_dir", "data"))
    rows = filter_contracts(
        conn,
        signer=signer or "",
        direction=direction or "",
        reviewed=reviewed or "",
        date_from=date_from or "",
        date_to=date_to or "",
        amount_min=amt_min if amt_min else None,
        amount_max=amt_max if amt_max else None,
    )
    conn.close()
    if not rows:
        return [["无匹配结果", "—", "—", "—", "—", "—", "—", "—", "—", "—"]]
    table = []
    for r in rows:
        title = r.get("title") or os.path.basename(r.get("file_path") or "")
        total = r.get("total_amount")
        table.append([
            title[:25],
            r.get("contract_no") or "—",
            r.get("signer") or "—",
            (r.get("counterparty") or "—")[:15],
            f"{total:,.0f}" if isinstance(total, (int, float)) else "—",
            f"{r.get('collected_sum', 0):,.0f}",
            f"{r.get('outstanding', 0):,.0f}",
            _direction_label(r.get("direction")),
            "是" if r.get("reviewed") else "否",
            f"{r.get('confidence', 0):.0%}",
        ])
    return table
