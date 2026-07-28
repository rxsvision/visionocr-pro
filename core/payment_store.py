"""合同/应收存储与到期提醒 (Phase 3A)

职责:
- 把抽取结果落库 (contracts + receivables 两表)。
- 实收登记 (collections), 计算未收余额。
- 按 逾期/7/3/1 天四级阈值扫描到期项, 触发通知并标记 reminded_*。
- 结构化错误日志 (error_log), 支持按阶段/字段快速定位。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from core.notifications import notify


# ─── 写入 ────────────────────────────────────────────────────
def save_contract(conn: sqlite3.Connection, file_path: str,
                  result: dict, raw_text: str) -> int:
    """写入合同主记录 (Phase 3A 新签名), 返回 contract_id。

    result 来自 contract_extractor.extract_contract(), 包含:
    contract_no, title, our_party, counterparty, signer,
    start_date, end_date, total_amount, currency, direction,
    payments, confidence, valid, warnings
    """
    cur = conn.execute(
        """INSERT INTO contracts
           (file_path, title, parties, raw_text, structured_json,
            contract_no, our_party, counterparty, signer,
            start_date, end_date, total_amount, currency,
            direction, status, extract_source, confidence, reviewed)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            file_path,
            result.get("title", ""),
            # parties 兼容旧列: 拼接双方
            f"{result.get('our_party', '')} / {result.get('counterparty', '')}".strip(" /"),
            raw_text[:20000],
            json.dumps(result, ensure_ascii=False),
            result.get("contract_no", ""),
            result.get("our_party", ""),
            result.get("counterparty", ""),
            result.get("signer", ""),
            result.get("start_date", ""),
            result.get("end_date", ""),
            result.get("total_amount"),
            result.get("currency", "CNY"),
            result.get("direction", "receivable"),
            "active",
            result.get("_method", "llm"),
            result.get("confidence", 0.0),
            0,  # reviewed = False
        ),
    )
    return int(cur.lastrowid)


def save_receivables(conn: sqlite3.Connection, contract_id: int,
                     payments: list[dict]) -> int:
    """批量写入应收条目 (receivables 表), 返回写入数量。"""
    rows = [
        (
            contract_id,
            p.get("due_date") or None,
            p.get("amount"),
            p.get("currency", "CNY"),
            p.get("condition_text", ""),
            p.get("penalty", ""),
            p.get("direction", "receivable"),
            p.get("status", "pending"),
            p.get("method", "regex"),
        )
        for p in payments
    ]
    conn.executemany(
        """INSERT INTO receivables
           (contract_id, due_date, amount, currency, condition_text,
            penalty, direction, status, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)


def add_collection(conn: sqlite3.Connection, contract_id: int, amount: float,
                   receivable_id: Optional[int] = None, currency: str = "CNY",
                   method: str = "", note: str = "") -> int:
    """登记一笔实收, 返回 collection_id。"""
    cur = conn.execute(
        """INSERT INTO collections
           (collected_at, contract_id, receivable_id, amount, currency, method, note)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (date.today().isoformat(), contract_id, receivable_id,
         amount, currency, method, note),
    )
    conn.commit()
    return int(cur.lastrowid)


# ─── 查询 ────────────────────────────────────────────────────
def list_contracts(conn: sqlite3.Connection) -> list[dict]:
    """列出所有合同, 附带应收合计/实收合计/未收余额。"""
    sql = """
        SELECT c.*,
               COALESCE(r_sum.total, 0)  AS receivable_sum,
               COALESCE(col_sum.total, 0) AS collected_sum
        FROM contracts c
        LEFT JOIN (
            SELECT contract_id, SUM(amount) AS total
            FROM receivables GROUP BY contract_id
        ) r_sum ON r_sum.contract_id = c.id
        LEFT JOIN (
            SELECT contract_id, SUM(amount) AS total
            FROM collections GROUP BY contract_id
        ) col_sum ON col_sum.contract_id = c.id
        ORDER BY c.id DESC
    """
    rows = []
    for r in conn.execute(sql).fetchall():
        d = dict(r)
        total = d.get("total_amount") or 0
        recv = d.get("receivable_sum") or 0
        base = max(total, recv)
        d["outstanding"] = round(base - (d.get("collected_sum") or 0), 2)
        rows.append(d)
    return rows


def list_receivables(conn: sqlite3.Connection) -> list[dict]:
    """列出所有应收条目 (含合同标题), 按到期日升序。"""
    sql = """
        SELECT r.id, c.title AS contract_title, c.file_path, c.signer,
               r.due_date, r.amount, r.currency, r.condition_text,
               r.direction, r.status, r.source,
               r.reminded_7d, r.reminded_3d, r.reminded_1d, r.reminded_overdue
        FROM receivables r LEFT JOIN contracts c ON r.contract_id = c.id
        ORDER BY (r.due_date IS NULL), r.due_date ASC
    """
    return [dict(r) for r in conn.execute(sql).fetchall()]


# ─── 提醒 ────────────────────────────────────────────────────
def check_reminders(conn: sqlite3.Connection, today: Optional[date] = None,
                    do_notify: bool = True,
                    config: Optional[dict] = None) -> list[dict]:
    """扫描 pending 应收, 触发 逾期/7/3/1 天四级提醒。

    Args:
        config: 完整配置; 提供时启用 IM 通知 (飞书/企微 Webhook @签单人)

    Returns:
        触发的提醒列表 [{"id","title","due_date","days_left","level","message"}, ...]
    """
    today = today or date.today()
    fired: list[dict] = []

    # 仅对已复核合同 (reviewed=1) 的应收触发提醒
    sql = """
        SELECT r.id, c.title AS contract_title, c.file_path, c.signer,
               r.due_date, r.amount, r.currency, r.condition_text,
               r.direction, r.status, r.source,
               r.reminded_7d, r.reminded_3d, r.reminded_1d, r.reminded_overdue
        FROM receivables r
        JOIN contracts c ON r.contract_id = c.id
        WHERE c.reviewed = 1
        ORDER BY (r.due_date IS NULL), r.due_date ASC
    """
    rows = [dict(r) for r in conn.execute(sql).fetchall()]

    for row in rows:
        if row["status"] != "pending" or not row["due_date"]:
            continue
        try:
            due = datetime.strptime(row["due_date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        days_left = (due - today).days

        level, flag_col = _level(days_left)
        if level is None:
            continue
        if row.get(flag_col):
            continue  # 该级别已提醒过

        title = row["contract_title"] or Path(row["file_path"] or "").name or "合同"
        signer = row.get("signer") or ""
        if days_left < 0:
            msg = (f"{title} · {row['amount']} {row['currency']} · "
                   f"已逾期 {-days_left} 天 (原定 {row['due_date']})")
        else:
            msg = (f"{title} · {row['amount']} {row['currency']} · "
                   f"{days_left}天后到期 ({row['due_date']})")
        if signer:
            msg += f" · 签单人: {signer}"

        if do_notify:
            _send_reminder(conn, config, signer, f"[{level}] {msg}")

        conn.execute(f"UPDATE receivables SET {flag_col}=1 WHERE id=?", (row["id"],))
        fired.append({
            "id": row["id"], "title": title, "due_date": row["due_date"],
            "days_left": days_left, "level": level, "message": msg,
        })

    if fired:
        conn.commit()
    return fired


def _send_reminder(conn: sqlite3.Connection, config: Optional[dict],
                   signer: str, message: str) -> None:
    """发送提醒: 有 config 时走 IM (飞书/企微), 否则桌面通知。

    业务规则: 签单人缺失或映射表查不到 → 降级通知老板指定的默认联系人。
    """
    if not config:
        notify("回款提醒", message)
        return

    from core.notifier import notify_signer
    ncfg = config.get("notify", {}) or {}
    feishu_id, wecom_id, target_name = "", "", signer

    if signer:
        smap = get_signer_by_name(conn, signer)
        if smap:
            feishu_id = smap.get("feishu_id", "")
            wecom_id = smap.get("wecom_id", "")
        else:
            # 映射表未收录 → 降级默认联系人
            target_name = ""

    if not signer or (not feishu_id and not wecom_id):
        # 签单人缺失 或 无 IM 账号 → 用默认联系人
        dc = ncfg.get("default_contact", {}) or {}
        if dc.get("name"):
            target_name = dc["name"]
            feishu_id = dc.get("feishu_id", "")
            wecom_id = dc.get("wecom_id", "")
            message = f"[默认联系人] {message}"

    notify_signer(config, target_name or "未指定", message,
                  feishu_id=feishu_id, wecom_id=wecom_id)


def _level(days_left: int) -> tuple[Optional[str], str]:
    if days_left < 0:
        return "逾期", "reminded_overdue"
    if days_left <= 1:
        return "1天", "reminded_1d"
    if days_left <= 3:
        return "3天", "reminded_3d"
    if days_left <= 7:
        return "7天", "reminded_7d"
    return None, ""


# ─── 错误日志 ────────────────────────────────────────────────
def log_error(conn: sqlite3.Connection, stage: str, error_code: str,
              message: str, file_path: str = "", field: str = "",
              context: str = "", suggestion: str = "") -> int:
    """写入结构化错误日志, 返回 error_id。"""
    cur = conn.execute(
        """INSERT INTO error_log
           (stage, error_code, file_path, field, message, context, suggestion)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (stage, error_code, file_path, field, message, context[:2000], suggestion),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_errors(conn: sqlite3.Connection, limit: int = 50,
                stage: str = "", error_code: str = "",
                file_path: str = "") -> list[dict]:
    """按时间倒序列出最近错误, 支持按阶段/错误码/文件筛选。"""
    clauses, params = [], []
    if stage:
        clauses.append("stage = ?")
        params.append(stage)
    if error_code:
        clauses.append("error_code LIKE ?")
        params.append(f"%{error_code}%")
    if file_path:
        clauses.append("file_path LIKE ?")
        params.append(f"%{file_path}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM error_log {where} ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


# ─── 人工复核门控 ────────────────────────────────────────────
def list_pending_review(conn: sqlite3.Connection) -> list[dict]:
    """列出所有待复核合同 (reviewed=0), 按置信度升序 (低的排前面)。"""
    sql = """SELECT id, file_path, contract_no, title, our_party, counterparty,
                    signer, total_amount, currency, direction,
                    confidence, extract_source, created_at
             FROM contracts WHERE reviewed = 0
             ORDER BY confidence ASC, id DESC"""
    return [dict(r) for r in conn.execute(sql).fetchall()]


def get_contract_detail(conn: sqlite3.Connection, contract_id: int) -> Optional[dict]:
    """获取单份合同完整信息 (含原文 + 结构化 JSON + 应收条目)。"""
    row = conn.execute(
        "SELECT * FROM contracts WHERE id = ?", (contract_id,)
    ).fetchone()
    if not row:
        return None
    detail = dict(row)
    detail["receivables"] = [
        dict(r) for r in conn.execute(
            "SELECT * FROM receivables WHERE contract_id = ?", (contract_id,)
        ).fetchall()
    ]
    return detail


def update_contract_fields(conn: sqlite3.Connection, contract_id: int,
                           fields: dict) -> None:
    """人工修正合同字段 (仅允许白名单列)。"""
    allowed = {"contract_no", "title", "our_party", "counterparty", "signer",
               "start_date", "end_date", "total_amount", "currency", "direction"}
    sets, params = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            params.append(v)
    if not sets:
        return
    params.append(contract_id)
    conn.execute(f"UPDATE contracts SET {', '.join(sets)} WHERE id = ?", params)
    conn.commit()


def update_receivable_fields(conn: sqlite3.Connection, receivable_id: int,
                             fields: dict) -> None:
    """人工修正应收条目字段。"""
    allowed = {"due_date", "amount", "currency", "condition_text", "penalty",
               "direction", "status"}
    sets, params = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            params.append(v)
    if not sets:
        return
    params.append(receivable_id)
    conn.execute(f"UPDATE receivables SET {', '.join(sets)} WHERE id = ?", params)
    conn.commit()


def mark_reviewed(conn: sqlite3.Connection, contract_id: int) -> None:
    """确认复核通过, 合同正式进入回款日程。"""
    conn.execute("UPDATE contracts SET reviewed = 1 WHERE id = ?", (contract_id,))
    conn.commit()


def reject_contract(conn: sqlite3.Connection, contract_id: int,
                    reason: str = "") -> None:
    """驳回合同 (标记为 terminated + 记录原因)。"""
    conn.execute(
        "UPDATE contracts SET reviewed = 1, status = 'terminated' WHERE id = ?",
        (contract_id,),
    )
    if reason:
        log_error(conn, "review", "REJECTED", reason,
                  file_path=_get_file_path(conn, contract_id))
    conn.commit()


def _get_file_path(conn: sqlite3.Connection, contract_id: int) -> str:
    row = conn.execute(
        "SELECT file_path FROM contracts WHERE id = ?", (contract_id,)
    ).fetchone()
    return row[0] if row else ""


# ─── 签单人映射 ──────────────────────────────────────────────
def upsert_signer(conn: sqlite3.Connection, name: str,
                  feishu_id: str = "", wecom_id: str = "",
                  phone: str = "", note: str = "") -> None:
    """新增或更新签单人映射 (name 为唯一键)。"""
    conn.execute(
        """INSERT INTO signer_map (name, feishu_id, wecom_id, phone, note)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET
             feishu_id=excluded.feishu_id, wecom_id=excluded.wecom_id,
             phone=excluded.phone, note=excluded.note""",
        (name, feishu_id, wecom_id, phone, note),
    )
    conn.commit()


def list_signers(conn: sqlite3.Connection) -> list[dict]:
    """列出所有签单人映射。"""
    return [dict(r) for r in conn.execute(
        "SELECT * FROM signer_map ORDER BY name").fetchall()]


def get_signer_by_name(conn: sqlite3.Connection, name: str) -> Optional[dict]:
    """按人名查询签单人映射。"""
    row = conn.execute(
        "SELECT * FROM signer_map WHERE name = ?", (name,)
    ).fetchone()
    return dict(row) if row else None


def delete_signer(conn: sqlite3.Connection, name: str) -> None:
    """删除签单人映射。"""
    conn.execute("DELETE FROM signer_map WHERE name = ?", (name,))
    conn.commit()


def outstanding_by_signer(conn: sqlite3.Connection) -> list[dict]:
    """按签单人聚合未收金额 (仅已复核 + active 合同)。

    Returns:
        [{"signer", "contract_count", "total_receivable",
          "total_collected", "total_outstanding"}, ...]
    """
    sql = """
        SELECT c.signer,
               COUNT(DISTINCT c.id) AS contract_count,
               COALESCE(SUM(r.amount), 0) AS total_receivable,
               COALESCE((SELECT SUM(col.amount) FROM collections col
                         WHERE col.contract_id IN
                           (SELECT id FROM contracts WHERE signer = c.signer)), 0)
                 AS total_collected
        FROM contracts c
        LEFT JOIN receivables r ON r.contract_id = c.id
        WHERE c.reviewed = 1 AND c.status = 'active' AND c.signer != ''
        GROUP BY c.signer
        ORDER BY total_outstanding DESC
    """
    # SQLite 不支持在 ORDER BY 引用别名内的表达式, 用子查询
    sql = """
        SELECT signer, contract_count, total_receivable, total_collected,
               (total_receivable - total_collected) AS total_outstanding
        FROM (
            SELECT c.signer,
                   COUNT(DISTINCT c.id) AS contract_count,
                   COALESCE(SUM(r.amount), 0) AS total_receivable,
                   COALESCE((SELECT SUM(col.amount) FROM collections col
                             WHERE col.contract_id IN
                               (SELECT id FROM contracts c2
                                WHERE c2.signer = c.signer AND c2.reviewed=1)), 0)
                     AS total_collected
            FROM contracts c
            LEFT JOIN receivables r ON r.contract_id = c.id
            WHERE c.reviewed = 1 AND c.status = 'active' AND c.signer != ''
            GROUP BY c.signer
        )
        ORDER BY total_outstanding DESC
    """
    return [dict(r) for r in conn.execute(sql).fetchall()]


# ─── 风险预警 ────────────────────────────────────────────────
def save_risk_alerts(conn: sqlite3.Connection, contract_id: int,
                     alerts: list[dict]) -> int:
    """批量写入风险预警, 返回写入数量。"""
    rows = [
        (contract_id, a.get("level", "yellow"), a.get("rule", ""),
         a.get("message", ""), a.get("evidence", ""))
        for a in alerts
    ]
    conn.executemany(
        "INSERT INTO risk_alert (contract_id, level, rule, message, evidence) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def list_risk_alerts(conn: sqlite3.Connection,
                     contract_id: Optional[int] = None) -> list[dict]:
    """列出风险预警 (可按合同筛选), 按 red 优先 + 时间倒序。"""
    if contract_id:
        sql = """SELECT ra.*, c.title, c.file_path
                 FROM risk_alert ra LEFT JOIN contracts c ON ra.contract_id = c.id
                 WHERE ra.contract_id = ?
                 ORDER BY (ra.level != 'red'), ra.id DESC"""
        rows = conn.execute(sql, (contract_id,)).fetchall()
    else:
        sql = """SELECT ra.*, c.title, c.file_path
                 FROM risk_alert ra LEFT JOIN contracts c ON ra.contract_id = c.id
                 ORDER BY (ra.level != 'red'), ra.id DESC LIMIT 200"""
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def list_contracts_with_risks(conn: sqlite3.Connection) -> list[dict]:
    """列出有风险预警的合同 (去重), red 优先。"""
    sql = """
        SELECT c.id, c.title, c.file_path, c.confidence, c.reviewed,
               COUNT(ra.id) AS risk_count,
               SUM(CASE WHEN ra.level='red' THEN 1 ELSE 0 END) AS red_count
        FROM contracts c
        JOIN risk_alert ra ON ra.contract_id = c.id
        GROUP BY c.id
        ORDER BY red_count DESC, risk_count DESC
    """
    return [dict(r) for r in conn.execute(sql).fetchall()]


# ─── 向后兼容别名 (Phase 2 调用方) ──────────────────────────
def save_payments(conn: sqlite3.Connection, contract_id: int,
                  payments: list[dict]) -> int:
    """[兼容] 等价于 save_receivables。"""
    return save_receivables(conn, contract_id, payments)


def list_payments(conn: sqlite3.Connection) -> list[dict]:
    """[兼容] 等价于 list_receivables。"""
    return list_receivables(conn)
