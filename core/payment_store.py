"""付款计划存储与到期提醒 (Phase 2)

职责:
- 把抽取出的付款条目落库 (contracts + payments 两表)。
- 按 7/3/1 天三级阈值扫描到期项, 触发桌面通知并标记 reminded_*。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from core.notifications import notify


# ─── 写入 ────────────────────────────────────────────────────
def save_contract(conn: sqlite3.Connection, file_path: str, title: str,
                  parties: str, raw_text: str, structured: dict) -> int:
    """写入合同主记录, 返回 contract_id。"""
    cur = conn.execute(
        "INSERT INTO contracts (file_path, title, parties, raw_text, structured_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (file_path, title, parties, raw_text[:20000], json.dumps(structured, ensure_ascii=False)),
    )
    return int(cur.lastrowid)


def save_payments(conn: sqlite3.Connection, contract_id: int, payments: list[dict]) -> int:
    """批量写入付款条目, 返回写入数量。"""
    rows = [
        (
            contract_id,
            p.get("due_date") or None,
            p.get("amount"),
            p.get("currency", "CNY"),
            p.get("condition_text", ""),
            p.get("penalty", ""),
            p.get("status", "pending"),
            p.get("method", "regex"),
        )
        for p in payments
    ]
    conn.executemany(
        "INSERT INTO payments "
        "(contract_id, due_date, amount, currency, condition_text, penalty, status, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


# ─── 查询 ────────────────────────────────────────────────────
def list_payments(conn: sqlite3.Connection) -> list[dict]:
    """列出所有付款条目 (含合同标题), 按到期日升序。"""
    sql = """
        SELECT p.id, c.title AS contract_title, c.file_path,
               p.due_date, p.amount, p.currency, p.condition_text, p.status,
               p.reminded_7d, p.reminded_3d, p.reminded_1d, p.source
        FROM payments p LEFT JOIN contracts c ON p.contract_id = c.id
        ORDER BY (p.due_date IS NULL), p.due_date ASC
    """
    return [dict(r) for r in conn.execute(sql).fetchall()]


# ─── 提醒 ────────────────────────────────────────────────────
def check_reminders(conn: sqlite3.Connection, today: Optional[date] = None,
                    do_notify: bool = True) -> list[dict]:
    """扫描 pending 付款, 触发 7/3/1 天提醒。

    Returns:
        触发的提醒列表 [{"id","title","due_date","days_left","level"}, ...]
    """
    today = today or date.today()
    fired: list[dict] = []

    for row in list_payments(conn):
        if row["status"] != "pending" or not row["due_date"]:
            continue
        try:
            due = datetime.strptime(row["due_date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        days_left = (due - today).days
        if days_left < 0:
            continue  # 已逾期不在本提醒范围 (可另行处理)

        level, flag_col = _level(days_left)
        if level is None:
            continue
        if row.get(flag_col):
            continue  # 该级别已提醒过

        title = row["contract_title"] or Path(row["file_path"] or "").name or "合同"
        msg = f"{title} · {row['amount']} {row['currency']} · {days_left}天后到期 ({row['due_date']})"
        if do_notify:
            notify(f"付款提醒 [{level}]", msg)

        conn.execute(f"UPDATE payments SET {flag_col}=1 WHERE id=?", (row["id"],))
        fired.append({
            "id": row["id"], "title": title, "due_date": row["due_date"],
            "days_left": days_left, "level": level, "message": msg,
        })

    if fired:
        conn.commit()
    return fired


def _level(days_left: int) -> tuple[Optional[str], str]:
    if days_left <= 1:
        return "1天", "reminded_1d"
    if days_left <= 3:
        return "3天", "reminded_3d"
    if days_left <= 7:
        return "7天", "reminded_7d"
    return None, ""
