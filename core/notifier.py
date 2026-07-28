"""IM 通知模块 - 飞书 + 企业微信 Webhook (Phase 3C + Phase 5 重试)

设计原则:
- 最简集成: 群机器人 Webhook (无需 OAuth, 无需审批)
- @mention: 飞书用 open_id, 企微用 userid
- 优雅降级: 未配置时回退桌面通知, 不阻断主流程
- 超时短 (5s), 失败记日志 + 指数退避重试 (最多3次)
- 所有发送记录落库 notification_log, 支持审计和失败重发
"""
from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger("visionocr.notifier")

_TIMEOUT = 5  # 秒
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = [0, 30, 120]  # 第1/2/3次尝试前的等待


# ─── 底层发送 ────────────────────────────────────────────────
def send_feishu(webhook_url: str, text: str,
                mention_ids: Optional[list[str]] = None) -> bool:
    """发送飞书群机器人消息。"""
    if not webhook_url:
        return False

    if mention_ids:
        content_lines = [[{"tag": "text", "text": text}]]
        for uid in mention_ids:
            content_lines[0].append({"tag": "at", "user_id": uid})
        payload = {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": "回款提醒",
                        "content": content_lines,
                    }
                }
            },
        }
    else:
        payload = {"msg_type": "text", "content": {"text": text}}

    try:
        r = requests.post(webhook_url, json=payload, timeout=_TIMEOUT)
        data = r.json()
        return data.get("code", -1) == 0 or data.get("StatusCode", -1) == 0
    except Exception as e:  # noqa: BLE001
        logger.warning("飞书发送失败: %s", e)
        return False


def send_wecom(webhook_url: str, text: str,
               mention_ids: Optional[list[str]] = None) -> bool:
    """发送企业微信群机器人消息。"""
    if not webhook_url:
        return False

    payload: dict = {
        "msgtype": "text",
        "text": {"content": text},
    }
    if mention_ids:
        payload["text"]["mentioned_list"] = mention_ids

    try:
        r = requests.post(webhook_url, json=payload, timeout=_TIMEOUT)
        data = r.json()
        return data.get("errcode", -1) == 0
    except Exception as e:  # noqa: BLE001
        logger.warning("企微发送失败: %s", e)
        return False


# ─── 通知日志 ────────────────────────────────────────────────
def log_notification(conn: sqlite3.Connection, channel: str, recipient: str,
                     message: str, success: bool, attempts: int = 1,
                     error: str = "", receivable_id: Optional[int] = None,
                     contract_id: Optional[int] = None) -> int:
    """写入通知日志, 返回 log_id。"""
    next_retry = ""
    if not success and attempts < _MAX_ATTEMPTS:
        delay = _BACKOFF_SECONDS[min(attempts, len(_BACKOFF_SECONDS) - 1)]
        next_retry = (datetime.now() + timedelta(seconds=delay)).isoformat(
            timespec="seconds")
    cur = conn.execute(
        """INSERT INTO notification_log
           (channel, recipient, message, success, attempts, max_attempts,
            next_retry_at, error, receivable_id, contract_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (channel, recipient, message[:500], int(success), attempts,
         _MAX_ATTEMPTS, next_retry, error[:300], receivable_id, contract_id),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_notification_logs(conn: sqlite3.Connection, limit: int = 100,
                           channel: str = "", success_only: bool = False,
                           failed_only: bool = False) -> list[dict]:
    """查询通知日志。"""
    clauses, params = [], []
    if channel:
        clauses.append("channel = ?")
        params.append(channel)
    if success_only:
        clauses.append("success = 1")
    if failed_only:
        clauses.append("success = 0")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM notification_log {where} ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


# ─── 重试失败通知 ────────────────────────────────────────────
def retry_failed_notifications(conn: sqlite3.Connection,
                               config: dict) -> int:
    """重试所有待重试的失败通知 (由调度器定期调用)。

    Returns: 成功重发的数量。
    """
    now = datetime.now().isoformat(timespec="seconds")
    rows = conn.execute(
        """SELECT * FROM notification_log
           WHERE success = 0 AND attempts < max_attempts
             AND (next_retry_at = '' OR next_retry_at <= ?)
           ORDER BY id ASC LIMIT 50""",
        (now,),
    ).fetchall()

    ncfg = config.get("notify", {}) or {}
    retried_ok = 0

    for row in rows:
        row = dict(row)
        channel = row["channel"]
        msg = row["message"]
        recipient = row["recipient"]
        attempts = row["attempts"] + 1

        ok = False
        if channel == "feishu":
            url = ncfg.get("feishu_webhook", "")
            ok = send_feishu(url, msg)
        elif channel == "wecom":
            url = ncfg.get("wecom_webhook", "")
            ok = send_wecom(url, msg)
        elif channel == "desktop":
            try:
                from core.notifications import notify
                notify(f"回款提醒 · {recipient}", msg)
                ok = True
            except Exception:  # noqa: BLE001
                ok = False

        if ok:
            conn.execute(
                "UPDATE notification_log SET success=1, attempts=? WHERE id=?",
                (attempts, row["id"]),
            )
            retried_ok += 1
        else:
            next_retry = ""
            if attempts < _MAX_ATTEMPTS:
                delay = _BACKOFF_SECONDS[min(attempts, len(_BACKOFF_SECONDS) - 1)]
                next_retry = (datetime.now() + timedelta(seconds=delay)).isoformat(
                    timespec="seconds")
            conn.execute(
                "UPDATE notification_log SET attempts=?, next_retry_at=? WHERE id=?",
                (attempts, next_retry, row["id"]),
            )

    if rows:
        conn.commit()
        logger.info("通知重试: %d 条待重试, %d 条成功", len(rows), retried_ok)
    return retried_ok


# ─── 主入口 (带重试) ────────────────────────────────────────
def notify_signer(config: dict, signer_name: str, message: str,
                  feishu_id: str = "", wecom_id: str = "",
                  conn: Optional[sqlite3.Connection] = None,
                  receivable_id: Optional[int] = None,
                  contract_id: Optional[int] = None) -> dict:
    """根据配置向签单人发送提醒, 返回各渠道发送结果。

    带重试: 首次失败后等待 30s/120s 再试 (最多3次)。
    带日志: 所有尝试写入 notification_log (需传入 conn)。

    Returns:
        {"feishu": bool, "wecom": bool, "desktop": bool}
    """
    ncfg = config.get("notify", {}) or {}
    results = {"feishu": False, "wecom": False, "desktop": False}

    # 飞书
    feishu_url = ncfg.get("feishu_webhook", "")
    if feishu_url:
        mentions = [feishu_id] if feishu_id else None
        results["feishu"] = _send_with_retry(
            lambda: send_feishu(feishu_url, message, mentions),
            "feishu", signer_name, message, conn,
            receivable_id, contract_id)

    # 企微
    wecom_url = ncfg.get("wecom_webhook", "")
    if wecom_url:
        mentions = [wecom_id] if wecom_id else None
        results["wecom"] = _send_with_retry(
            lambda: send_wecom(wecom_url, message, mentions),
            "wecom", signer_name, message, conn,
            receivable_id, contract_id)

    # 桌面通知兜底
    if not any(results.values()) and ncfg.get("desktop_fallback", True):
        try:
            from core.notifications import notify
            notify(f"回款提醒 · {signer_name}", message)
            results["desktop"] = True
            if conn:
                log_notification(conn, "desktop", signer_name, message, True,
                                 receivable_id=receivable_id,
                                 contract_id=contract_id)
        except Exception:  # noqa: BLE001
            if conn:
                log_notification(conn, "desktop", signer_name, message, False,
                                 error="桌面通知异常",
                                 receivable_id=receivable_id,
                                 contract_id=contract_id)

    return results


def _send_with_retry(send_fn, channel: str, recipient: str, message: str,
                     conn: Optional[sqlite3.Connection],
                     receivable_id: Optional[int] = None,
                     contract_id: Optional[int] = None) -> bool:
    """带指数退避重试的发送包装。"""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        if attempt > 1:
            delay = _BACKOFF_SECONDS[min(attempt - 1, len(_BACKOFF_SECONDS) - 1)]
            logger.debug("%s 第%d次重试, 等待%ds...", channel, attempt, delay)
            time.sleep(delay)

        ok = send_fn()
        if ok:
            if conn:
                log_notification(conn, channel, recipient, message, True,
                                 attempts=attempt,
                                 receivable_id=receivable_id,
                                 contract_id=contract_id)
            return True

    # 全部失败
    if conn:
        log_notification(conn, channel, recipient, message, False,
                         attempts=_MAX_ATTEMPTS,
                         error=f"{_MAX_ATTEMPTS}次尝试均失败",
                         receivable_id=receivable_id,
                         contract_id=contract_id)
    return False
