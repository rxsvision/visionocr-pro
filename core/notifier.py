"""IM 通知模块 - 飞书 + 企业微信 Webhook (Phase 3C)

设计原则:
- 最简集成: 群机器人 Webhook (无需 OAuth, 无需审批)
- @mention: 飞书用 open_id, 企微用 userid
- 优雅降级: 未配置时回退桌面通知, 不阻断主流程
- 超时短 (5s), 失败只记日志不抛异常
"""
from __future__ import annotations

import json
from typing import Optional

import requests

_TIMEOUT = 5  # 秒


def send_feishu(webhook_url: str, text: str,
                mention_ids: Optional[list[str]] = None) -> bool:
    """发送飞书群机器人消息。

    Args:
        webhook_url: https://open.feishu.cn/open-apis/bot/v2/hook/{token}
        text: 消息正文
        mention_ids: 需要 @mention 的 open_id 列表

    Returns:
        是否发送成功
    """
    if not webhook_url:
        return False

    # 飞书富文本支持 @mention
    if mention_ids:
        # 使用 post 类型支持 at 标签
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
        print(f"[Notifier] 飞书发送失败: {e}")
        return False


def send_wecom(webhook_url: str, text: str,
               mention_ids: Optional[list[str]] = None) -> bool:
    """发送企业微信群机器人消息。

    Args:
        webhook_url: https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}
        text: 消息正文
        mention_ids: 需要 @mention 的 userid 列表

    Returns:
        是否发送成功
    """
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
        print(f"[Notifier] 企微发送失败: {e}")
        return False


def notify_signer(config: dict, signer_name: str, message: str,
                  feishu_id: str = "", wecom_id: str = "") -> dict:
    """根据配置向签单人发送提醒, 返回各渠道发送结果。

    Args:
        config: 完整配置 (含 notify 段)
        signer_name: 签单人名 (用于日志)
        message: 消息正文
        feishu_id: 签单人飞书 open_id (可空)
        wecom_id: 签单人企微 userid (可空)

    Returns:
        {"feishu": bool, "wecom": bool, "desktop": bool}
    """
    ncfg = config.get("notify", {}) or {}
    results = {"feishu": False, "wecom": False, "desktop": False}

    # 飞书
    feishu_url = ncfg.get("feishu_webhook", "")
    if feishu_url:
        mentions = [feishu_id] if feishu_id else None
        results["feishu"] = send_feishu(feishu_url, message, mentions)

    # 企微
    wecom_url = ncfg.get("wecom_webhook", "")
    if wecom_url:
        mentions = [wecom_id] if wecom_id else None
        results["wecom"] = send_wecom(wecom_url, message, mentions)

    # 桌面通知兜底 (IM 全未配置或全失败时)
    if not any(results.values()):
        try:
            from core.notifications import notify
            notify(f"回款提醒 · {signer_name}", message)
            results["desktop"] = True
        except Exception:  # noqa: BLE001
            pass

    return results
