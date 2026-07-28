"""合同风险规则引擎 (Phase 3D)

抽取完成后自动扫描, 命中规则写入 risk_alert 表。
规则分级:
  red    - 高风险: 可能导致经济损失或法律纠纷
  yellow - 中风险: 信息缺失或不规范, 需人工关注

设计原则:
- 纯规则, 不依赖 LLM (快、稳、可解释)
- 每条规则独立, 易扩展
- 返回 evidence (原文定位), 方便人工复核时跳转
"""
from __future__ import annotations

from typing import Optional


def scan_risks(result: dict, raw_text: str = "") -> list[dict]:
    """对抽取结果执行全部风险规则扫描。

    Args:
        result: extract_contract() 的返回值
        raw_text: 合同原文 (用于 evidence 定位)

    Returns:
        [{"level": "red"|"yellow", "rule": str, "message": str, "evidence": str}, ...]
    """
    alerts: list[dict] = []
    for rule_fn in _RULES:
        hit = rule_fn(result, raw_text)
        if hit:
            alerts.append(hit)
    return alerts


# ─── 规则实现 ────────────────────────────────────────────────

def _rule_amount_mismatch(result: dict, raw_text: str) -> Optional[dict]:
    """条款合计 ≠ 合同总额 (超出 1% 容差)。"""
    total = result.get("total_amount")
    payments = result.get("payments", [])
    if not total or not payments:
        return None
    pay_sum = sum(p.get("amount") or 0 for p in payments)
    if pay_sum <= 0:
        return None
    if abs(pay_sum - total) / total > 0.01:
        return {
            "level": "red",
            "rule": "amount_mismatch",
            "message": f"金额矛盾: 条款合计 {pay_sum:,.0f} ≠ 合同总额 {total:,.0f}",
            "evidence": f"合同总额={total}, 条款合计={pay_sum}",
        }
    return None


def _rule_date_contradiction(result: dict, raw_text: str) -> Optional[dict]:
    """终止日期 < 起始日期。"""
    start = result.get("start_date", "")
    end = result.get("end_date", "")
    if start and end and end < start:
        return {
            "level": "red",
            "rule": "date_contradiction",
            "message": f"日期矛盾: 终止日 {end} 早于起始日 {start}",
            "evidence": f"start_date={start}, end_date={end}",
        }
    return None


def _rule_missing_penalty(result: dict, raw_text: str) -> Optional[dict]:
    """所有应收条目均无违约条款。"""
    payments = result.get("payments", [])
    if not payments:
        return None
    has_penalty = any(p.get("penalty", "").strip() for p in payments)
    if not has_penalty:
        # 尝试从原文找证据
        evidence = ""
        for kw in ("违约", "滞纳金", "罚则", "逾期"):
            idx = raw_text.find(kw)
            if idx >= 0:
                evidence = raw_text[max(0, idx - 10):idx + 40]
                break
        if not evidence:
            evidence = "全文未找到违约/滞纳金相关条款"
        return {
            "level": "yellow",
            "rule": "missing_penalty",
            "message": "缺少违约条款: 所有收付款条目均无违约金/滞纳金约定",
            "evidence": evidence,
        }
    return None


def _rule_missing_signer(result: dict, raw_text: str) -> Optional[dict]:
    """未抽取到签单人/负责人。"""
    signer = (result.get("signer") or "").strip()
    if not signer:
        return {
            "level": "yellow",
            "rule": "missing_signer",
            "message": "未识别到签单人/负责人, 提醒将降级发送给默认联系人",
            "evidence": "signer 字段为空",
        }
    return None


def _rule_party_unidentified(result: dict, raw_text: str) -> Optional[dict]:
    """我方主体未识别 (company 未配置或合同中未匹配到)。"""
    our = (result.get("our_party") or "").strip()
    if not our:
        return {
            "level": "yellow",
            "rule": "party_unidentified",
            "message": "我方主体未识别: 请检查 config.yaml company.name 配置",
            "evidence": "our_party 字段为空",
        }
    return None


def _rule_missing_amount(result: dict, raw_text: str) -> Optional[dict]:
    """合同总额和条款金额均为空。"""
    total = result.get("total_amount")
    payments = result.get("payments", [])
    has_pay_amount = any(p.get("amount") for p in payments)
    if not total and not has_pay_amount:
        return {
            "level": "red",
            "rule": "missing_amount",
            "message": "未抽取到任何金额: 合同总额和条款金额均为空",
            "evidence": "total_amount=null, payments 无有效金额",
        }
    return None


def _rule_no_due_date(result: dict, raw_text: str) -> Optional[dict]:
    """有条款但全部无绝对到期日 (回款日程无法排列)。"""
    payments = result.get("payments", [])
    if not payments:
        return None
    has_date = any(p.get("due_date", "").strip() for p in payments)
    if not has_date:
        return {
            "level": "yellow",
            "rule": "no_due_date",
            "message": "所有条款均无绝对到期日, 回款日程无法自动排列, 需人工补充",
            "evidence": "所有 payments[].due_date 为空",
        }
    return None


# ─── 规则注册表 ──────────────────────────────────────────────
_RULES = [
    _rule_amount_mismatch,
    _rule_date_contradiction,
    _rule_missing_penalty,
    _rule_missing_signer,
    _rule_party_unidentified,
    _rule_missing_amount,
    _rule_no_due_date,
]
