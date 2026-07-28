"""合同应收条款提取器 (Phase 3A 重构)

收款方视角, 管应收回款。管线:
    文档文本 → LLM 结构化抽取 (本地优先/云端兜底) → 规则兜底
             → 方向判定 (基于我方主体) → 金额勾稽校验 → 置信度评估

输出结构 (extract_contract):
    {
      "contract_no": str, "title": str,
      "our_party": str, "counterparty": str, "signer": str,
      "start_date": str, "end_date": str,
      "total_amount": float|None, "currency": str,
      "direction": "receivable"|"payable",
      "payments": [ {due_date, amount, currency, condition_text, penalty,
                     payer, payee, direction, method}, ... ],
      "confidence": float,   # 0~1 整体置信度 (供分级路由/人工复核)
      "valid": bool,         # 金额勾稽/JSON 校验是否通过 (供路由升级判定)
      "warnings": [str, ...] # 校验告警
    }
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any, Optional

# ─── 抽取 Prompt ─────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "你是合同审阅专家, 擅长从中文商务合同中抽取结构化信息。"
    "严格只输出 JSON, 不输出任何解释文字。"
)

_USER_TEMPLATE = """请从下面的合同文本中抽取合同要素与所有收付款条款。

输出 JSON, 结构如下:
{{
  "contract_no": "合同编号(没有则空字符串)",
  "title": "合同标题(没有则空字符串)",
  "party_a": "甲方全称",
  "party_b": "乙方全称",
  "signer": "签单人/负责人/委托代理人姓名(没有则空字符串)",
  "start_date": "YYYY-MM-DD 或空字符串",
  "end_date": "YYYY-MM-DD 或空字符串",
  "total_amount": 合同总金额数字或null,
  "currency": "CNY/USD等, 默认CNY",
  "payments": [
    {{
      "due_date": "YYYY-MM-DD 或空字符串(无法确定绝对日期时留空)",
      "amount": 数字或null,
      "currency": "CNY/USD等, 默认CNY",
      "condition_text": "付款条件原文, 如'验收合格后30日内'",
      "penalty": "违约金/滞纳金条款原文, 没有则空字符串",
      "payer": "付款方名称(谁付钱)",
      "payee": "收款方名称(谁收钱)"
    }}
  ]
}}

规则:
1. 今天日期是 {today}。相对日期(如'签订后30天')若能确定基准日则推算绝对 due_date, 否则留空并把相对描述放进 condition_text。
2. 金额去掉千分位逗号。'万元'换算成元(如 5万 -> 50000)。
3. payer/payee 尽量用合同里的主体名称(甲方/乙方全称)。
4. 没有收付款条款时 payments 为空数组。

合同文本:
---
{text}
---
"""


# ─── 主入口 ──────────────────────────────────────────────────
def extract_contract(text: str, llm: Optional[Any] = None,
                     base_date: Optional[date] = None,
                     company: Optional[dict] = None) -> dict:
    """从合同文本抽取完整合同要素与应收计划。

    Args:
        text: 合同全文
        llm: 具备 .chat(messages) 的引擎; None 时仅规则兜底
        base_date: 相对日期推算基准, 默认今天
        company: {"name": str, "aliases": [str]} 我方主体, 用于方向判定

    Returns:
        见模块 docstring 的结构
    """
    base_date = base_date or date.today()
    company = company or {}
    text = (text or "").strip()
    if not text:
        return _empty_result()

    # 1. LLM 抽取
    data = None
    method = "regex"
    if llm is not None:
        data = _llm_extract(text, llm, base_date)
        if data is not None:
            method = "llm"

    # 2. 规则兜底 (LLM 失败或未提供)
    if data is None:
        data = _regex_extract(text, base_date)
        method = "regex"

    # 3. 方向判定 + 校验 + 置信度
    return _finalize(data, method, company)


def _empty_result() -> dict:
    return {
        "contract_no": "", "title": "", "our_party": "", "counterparty": "",
        "signer": "", "start_date": "", "end_date": "",
        "total_amount": None, "currency": "CNY", "direction": "receivable",
        "payments": [], "confidence": 0.0, "valid": False,
        "warnings": ["空文本"],
    }


# ─── 方向判定 + 校验 + 置信度 ────────────────────────────────
def _finalize(data: dict, method: str, company: dict) -> dict:
    our_names = _company_names(company)

    # 合同级主体
    party_a = str(data.get("party_a", "") or "")
    party_b = str(data.get("party_b", "") or "")
    parties = data.get("parties", "") or ""
    # 兼容旧结构: 只有 parties 字符串时尝试拆分
    if not party_a and not party_b and parties:
        party_a, party_b = _split_parties(parties)

    our_party, counterparty = _identify_our_party(party_a, party_b, parties, our_names)

    # 逐笔款项方向判定
    payments = []
    for p in data.get("payments", []) or []:
        if not isinstance(p, dict):
            continue
        norm = _normalize_payment(p, method)
        if not norm:
            continue
        norm["direction"] = _judge_direction(norm, our_party, our_names)
        payments.append(norm)

    # 合同方向: 以应收条目占比判定
    recv = sum(1 for p in payments if p["direction"] == "receivable")
    direction = "receivable" if recv >= len(payments) / 2 else ("payable" if payments else "receivable")

    total_amount = _to_float(data.get("total_amount"))
    currency = str(data.get("currency", "CNY") or "CNY")

    # 金额勾稽校验
    warnings: list[str] = []
    valid = True
    if total_amount and payments:
        pay_sum = round(sum(p["amount"] for p in payments if p["amount"]), 2)
        # 允许 1% 误差 (含税/四舍五入)
        if pay_sum > 0 and abs(pay_sum - total_amount) / total_amount > 0.01:
            warnings.append(
                f"金额勾稽不符: 条款合计 {pay_sum} ≠ 合同总额 {total_amount}"
            )
            valid = False
    if method == "llm" and not payments and not total_amount:
        warnings.append("LLM 未抽取到任何金额")
        valid = False

    confidence = _assess_confidence(method, payments, total_amount, data, our_party)

    return {
        "contract_no": str(data.get("contract_no", "") or ""),
        "title": str(data.get("title", "") or ""),
        "our_party": our_party,
        "counterparty": counterparty,
        "signer": str(data.get("signer", "") or ""),
        "start_date": _norm_date(data.get("start_date")),
        "end_date": _norm_date(data.get("end_date")),
        "total_amount": total_amount,
        "currency": currency,
        "direction": direction,
        "payments": payments,
        "confidence": confidence,
        "valid": valid,
        "warnings": warnings,
        "_method": method,
    }


def _company_names(company: dict) -> list[str]:
    names = []
    if company.get("name"):
        names.append(str(company["name"]).strip())
    for a in company.get("aliases", []) or []:
        if a:
            names.append(str(a).strip())
    return [n for n in names if n]


def _split_parties(parties: str) -> tuple[str, str]:
    """从 '甲方：X，乙方：Y' 类字符串拆出双方。"""
    a = re.search(r"甲方[:：]?\s*([^，,；;。\n]+)", parties)
    b = re.search(r"乙方[:：]?\s*([^，,；;。\n]+)", parties)
    return (a.group(1).strip() if a else "", b.group(1).strip() if b else "")


def _identify_our_party(party_a: str, party_b: str, parties: str,
                        our_names: list[str]) -> tuple[str, str]:
    """识别我方主体与对方主体。未配置我方名称时返回空。"""
    if not our_names:
        return "", ""
    haystacks = [("a", party_a), ("b", party_b), ("all", parties)]
    for name in our_names:
        for tag, text in haystacks:
            if text and name in text:
                if tag == "a":
                    return party_a, party_b
                if tag == "b":
                    return party_b, party_a
                # 在 parties 混合串里命中, 取较完整的一方
                return name, ""
    return "", ""


def _judge_direction(payment: dict, our_party: str, our_names: list[str]) -> str:
    """判断单笔款项方向: 我方是收款方 → receivable, 付款方 → payable。"""
    payer = str(payment.get("payer", "") or "")
    payee = str(payment.get("payee", "") or "")
    names = our_names + ([our_party] if our_party else [])
    if not names:
        return "receivable"  # 未配置我方主体, 默认按应收 (业务定位为收款方)
    # 收款方含我方 → 应收
    if payee and any(n in payee for n in names):
        return "receivable"
    # 付款方含我方 → 应付
    if payer and any(n in payer for n in names):
        return "payable"
    return "receivable"


def _assess_confidence(method: str, payments: list[dict], total_amount,
                       data: dict, our_party: str) -> float:
    """整体置信度评估 (0~1), 供分级路由与人工复核闸门使用。"""
    if method == "regex":
        base = 0.5  # 规则抽取置信度上限较低
    else:
        base = 0.85
    # 有金额条目加分
    if payments and any(p["amount"] for p in payments):
        base += 0.05
    # 有合同总额加分
    if total_amount:
        base += 0.05
    # 识别出我方主体加分 (方向可信)
    if our_party:
        base += 0.05
    # 相对日期(无绝对 due_date)略降
    if payments and not any(p["due_date"] for p in payments):
        base -= 0.05
    return round(max(0.0, min(1.0, base)), 3)


# ─── LLM 路径 ────────────────────────────────────────────────
def _llm_extract(text: str, llm: Any, base_date: date) -> Optional[dict]:
    snippet = text[:12000]
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",
         "content": _USER_TEMPLATE.format(today=base_date.isoformat(), text=snippet)},
    ]
    try:
        raw = llm.chat(messages, max_tokens=8192)
    except Exception as e:  # noqa: BLE001
        print(f"[Extractor] LLM 调用失败: {e}")
        return None
    if not raw:
        return None
    data = _parse_json(raw)
    return data if isinstance(data, dict) else None


def _parse_json(raw: str) -> Optional[dict]:
    """从模型输出中稳健地提取 JSON (容忍 ```json 包裹、前后缀文字与截断)。"""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    if start != -1:
        return _repair_truncated(raw[start:])
    return None


def _repair_truncated(s: str) -> Optional[dict]:
    """修复被截断的 JSON: 丢弃末尾不完整片段, 按括号栈补全闭合符号。"""
    stack: list[str] = []
    in_str = False
    esc = False
    last_valid = 0
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            last_valid = i
    if not stack and not in_str:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    candidate = s[:last_valid + 1] if last_valid > 0 else s
    candidate = candidate.rstrip()
    if candidate.endswith(","):
        candidate = candidate[:-1]
    stack2: list[str] = []
    in_str2 = False
    esc2 = False
    for ch in candidate:
        if in_str2:
            if esc2:
                esc2 = False
            elif ch == "\\":
                esc2 = True
            elif ch == '"':
                in_str2 = False
            continue
        if ch == '"':
            in_str2 = True
        elif ch in "{[":
            stack2.append(ch)
        elif ch in "}]":
            if stack2:
                stack2.pop()
    closing = "".join("]" if c == "[" else "}" for c in reversed(stack2))
    try:
        return json.loads(candidate + closing)
    except json.JSONDecodeError:
        return None


# ─── 规则兜底路径 ────────────────────────────────────────────
_AMOUNT_RE = re.compile(
    r"(?:人民币|RMB|￥|¥)\s*([0-9][0-9,，]*(?:\.[0-9]+)?)\s*(万元|万|元|圆)?"
    r"|([0-9][0-9,，]*(?:\.[0-9]+)?)\s*(万元|万|元|圆)",
)
_REL_DATE_RE = re.compile(
    r"(签订|签署|生效|验收|交付|开票|收到发票|到货|质保期满)?[^0-9]{0,15}?"
    r"(\d{1,3})\s*(?:个)?(?:工作)?(?:日|天)(?:内|日内)?"
)
_ABS_DATE_RE = re.compile(r"(\d{4})\s*[-年/.]\s*(\d{1,2})\s*[-月/.]\s*(\d{1,2})")
_ORDINAL_RE = re.compile(r"^\s*[(（]?\d{1,2}[)）]?[.、．]\s*")
_PENALTY_KEYS = ("违约金", "滞纳金", "逾期付款", "罚息", "按未付", "每日按")
_TOTAL_KEYS = ("合同总价", "合同金额", "合同总额", "总金额", "总价款", "合同价款")


def _regex_extract(text: str, base_date: date) -> dict:
    payments: list[dict] = []
    penalty_text = ""
    total_amount = None

    for seg in re.split(r"[。\n;；]", text):
        seg = seg.strip()
        if not seg:
            continue
        # 合同总额
        if any(k in seg for k in _TOTAL_KEYS) and total_amount is None:
            total_amount = _parse_amount(seg)
        # 违约条款单独收集
        if any(k in seg for k in _PENALTY_KEYS):
            if not penalty_text:
                penalty_text = seg[:120]
            continue
        if not any(k in seg for k in ("付款", "支付", "结算", "定金", "尾款", "预付款", "进度款", "验收款")):
            continue
        amt = _parse_amount(seg)
        if amt is None:
            continue
        due = ""
        m_abs = _ABS_DATE_RE.search(seg)
        if m_abs:
            due = _safe_date(int(m_abs.group(1)), int(m_abs.group(2)), int(m_abs.group(3)))
        else:
            m_rel = _REL_DATE_RE.search(seg)
            if m_rel:
                due = (base_date + timedelta(days=int(m_rel.group(2)))).isoformat()
        cond = _ORDINAL_RE.sub("", seg)[:120]
        payments.append({
            "due_date": due, "amount": amt, "currency": "CNY",
            "condition_text": cond, "penalty": "", "payer": "", "payee": "",
            "method": "regex",
        })

    if penalty_text:
        for p in payments:
            p["penalty"] = penalty_text

    title_m = re.search(r"(?:合同|协议)[名称]*[:：]?\s*([^\n]{4,40})", text[:500])
    party_a, party_b = _split_parties(text[:800])
    return {
        "contract_no": "",
        "title": title_m.group(1).strip() if title_m else "",
        "party_a": party_a, "party_b": party_b,
        "signer": "",
        "total_amount": total_amount,
        "payments": payments,
    }


def _parse_amount(seg: str) -> Optional[float]:
    candidates: list[float] = []
    for m in _AMOUNT_RE.finditer(seg):
        num_str = m.group(1) or m.group(3)
        unit = m.group(2) or m.group(4) or ""
        if not num_str:
            continue
        try:
            val = float(num_str.replace(",", "").replace("，", ""))
        except ValueError:
            continue
        if "万" in unit:
            val *= 10000
        candidates.append(val)
    if not candidates:
        return None
    return round(max(candidates), 2)


def _safe_date(y: int, mo: int, d: int) -> str:
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return ""


# ─── 归一化工具 ──────────────────────────────────────────────
def _to_float(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return round(float(v), 2)
    except (TypeError, ValueError):
        return None


def _norm_date(v: Any) -> str:
    s = str(v or "").strip()
    return s if re.match(r"^\d{4}-\d{2}-\d{2}$", s) else ""


def _normalize_payment(p: dict, method: str) -> Optional[dict]:
    amount = _to_float(p.get("amount"))
    due = _norm_date(p.get("due_date"))
    if amount is None and not p.get("condition_text"):
        return None
    return {
        "due_date": due,
        "amount": amount,
        "currency": str(p.get("currency", "CNY") or "CNY"),
        "condition_text": str(p.get("condition_text", "") or "")[:200],
        "penalty": str(p.get("penalty", "") or "")[:200],
        "payer": str(p.get("payer", "") or "")[:100],
        "payee": str(p.get("payee", "") or "")[:100],
        "status": "pending",
        "method": method,
    }


# ─── 向后兼容别名 ────────────────────────────────────────────
def extract_payments(text: str, llm: Optional[Any] = None,
                     base_date: Optional[date] = None) -> dict:
    """旧接口兼容: 返回 {title, parties, payments}。"""
    r = extract_contract(text, llm=llm, base_date=base_date)
    return {
        "title": r["title"],
        "parties": " / ".join(x for x in [r["our_party"], r["counterparty"]] if x),
        "payments": r["payments"],
    }
