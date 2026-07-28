"""合同付款条款提取器 (Phase 2 核心)

管线: 文档文本 → LLM 结构化抽取 → 规则兜底解析 → 归一化付款计划。

输出付款条目结构:
    {
        "due_date": "YYYY-MM-DD" | "",     # 绝对日期 (能推断时)
        "amount": float | None,
        "currency": "CNY" | "USD" | ...,
        "condition_text": str,             # 付款条件原文, e.g. "验收后30天"
        "penalty": str,                    # 违约/滞纳金条款
        "status": "pending",
        "method": "llm" | "regex",         # 抽取来源
    }
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any, Optional

# ─── 抽取 Prompt ─────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "你是合同审阅专家, 擅长从中文商务合同中抽取付款条款。"
    "严格只输出 JSON, 不输出任何解释文字。"
)

_USER_TEMPLATE = """请从下面的合同文本中抽取所有付款条款, 以及合同标题与签约方。

要求:
1. 输出 JSON, 结构如下:
{{
  "title": "合同标题(没有则空字符串)",
  "parties": "甲方/乙方等签约方概述",
  "payments": [
    {{
      "due_date": "YYYY-MM-DD 或空字符串(无法确定绝对日期时留空)",
      "amount": 数字或null,
      "currency": "CNY/USD等, 默认CNY",
      "condition_text": "付款条件原文, 如'验收合格后30日内'",
      "penalty": "违约金/滞纳金条款原文, 没有则空字符串"
    }}
  ]
}}
2. 今天日期是 {today}。若条款是相对日期(如'签订后30天'), 且能确定基准日, 请推算出绝对 due_date; 否则 due_date 留空, 把相对描述放进 condition_text。
3. 金额去掉千分位逗号, 只保留数字。'万元'请换算成元(如 5万 -> 50000)。
4. 没有付款条款时 payments 为空数组。

合同文本:
---
{text}
---
"""


def extract_payments(text: str, llm: Optional[Any] = None,
                     base_date: Optional[date] = None) -> dict:
    """从合同文本抽取付款计划。

    Args:
        text: 合同全文
        llm: 具备 .chat(messages) 的引擎; 为 None 时仅用规则兜底
        base_date: 相对日期推算基准, 默认今天

    Returns:
        {"title": str, "parties": str, "payments": [ {...}, ... ]}
    """
    base_date = base_date or date.today()
    text = (text or "").strip()
    if not text:
        return {"title": "", "parties": "", "payments": []}

    # 1. LLM 抽取
    if llm is not None:
        parsed = _llm_extract(text, llm, base_date)
        if parsed and parsed.get("payments"):
            return parsed

    # 2. 规则兜底
    return _regex_extract(text, base_date)


# ─── LLM 路径 ────────────────────────────────────────────────
def _llm_extract(text: str, llm: Any, base_date: date) -> Optional[dict]:
    # 截断过长文本, 避免超上下文 (保留前 12000 字符)
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
    if not isinstance(data, dict):
        return None

    payments = [_normalize_payment(p, "llm") for p in data.get("payments", [])
                if isinstance(p, dict)]
    payments = [p for p in payments if p]
    return {
        "title": str(data.get("title", "") or ""),
        "parties": str(data.get("parties", "") or ""),
        "payments": payments,
    }


def _parse_json(raw: str) -> Optional[dict]:
    """从模型输出中稳健地提取 JSON (容忍 ```json 包裹、前后缀文字与截断)。"""
    raw = raw.strip()
    # 去掉 markdown 代码块
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    # 直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 截取第一个 { 到最后一个 }
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    # 截断修复: 输出被 token 预算截断时, 补全未闭合括号抢救已生成条目
    if start != -1:
        return _repair_truncated(raw[start:])
    return None


def _repair_truncated(s: str) -> Optional[dict]:
    """修复被截断的 JSON: 丢弃末尾不完整片段, 按括号栈补全闭合符号。"""
    stack: list[str] = []
    in_str = False
    esc = False
    last_valid = 0  # 最近一个处于"安全边界"(对象/数组闭合)的索引

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
            if not stack:
                continue
            stack.pop()
            # 弹栈后若回到顶层对象内或数组内, 记录为可截断的安全边界
            last_valid = i

    if not stack and not in_str:
        # 括号其实平衡, 可能是其它字符问题, 直接尝试
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    # 在安全边界处截断, 丢弃不完整的尾部对象
    candidate = s[:last_valid + 1] if last_valid > 0 else s
    # 去掉尾部悬挂的逗号
    candidate = candidate.rstrip()
    if candidate.endswith(","):
        candidate = candidate[:-1]
    # 按剩余栈补全闭合符号 (后开的先闭)
    # 重新计算 candidate 的栈
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
# 金额必须带明确货币单位或货币前缀, 避免把条款序号 "1." "2." 误判为金额。
_AMOUNT_RE = re.compile(
    r"(?:人民币|RMB|￥|¥)\s*([0-9][0-9,，]*(?:\.[0-9]+)?)\s*(万元|万|元|圆)?"
    r"|([0-9][0-9,，]*(?:\.[0-9]+)?)\s*(万元|万|元|圆)",
)
_REL_DATE_RE = re.compile(
    r"(签订|签署|生效|验收|交付|开票|收到发票|到货|质保期满)?[^0-9]{0,15}?"
    r"(\d{1,3})\s*(?:个)?(?:工作)?(?:日|天)(?:内|日内)?"
)
_ABS_DATE_RE = re.compile(r"(\d{4})\s*[-年/.]\s*(\d{1,2})\s*[-月/.]\s*(\d{1,2})")
# 条款序号前缀, e.g. "1." "2、" "(3)"
_ORDINAL_RE = re.compile(r"^\s*[(（]?\d{1,2}[)）]?[.、．]\s*")
# 违约金/滞纳条款关键词: 命中则该段不作为独立付款条目
_PENALTY_KEYS = ("违约金", "滞纳金", "逾期付款", "罚息", "按未付", "每日按")


def _regex_extract(text: str, base_date: date) -> dict:
    payments: list[dict] = []
    penalty_text = ""
    # 以句子为单位扫描含金额/付款关键词的行
    for seg in re.split(r"[。\n;；]", text):
        seg = seg.strip()
        if not seg:
            continue
        # 单独收集违约条款, 不当作付款
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
            "due_date": due,
            "amount": amt,
            "currency": "CNY",
            "condition_text": cond,
            "penalty": "",
            "status": "pending",
            "method": "regex",
        })

    # 把违约条款挂到每条付款上 (若有)
    if penalty_text:
        for p in payments:
            p["penalty"] = penalty_text

    title_m = re.search(r"(?:合同|协议)[名称]*[:：]?\s*([^\n]{4,40})", text[:500])
    return {
        "title": title_m.group(1).strip() if title_m else "",
        "parties": "",
        "payments": payments,
    }


def _parse_amount(seg: str) -> Optional[float]:
    """抽取段落中的真实金额。

    一段可能同时出现序号(1./2.)与真实金额, 取所有候选中的最大值,
    因为真实付款金额通常远大于条款序号。
    """
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


# ─── 归一化 ──────────────────────────────────────────────────
def _normalize_payment(p: dict, method: str) -> Optional[dict]:
    amount = p.get("amount")
    try:
        amount = round(float(amount), 2) if amount not in (None, "") else None
    except (TypeError, ValueError):
        amount = None

    due = str(p.get("due_date", "") or "").strip()
    # 校验日期格式
    if due and not re.match(r"^\d{4}-\d{2}-\d{2}$", due):
        due = ""

    if amount is None and not p.get("condition_text"):
        return None

    return {
        "due_date": due,
        "amount": amount,
        "currency": str(p.get("currency", "CNY") or "CNY"),
        "condition_text": str(p.get("condition_text", "") or "")[:200],
        "penalty": str(p.get("penalty", "") or "")[:200],
        "status": "pending",
        "method": method,
    }
