"""OCR 后处理纠错模块 - 正则规则 + 混淆字符修正

工业 OCR 常见错误模式:
- 字符混淆: 0/O, 1/I/l, 5/S, 8/B, 6/G, 2/Z, 9/q
- 格式违规: 日期码中混入字母, 编号中混入特殊字符
- 多余/缺失字符

设计:
- 规则链式执行, 每条规则独立可开关
- 用户可通过 UI 自定义正则替换规则
- 不改变原始结果, 输出纠错后文本 + 修改日志
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CorrectionRule:
    """单条纠错规则"""
    name: str
    pattern: str          # 正则表达式
    replacement: str      # 替换字符串 (支持 \1 \2 反向引用)
    enabled: bool = True
    description: str = ""


# ─── 内置规则集 (工业标记常见) ────────────────────────────────
BUILTIN_RULES: list[CorrectionRule] = [
    CorrectionRule(
        name="date_code_digits",
        pattern=r"\b([A-Z]?)20(\d{2})(\d{2})(\d{2})\b",
        replacement=r"\g<1>20\2\3\4",
        description="日期码 P20130328: 确保 20XX 后全为数字",
    ),
    CorrectionRule(
        name="hex_like_serial",
        pattern=r"\b(\d{6,12})\b",
        replacement=r"\1",
        description="纯数字序列号: 不做修改 (占位, 供上下文参考)",
        enabled=False,
    ),
    CorrectionRule(
        name="O_to_0_in_numbers",
        pattern=r"(?<=\d)O(?=\d)",
        replacement="0",
        description="数字序列中的 O → 0 (如 80O8 → 8008)",
    ),
    CorrectionRule(
        name="I_to_1_in_numbers",
        pattern=r"(?<=\d)[Il](?=\d)",
        replacement="1",
        description="数字序列中的 I/l → 1",
    ),
    CorrectionRule(
        name="S_to_5_in_numbers",
        pattern=r"(?<=\d)S(?=\d)",
        replacement="5",
        description="数字序列中的 S → 5",
    ),
    CorrectionRule(
        name="B_to_8_in_numbers",
        pattern=r"(?<=\d)B(?=\d)",
        replacement="8",
        description="数字序列中的 B → 8",
    ),
    CorrectionRule(
        name="Z_to_2_in_numbers",
        pattern=r"(?<=\d)Z(?=\d)",
        replacement="2",
        description="数字序列中的 Z → 2",
    ),
    CorrectionRule(
        name="G_to_6_trailing",
        pattern=r"(?<=\d)G\b",
        replacement="6",
        description="数字末尾的 G → 6 (如 GE → 6E, 但保留独立 G)",
        enabled=False,  # 默认关: GE 可能是合法后缀
    ),
]


def apply_corrections(text: str,
                      rules: list[CorrectionRule] | None = None,
                      custom_rules: list[dict] | None = None,
                      ) -> tuple[str, list[str]]:
    """对 OCR 文本执行规则链纠错。

    Args:
        text: 原始 OCR 文本 (多行)
        rules: 内置规则列表 (None 则使用 BUILTIN_RULES)
        custom_rules: 用户自定义规则 [{"pattern":..., "replacement":..., "name":...}]

    Returns:
        (corrected_text, change_log) - 纠错后文本 + 修改记录
    """
    if rules is None:
        rules = BUILTIN_RULES

    changes: list[str] = []
    lines = text.split("\n")
    corrected_lines = []

    for line in lines:
        original_line = line
        # 内置规则
        for rule in rules:
            if not rule.enabled:
                continue
            try:
                new_line = re.sub(rule.pattern, rule.replacement, line)
                if new_line != line:
                    changes.append(f"[{rule.name}] \"{line}\" → \"{new_line}\"")
                    line = new_line
            except re.error:
                pass  # 无效正则跳过

        # 自定义规则
        if custom_rules:
            for cr in custom_rules:
                pat = cr.get("pattern", "")
                rep = cr.get("replacement", "")
                if not pat:
                    continue
                try:
                    new_line = re.sub(pat, rep, line)
                    if new_line != line:
                        changes.append(f"[custom:{cr.get('name','')}] \"{line}\" → \"{new_line}\"")
                        line = new_line
                except re.error:
                    pass

        corrected_lines.append(line)

    return "\n".join(corrected_lines), changes


def get_default_rules_display() -> str:
    """返回内置规则的可读描述 (供 UI 展示)。"""
    lines = []
    for r in BUILTIN_RULES:
        status = "✓" if r.enabled else "✗"
        lines.append(f"{status} {r.name}: {r.description}")
    return "\n".join(lines)
