"""YOLO 缺陷检测权重的产品绑定解析 — 跨域误报防护

设计动机:
    YOLO 检测的是训练集标注的缺陷类别, 跨产品使用会大量误报
    (实测 PCB 权重把金属划伤误判为「鼠咬」)。因此 YOLO 检测源
    只在「当前产品有专属训练权重」时激活, 否则 Union 跳过该源。

权重解析规则 (product_name → 权重路径):
    - 空产品名 / (新建) / (自定义)  → None (无产品上下文, 不激活 YOLO)
    - models/yolo/{product}.pt 存在  → 该产品专属权重 (激活)
    - 否则                            → None (该产品未训练 YOLO, 跳过)

注意: 不做「通用兜底权重」回退 — 那正是跨域误报的根源。
      PCB 权重要参与检测, 必须显式命名为对应产品 (如 models/yolo/PCB.pt)。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger("visionocr.yolo_products")

_ROOT = Path(__file__).resolve().parents[1]
_YOLO_DIR = _ROOT / "models" / "yolo"

# 非产品占位符 (下拉框默认项)
_PLACEHOLDERS = {"", "(新建)", "(自定义)"}


def _safe_stem(product_name: str) -> str:
    """产品名 → 安全文件名主干 (去除路径分隔符等非法字符)。"""
    return re.sub(r'[\\/:*?"<>|]', "_", product_name.strip())


def is_real_product(product_name: str | None) -> bool:
    """是否为真实产品名 (排除空值与下拉框占位符)。"""
    if not product_name:
        return False
    return product_name.strip() not in _PLACEHOLDERS


def resolve_yolo_weights(product_name: str | None) -> Path | None:
    """按产品解析 YOLO 权重路径, 无专属权重返回 None。"""
    if not is_real_product(product_name):
        return None
    candidate = _YOLO_DIR / f"{_safe_stem(product_name)}.pt"
    if candidate.exists():
        return candidate
    logger.debug("产品「%s」无 YOLO 权重 (%s 不存在), 跳过 YOLO 检测源",
                 product_name, candidate)
    return None


def list_yolo_products() -> list[str]:
    """列出已训练 YOLO 权重的产品名 (供 UI/状态面板)。"""
    if not _YOLO_DIR.is_dir():
        return []
    return sorted(p.stem for p in _YOLO_DIR.glob("*.pt"))


def yolo_dir() -> Path:
    """YOLO 产品权重目录 (供文档/部署提示)。"""
    return _YOLO_DIR
