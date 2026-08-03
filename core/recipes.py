"""产品配方管理与缺陷提示词翻译 (自 defect_detector.py 拆分, v1.5.0)

职责:
- 产品配方 CRUD (每个产品保存缺陷描述词/阈值/尺寸过滤, 一键切换)
- 中文缺陷提示词 → 英文翻译 (Grounding DINO 仅支持英文 BERT)
- 配方文件名清洗与路径越界防护
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# ─── 中英缺陷词对照表 (工业外观常见) ─────────────────────────
_ZH_EN_MAP = {
    "划痕": "scratch", "刮伤": "scratch", "划伤": "scratch",
    "凹陷": "dent", "凹坑": "dent", "压痕": "dent",
    "裂纹": "crack", "裂缝": "crack", "开裂": "crack",
    "污渍": "stain", "脏污": "stain", "污点": "stain",
    "毛刺": "burr", "飞边": "burr",
    "色差": "color difference", "变色": "discoloration",
    "缺件": "missing part", "缺失": "missing", "漏装": "missing component",
    "变形": "deformation", "翘曲": "warp", "弯曲": "bend",
    "气泡": "bubble", "气孔": "porosity", "砂眼": "blowhole",
    "锈": "rust", "锈蚀": "rust", "氧化": "oxidation",
    "磨损": "wear", "磨伤": "abrasion",
    "异物": "foreign object", "杂质": "impurity",
    "错位": "misalignment", "偏移": "offset", "倾斜": "tilt",
    "破损": "damage", "断裂": "fracture", "缺口": "notch",
    "溢胶": "glue overflow", "胶渍": "glue residue",
    "焊渣": "solder spatter", "虚焊": "cold solder joint",
    "短路": "short circuit", "断路": "open circuit",
    "标签歪": "misaligned label", "贴歪": "crooked label",
    "印刷不良": "print defect", "漏印": "missing print",
    "缩水": "shrinkage", "飞料": "flash",
    "缺陷": "defect", "不良": "defect", "异常": "anomaly",
}

# 默认提示词 (中文界面, 内部自动翻译为英文)
DEFAULT_PROMPT = "划痕.凹陷.裂纹.污渍.毛刺.色差.缺件.变形"

# 产品配方存储路径
_RECIPES_DIR = Path("data/recipes")


def _safe_name(name: str) -> str:
    """清洗产品配方名，防止路径穿越攻击。"""
    import re
    s = re.sub(r'[\\/:*?"<>|.]', '_', str(name).strip())
    return s or '_'


def _recipe_path(name: str) -> Path:
    """构造并校验配方路径: 优先清洗名, 旧文件用原始名回退兼容 (均不越界)。"""
    root = _RECIPES_DIR.resolve()
    p = (_RECIPES_DIR / f"{_safe_name(name)}.json").resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"路径越界: {name} 非法")
    if p.exists():
        return p
    # 回退兼容 v1.4.1 前已存在的旧文件 (名称含 . 等特殊字符)
    legacy = (_RECIPES_DIR / f"{str(name).strip()}.json").resolve()
    if legacy.is_relative_to(root) and legacy.exists():
        return legacy
    return p


def translate_prompt(prompt: str) -> str:
    """将中文缺陷提示词翻译为英文 (点号分隔)。

    规则:
    - 逐词查表, 命中则替换为英文
    - 未命中且含中文 → 保留原文 (模型可能部分识别)
    - 已是英文 → 原样保留
    """
    terms = [t.strip() for t in prompt.replace("。", ".").split(".") if t.strip()]
    translated = []
    for term in terms:
        en = _ZH_EN_MAP.get(term)
        if en:
            translated.append(en)
        else:
            translated.append(term)
    return ".".join(translated)


# ─── 产品配方 ────────────────────────────────────────────────
def list_recipes() -> list[str]:
    """列出所有已保存的产品配方名。"""
    if not _RECIPES_DIR.exists():
        return []
    return sorted(p.stem for p in _RECIPES_DIR.glob("*.json"))


def load_recipe(name: str) -> Optional[dict]:
    """加载产品配方。"""
    try:
        p = _recipe_path(name)
    except ValueError:
        return None
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_recipe(name: str, prompt: str, threshold: float = 0.3,
                note: str = "",
                min_area_px: int = 0, max_area_px: int = 0,
                pixels_per_mm: float = 0.0) -> None:
    """保存产品配方 (含瑕疵尺寸阈值)。"""
    _RECIPES_DIR.mkdir(parents=True, exist_ok=True)
    p = _recipe_path(name)
    data = {
        "name": p.stem,
        "prompt": prompt,
        "threshold": threshold,
        "note": note,
        "defect_size": {
            "min_area_px": min_area_px,
            "max_area_px": max_area_px,
            "pixels_per_mm": pixels_per_mm,
        },
    }
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_recipe(name: str) -> bool:
    """删除产品配方。"""
    try:
        p = _recipe_path(name)
    except ValueError:
        return False
    if p.exists():
        p.unlink()
        return True
    return False
