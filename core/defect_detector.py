"""缺陷检测高层逻辑 (Phase 4A)

职责:
- 调用 Grounding DINO 引擎执行零样本检测
- 在图像上绘制检测框 + 标签
- OK/NG 判定 (有缺陷 → NG, 无缺陷 → OK)
- 产品配方管理 (每个产品保存缺陷描述词, 一键切换)
- 检测结果落库 (qc_results 表)
- 中文提示词自动翻译为英文 (Grounding DINO 仅支持英文 BERT)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("visionocr.defect")

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
    p = _RECIPES_DIR / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_recipe(name: str, prompt: str, threshold: float = 0.3,
                note: str = "") -> None:
    """保存产品配方。"""
    _RECIPES_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "prompt": prompt,
        "threshold": threshold,
        "note": note,
    }
    p = _RECIPES_DIR / f"{name}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def delete_recipe(name: str) -> bool:
    """删除产品配方。"""
    p = _RECIPES_DIR / f"{name}.json"
    if p.exists():
        p.unlink()
        return True
    return False


# ─── 检测 + 标注 ────────────────────────────────────────────
def run_detection(registry, image_path: str, prompt: str = "",
                  threshold: float = 0.3) -> dict:
    """执行缺陷检测并返回标注结果。

    Args:
        registry: EngineRegistry 实例
        image_path: 图像文件路径
        prompt: 缺陷描述词 (点分隔)
        threshold: 置信度阈值

    Returns:
        {"image": np.ndarray (BGR标注后), "verdict": "OK"/"NG",
         "detections": [...], "count": int, "max_score": float}
    """
    import cv2

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return {"image": None, "verdict": "ERROR", "detections": [],
                "count": 0, "max_score": 0, "error": "无法读取图像"}

    # 获取引擎
    engine = registry.get("grounding_dino")
    if engine is None:
        return {"image": None, "verdict": "ERROR", "detections": [],
                "count": 0, "max_score": 0, "error": "Grounding DINO 引擎未注册"}

    # 确保加载
    if not engine.is_ready():
        registry.ensure_loaded("grounding_dino")
    if not engine.is_ready():
        return {"image": None, "verdict": "ERROR", "detections": [],
                "count": 0, "max_score": 0, "error": "模型加载失败"}

    # 推理 (中文提示词自动翻译为英文)
    if not prompt.strip():
        prompt = DEFAULT_PROMPT
    en_prompt = translate_prompt(prompt)
    result = engine.infer(image_path, prompt=en_prompt, threshold=threshold)

    if result.get("error"):
        return {"image": img, "verdict": "ERROR", "detections": [],
                "count": 0, "max_score": 0, "error": result["error"]}

    boxes = result["boxes"]
    labels = result["labels"]
    scores = result["scores"]

    # 标注图像
    annotated = _draw_detections(img, boxes, labels, scores)

    # 判定: 有检测框 → NG
    verdict = "NG" if len(boxes) > 0 else "OK"
    max_score = max(scores) if scores else 0.0

    detections = [
        {"box": b, "label": l, "score": s}
        for b, l, s in zip(boxes, labels, scores)
    ]

    return {
        "image": annotated,
        "verdict": verdict,
        "detections": detections,
        "count": len(boxes),
        "max_score": round(max_score, 4),
    }


def draw_verdict_badge(img: np.ndarray, verdict: str, count: int = 0,
                       alpha: float = 0.75) -> np.ndarray:
    """在图像右上角绘制大面积 OK/NG 印章, 供工人一眼判定。

    Args:
        img:     BGR 图像 (会被原地修改)。
        verdict: "OK" 或 "NG"。
        count:   缺陷数量 (NG 时显示)。
        alpha:   印章背景不透明度。

    Returns:
        修改后的图像 (同一引用)。
    """
    import cv2

    h, w = img.shape[:2]
    # 印章尺寸随图像等比缩放
    badge_h = max(60, h // 8)
    font_scale = badge_h / 55.0
    thickness_txt = max(3, int(font_scale * 2.5))

    if "OK" in verdict.upper() and "NG" not in verdict.upper():
        text = "OK"
        bg_color = (0, 180, 0)       # BGR 绿
    else:
        text = f"NG  x{count}" if count > 0 else "NG"
        bg_color = (0, 0, 220)       # BGR 红

    # 计算文字尺寸
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, thickness_txt)
    pad_x, pad_y = int(badge_h * 0.4), int(badge_h * 0.25)
    bw, bh = tw + pad_x * 2, th + pad_y * 2 + baseline

    # 右上角定位
    x0 = w - bw - max(10, w // 50)
    y0 = max(10, h // 50)
    x1, y1 = x0 + bw, y0 + bh

    # 半透明背景叠加
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, -1)
    # 边框加粗突出
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), max(3, badge_h // 20))
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # 文字 (白色粗体)
    tx = x0 + pad_x
    ty = y0 + pad_y + th
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness_txt, cv2.LINE_AA)
    return img


# 缺陷框配色 (BGR): 按常见类型区分颜色, 未匹配则用红色
_DEFECT_COLORS = {
    "scratch":   (0, 165, 255),   # 橙
    "dent":      (0, 0, 255),     # 红
    "crack":     (0, 0, 200),     # 深红
    "stain":     (255, 180, 0),   # 蓝
    "burr":      (0, 200, 200),   # 黄
    "missing":   (180, 0, 255),   # 紫
    "deform":    (0, 100, 255),   # 橙红
}


def _pick_color(label: str) -> tuple:
    """按缺陷类型关键词选颜色。"""
    low = label.lower()
    for key, color in _DEFECT_COLORS.items():
        if key in low:
            return color
    return (0, 0, 255)  # 默认红


def _draw_detections(img: np.ndarray, boxes: list, labels: list,
                 scores: list) -> np.ndarray:
    """在图像上绘制检测框、编号标签和 OK/NG 大印章。"""
    import cv2

    annotated = img.copy()
    h, w = annotated.shape[:2]

    base_thickness = max(3, min(w, h) // 200)
    font_scale = max(0.6, min(w, h) / 1200)
    circle_r = max(14, min(w, h) // 45)

    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores), 1):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = _pick_color(label)

        # 检测框 (加粗)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, base_thickness)

        # 编号圆圈 (左上角, 与明细表行号对应)
        cx, cy = x1, y1
        cv2.circle(annotated, (cx, cy), circle_r, color, -1)
        cv2.circle(annotated, (cx, cy), circle_r, (255, 255, 255), 2)
        num_text = str(idx)
        (ntw, nth), _ = cv2.getTextSize(num_text, cv2.FONT_HERSHEY_SIMPLEX,
                                         font_scale * 0.9, 2)
        cv2.putText(annotated, num_text,
                    (cx - ntw // 2, cy + nth // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # 标签 (编号 + 类型 + 分数, 带背景)
        text = f"#{idx} {label} {score:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, 2)
        label_y = y1 - th - 10
        if label_y < 0:
            label_y = y2 + th + 10  # 框上方放不下则放下方
        cv2.rectangle(annotated, (x1, label_y - th - 4),
                      (x1 + tw + 8, label_y + 4), color, -1)
        cv2.putText(annotated, text, (x1 + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # 大印章: OK / NG
    verdict = "NG" if boxes else "OK"
    draw_verdict_badge(annotated, verdict, count=len(boxes))

    return annotated


# ─── 结果落库 ────────────────────────────────────────────────
def save_qc_result(conn: sqlite3.Connection, image_path: str,
                   verdict: str, detections: list[dict],
                   max_score: float = 0.0, prompt: str = "") -> int:
    """将检测结果写入 qc_results 表。"""
    defect_json = json.dumps(detections, ensure_ascii=False)
    cur = conn.execute(
        """INSERT INTO qc_results
           (image_path, verdict, anomaly_score, defect_json, barcode_content)
           VALUES (?, ?, ?, ?, ?)""",
        (image_path, verdict, max_score, defect_json[:5000], prompt[:200]),
    )
    conn.commit()
    return int(cur.lastrowid)
