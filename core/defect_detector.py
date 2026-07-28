"""缺陷检测高层逻辑 (Phase 4A)

职责:
- 调用 Grounding DINO 引擎执行零样本检测
- 在图像上绘制检测框 + 标签
- OK/NG 判定 (有缺陷 → NG, 无缺陷 → OK)
- 产品配方管理 (每个产品保存缺陷描述词, 一键切换)
- 检测结果落库 (qc_results 表)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("visionocr.defect")

# 默认缺陷提示词 (通用工业外观)
DEFAULT_PROMPT = "scratch.dent.crack.stain.burr.color difference.missing part.deformation"

# 产品配方存储路径
_RECIPES_DIR = Path("data/recipes")


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

    # 推理
    if not prompt.strip():
        prompt = DEFAULT_PROMPT
    result = engine.infer(image_path, prompt=prompt, threshold=threshold)

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


def _draw_detections(img: np.ndarray, boxes: list, labels: list,
                     scores: list) -> np.ndarray:
    """在图像上绘制检测框和标签。"""
    import cv2

    annotated = img.copy()
    h, w = annotated.shape[:2]

    # 颜色: NG 红框, 按分数渐变
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        # 红色框 (BGR)
        color = (0, 0, 255)
        thickness = max(2, min(w, h) // 300)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # 标签背景
        text = f"{label} {score:.2f}"
        font_scale = max(0.5, min(w, h) / 1500)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1),
                      color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    # 左上角状态标记
    if boxes:
        cv2.putText(annotated, f"NG ({len(boxes)})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        cv2.putText(annotated, "OK", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)

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
