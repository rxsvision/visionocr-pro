"""质检结果落库: 图片持久化 + qc_results 写入
(自 defect_detector.py 拆分, v1.5.0)
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("visionocr.defect")


def persist_qc_image(image_path: str, dest_dir: Path | str) -> str:
    """入库前把图片复制到稳定目录, 返回持久化路径。

    Gradio 上传会产生临时文件, 清理后看板图片直链 404。
    入库前复制一份到 data/qc_images/, 文件名 = 内容 sha1[:16] + 原扩展名
    (同图重复检测天然去重, 不产生冗余副本)。

    降级策略: 源文件不存在 / 已在目标目录 / 复制失败时原样返回,
    持久化失败不阻断检测落库。
    """
    src = Path(image_path)
    dest_dir = Path(dest_dir)
    if not src.is_file():
        return image_path
    try:
        if src.resolve().parent == dest_dir.resolve():
            return image_path
    except OSError:
        pass
    try:
        data = src.read_bytes()
        name = hashlib.sha1(data).hexdigest()[:16] + (src.suffix.lower() or ".png")
        dest = dest_dir / name
        if not dest.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        return str(dest)
    except OSError:
        logger.warning("[落库] 图片持久化失败, 使用原路径: %s", image_path)
        return image_path


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
