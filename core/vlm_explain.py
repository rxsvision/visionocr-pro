"""VLM 缺陷解释 (Phase 3) — 智能 ROI 裁切 + 本地 VLM 局部解读

链路: Union 检测结果 (融合热力图 + 检测框)
      → core.roi_selector 选出 top-k 缺陷候选区
      → 裁切/缩放 → 本地 Ollama VLM (qwen3-vl) 逐区解释
      → 汇总文本 (供 UI 展示 / 留档)

设计取舍:
- VLM 看裁切局部而非整图: 高分辨率工业图整图解释慢且易失焦,
  候选区裁切后 token 消耗与延迟显著下降, 描述更聚焦。
- Ollama 为黑盒 HTTP 服务, 无法在其内部做 token 级剪枝 (如 MMTok),
  故在输入侧做 ROI 裁切是工程上可行的等效路径。
- VLM 不可达时优雅降级 (返回 error 文案), 不影响检测主链路。
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from core.imutils import imread_unicode, imwrite_unicode
from core.roi_selector import select_rois, crop_rois

logger = logging.getLogger("visionocr.vlm_explain")

# 内置工业质检提示词 (短回答约束, 产线可读)
DEFAULT_PROMPT = (
    "你是工业产品外观质检员。图片是产品表面的局部放大区域, 可能包含缺陷。"
    "请用不超过60字的中文回答: "
    "1) 缺陷类型(划痕/凹陷/污渍/异物/崩缺/色差/无缺陷等); "
    "2) 严重程度(高/中/低/无)。"
)


def _collect_boxes(union_result: dict) -> tuple[list, list]:
    """从 Union 结果提取 DINO/YOLO 检测框与置信度。"""
    boxes, scores = [], []
    dino = union_result.get("dino") or {}
    for det in dino.get("detections", []) or []:
        if det.get("box"):
            boxes.append(det["box"])
            scores.append(float(det.get("score", 1.0)))
    yolo = union_result.get("yolo") or {}
    for b, s in zip(yolo.get("boxes", []) or [],
                    yolo.get("scores", []) or []):
        boxes.append(b)
        scores.append(float(s))
    return boxes, scores


def explain_union(registry, image_path: str, union_result: dict,
                  config: dict | None = None) -> dict:
    """对 Union NG 结果做 VLM 局部解释。

    Args:
        registry:     EngineRegistry 实例。
        image_path:   待检图路径。
        union_result: run_union_detection 的返回 (含 anomaly_map)。
        config:       全局配置 (读取 qc.vlm_explain 段)。

    Returns:
        {"rois": [...], "crops": [np.ndarray], "texts": [str],
         "summary": str, "error"?: str}
        error 存在时表示解释未完成 (UI 应展示 error 文案)。
    """
    cfg = ((config or {}).get("qc", {}) or {}).get("vlm_explain", {})
    if not cfg.get("enabled", True):
        return {"rois": [], "crops": [], "texts": [], "summary": "",
                "error": "VLM 解释已在配置中禁用 (qc.vlm_explain.enabled)"}

    if not union_result or union_result.get("verdict") != "NG":
        return {"rois": [], "crops": [], "texts": [],
                "summary": "判定 OK, 无需缺陷解释。"}

    # ── VLM 引擎就绪 ──
    engine = registry.get("ollama_vlm") if registry else None
    if engine is None:
        return {"rois": [], "crops": [], "texts": [], "summary": "",
                "error": "ollama_vlm 引擎未注册"}
    if not engine.is_ready():
        try:
            registry.ensure_loaded("ollama_vlm")
        except Exception as e:  # noqa: BLE001
            logger.warning("VLM 加载失败: %s", e)
    if not engine.is_ready():
        return {"rois": [], "crops": [], "texts": [], "summary": "",
                "error": "VLM 未就绪: 请确认 Ollama 服务在线且已拉取 "
                         "qwen3-vl 模型 (ollama pull qwen3-vl:8b)"}

    # ── 读图 + 选 ROI ──
    img = imread_unicode(image_path)
    if img is None:
        return {"rois": [], "crops": [], "texts": [], "summary": "",
                "error": f"图像读取失败: {image_path}"}

    boxes, box_scores = _collect_boxes(union_result)
    rois = select_rois(
        img.shape,
        anomaly_map=union_result.get("anomaly_map"),
        boxes=boxes, box_scores=box_scores,
        max_rois=int(cfg.get("max_rois", 3)),
        min_area_frac=float(cfg.get("min_area_frac", 0.0005)),
        pad_frac=float(cfg.get("pad_frac", 0.25)),
        rel_thresh=float(cfg.get("rel_thresh", 0.45)),
    )
    if not rois:
        # NG 但无定位证据 (如仅分数触发): 整图兜底
        h, w = img.shape[:2]
        rois = [{"x": 0, "y": 0, "w": w, "h": h,
                 "score": 0.0, "source": "full"}]

    cropped = crop_rois(img, rois)
    if not cropped:
        return {"rois": rois, "crops": [], "texts": [], "summary": "",
                "error": "ROI 裁切为空"}

    # ── 逐区 VLM 解释 (临时 PNG → base64 由引擎内部处理) ──
    prompt = str(cfg.get("prompt", "") or DEFAULT_PROMPT)
    if cfg.get("no_think", True):
        prompt = prompt + " /no_think"
    max_tokens = int(cfg.get("max_tokens", 512))

    texts, crops = [], []
    tmp_dir = Path(tempfile.mkdtemp(prefix="visionocr_roi_"))
    try:
        for i, (roi, crop) in enumerate(cropped):
            p = tmp_dir / f"roi_{i}.png"
            if not imwrite_unicode(str(p), crop):
                texts.append("[裁切写入失败]")
                continue
            r = engine.infer(str(p), prompt, max_tokens=max_tokens)
            if r.get("error"):
                logger.warning("VLM ROI%d 解释失败: %s", i, r["error"])
                texts.append(f"[解释失败: {r['error']}]")
            else:
                texts.append((r.get("text") or "").strip() or "[空回答]")
            crops.append(crop)
    finally:
        for f in tmp_dir.glob("*.png"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    lines = [f"区域{i + 1} [{r['source']}]: {t}"
             for i, (r, t) in enumerate(zip(rois[:len(texts)], texts))]
    summary = "\n".join(lines)
    logger.info("VLM 解释完成: %d 个 ROI", len(texts))
    return {"rois": rois, "crops": crops, "texts": texts,
            "summary": summary}
