"""工业质检 UI 编排逻辑 (无 gradio 依赖, 可在最小 CI 依赖下测试)

自 ui/tab_qc.py 抽离: Union / Grounding DINO / 3D 融合三种检测
流程中"结果装配 + 判定文案 + 落库"的编排逻辑集中于此, tab 文件仅负责
gradio 组件装配与流式日志。

依赖约束: 本模块禁止 import gradio, 且仅使用 requirements-test.txt
最小依赖链 (sqlite3 / 标准库 + core 懒加载模块), 保证 pytest 套件
在无重推理栈环境下可导入、可运行。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from core.database import get_conn
from core.qc_persist import persist_qc_image, save_qc_result

logger = logging.getLogger("visionocr.qc_flow")


@dataclass
class QcView:
    """一次检测结果的展示视图 (对应 tab_qc 右列输出组件)。"""
    verdict_str: str
    score_str: str
    count_str: str
    table: list = field(default_factory=list)
    status: str = ""
    detections: list = field(default_factory=list)
    max_score: float = 0.0


def resolve_fusion_mode(fusion_mode_label: str | None) -> str:
    """融合策略文案 -> 内部枚举: 'and' / 'depth_only' / 'or' (默认)。"""
    if "AND" in (fusion_mode_label or ""):
        return "and"
    if "仅深度" in (fusion_mode_label or ""):
        return "depth_only"
    return "or"


def resolve_product_name(choice: str | None) -> str:
    """下拉选择 -> 产品名; '(新建)'/空 -> "" (走默认库/自动匹配)。"""
    if choice in ("(新建)", None):
        return ""
    return choice or ""


def assemble_union_view(result: dict, product: str = "") -> QcView:
    """Union 零漏检结果 -> 统一明细表 + detections (落库用) + 判定文案。

    注: max_score 仅统计 dino/yolo (0~1 概率语义); patchcore 距离分与
    dinov2 NLL 分为无界量纲, 混入会破坏百分比显示。
    """
    verdict = result["verdict"]
    sources = result.get("ng_sources", [])
    pc = result.get("patchcore")
    dino = result.get("dino")
    yolo = result.get("yolo")
    dv = result.get("dinov2")

    table = []
    detections = []
    max_score = 0.0
    row_no = 0
    if pc:
        row_no += 1
        table.append([str(row_no), "[PatchCore] 表面异常",
                      f"{pc.get('score', 0):.4f}", "热力图"])
        detections.append({"source": "patchcore", "label": "表面异常",
                           "score": pc.get("score", 0)})
    if dv:
        row_no += 1
        table.append([str(row_no), "[DINOv2] 表面异常",
                      f"{dv.get('score', 0):.4f}", "热力图"])
        detections.append({"source": "dinov2", "label": "表面异常",
                           "score": dv.get("score", 0)})
    if dino:
        for det in dino.get("detections", []):
            box = det["box"]
            row_no += 1
            table.append([str(row_no), f"[DINO] {det['label']}",
                          f"{det['score']:.2%}",
                          f"({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})"])
            detections.append({"source": "dino", **det})
        max_score = max(max_score, float(dino.get("max_score", 0)))
    if yolo:
        for b, l, s in zip(yolo.get("boxes", []),
                           yolo.get("labels", []),
                           yolo.get("scores", [])):
            row_no += 1
            table.append([str(row_no), f"[YOLO] {l}", f"{s:.2%}",
                          f"({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})"])
            detections.append({"source": "yolo", "box": b,
                               "label": l, "score": s})
        max_score = max(max_score, float(yolo.get("max_score", 0)))

    if verdict == "OK":
        verdict_str = "✓ OK (合格)"
    elif verdict == "REVIEW":
        verdict_str = (f"◐ REVIEW 待人工复核 "
                       f"(触发源: {'+'.join(sources)}; 单源孤证不自主判NG)")
    else:
        verdict_str = f"✗ NG (触发源: {'+'.join(sources)})"

    active = [s for s, r in (("PatchCore", pc), ("DINO", dino),
                             ("YOLO", yolo), ("DINOv2", dv))
              if r]
    _fused = result.get("fusion", {})
    if (_fused.get("mode") or "staged") == "or":
        _finfo = "融合: 纯OR (v1.3.0)"
    else:
        _ncal = _fused.get("n_cal")
        _finfo = (f"融合: 阶段{_fused.get('stage', '?')} "
                  f"(n_cal={_ncal if _ncal is not None else '—'})")
    status = (f"Union 零漏检 · 产品: {product or '默认'} · "
              f"激活源: {'+'.join(active) or '无'} · {_finfo}")

    return QcView(verdict_str=verdict_str,
                  score_str=f"{max_score:.2%}",
                  count_str=str(len(detections)),
                  table=table, status=status,
                  detections=detections, max_score=max_score)


def assemble_dino_view(result: dict, prompt: str, threshold) -> QcView:
    """Grounding DINO 零样本结果 -> 明细表 + 判定文案。"""
    verdict = result["verdict"]
    max_score = result["max_score"]
    count = result["count"]

    if verdict == "OK":
        verdict_str = "✓ OK (合格)"
    else:
        verdict_str = f"✗ NG (不合格 · {count}处缺陷)"

    table = []
    for idx, det in enumerate(result["detections"], 1):
        box = det["box"]
        table.append([
            str(idx),
            det["label"],
            f"{det['score']:.2%}",
            f"({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})",
        ])

    status = f"检测完成 · 提示词: {prompt[:60]}... · 阈值: {threshold}"
    return QcView(verdict_str=verdict_str,
                  score_str=f"{max_score:.2%}",
                  count_str=str(count),
                  table=table, status=status,
                  detections=list(result["detections"]),
                  max_score=max_score)


def format_fusion_display(fused: dict) -> tuple[str, str, str, list]:
    """3D 深度融合结果 -> (判定文案, 分数文案, 缺陷数, 明细表)。"""
    verdict = fused["verdict"]
    if verdict == "OK":
        verdict_str = "✓ OK (合格)"
        score_str = "—"
    else:
        max_conf = max((d["confidence"] for d in fused["fused_defects"]),
                       default=0)
        verdict_str = f"✗ NG (不合格 · {fused['count']}处)"
        score_str = f"{max_conf:.2f}"

    table = []
    for idx, d in enumerate(fused["fused_defects"], 1):
        x1, y1, x2, y2 = d["bbox"]
        table.append([
            str(idx),
            f"{d['source']} · {d['type']}",
            f"{d['confidence']:.0%}",
            f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})",
        ])
    return verdict_str, score_str, str(fused["count"]), table


def persist_qc_result(image_path: str, verdict: str, detections: list,
                      max_score: float, prompt: str, data_dir: str,
                      warn: bool = True) -> bool:
    """检测结果落库 (图片持久化 + qc_results 写入)。

    落库失败不阻断检测流程, 返回 False; warn=True 时记录警告日志
    (对齐 tab_qc 原行为: Union/DINO 记警告, 3D 融合静默)。
    """
    try:
        conn = get_conn(data_dir)
        save_qc_result(
            conn,
            persist_qc_image(image_path, Path(data_dir) / "qc_images"),
            verdict, detections, max_score, prompt)
        conn.close()
        return True
    except Exception as e:  # noqa: BLE001 — 落库失败不阻断检测
        if warn:
            logger.warning("QC 结果落库失败: %s", e)
        return False
