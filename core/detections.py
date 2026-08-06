"""统一检测结果模型 (轻量自研版, 借鉴 roboflow/supervision 设计模式)

设计出处: 竞品调研 (roboflow/supervision v0.30.0, MIT) 结论——值得吸收的
是「统一检测结果 dataclass + from_* 转换器」思想; 但 sv.Detections 装不下
anomaly_map/异常分数, 与 Union 零漏检架构冲突, 故不直接依赖, 仅借鉴模式。

职责边界:
- 覆盖 bbox 类检测源 (gdino/yolo): 统一 box/label/score/area 结构
- 异常源 (patchcore/dinov2) 保留独立通道: 其 score/anomaly_map 语义与
  bbox 检测不同源, 不强行装入本模型
- 对外序列化 (to_legacy_dicts) 与历史 dict 格式字节级一致,
  下游 qc_persist/dashboard/DB 不感知
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.qc_drawing import _bbox_area

# 合法检测源标记 (与 Union ng_sources 命名对齐)
VALID_SOURCES = ("gdino", "yolo", "patchcore", "dinov2")


@dataclass
class Detection:
    """单个检测框 (xyxy)。

    Attributes:
        box: [x1, y1, x2, y2], 原样保留引擎返回类型, 不做拷贝转换
        label: 缺陷类别/描述词
        score: 置信度 (原样, 不取整)
        area_px: 框面积 (像素², round 1 位, 与历史格式一致)
        source: 检测源标记 ("gdino"/"yolo")
    """
    box: list
    label: str
    score: float
    area_px: float
    source: str = ""


@dataclass
class DetectionSet:
    """统一检测结果容器 (轻量, 纯 Python)。"""
    items: list[Detection] = field(default_factory=list)

    # ─── from_* 转换器 (借鉴 supervision 的模型无关转换思想) ────
    @classmethod
    def from_gdino(cls, result: dict) -> "DetectionSet":
        """从 Grounding DINO 结果构造。

        Args:
            result: 含 "boxes"/"labels"/"scores" 的引擎/过滤后结果 dict
        """
        return cls._from_boxes(result, source="gdino")

    @classmethod
    def from_yolo(cls, result: dict) -> "DetectionSet":
        """从 YOLO 缺陷检测结果构造。"""
        return cls._from_boxes(result, source="yolo")

    @classmethod
    def _from_boxes(cls, result: dict, source: str) -> "DetectionSet":
        boxes = result.get("boxes") or []
        labels = result.get("labels") or []
        scores = result.get("scores") or []
        items = [
            Detection(box=b, label=l, score=s,
                      area_px=round(_bbox_area(b), 1), source=source)
            for b, l, s in zip(boxes, labels, scores)
        ]
        return cls(items=items)

    # ─── 序列化: 与历史 dict 格式字节级一致 ─────────────────────
    def to_legacy_dicts(self) -> list[dict]:
        """转为旧格式 dict 列表 ({"box","label","score","area_px"})。"""
        return [
            {"box": d.box, "label": d.label, "score": d.score,
             "area_px": d.area_px}
            for d in self.items
        ]

    # ─── 只读便捷属性 ──────────────────────────────────────────
    @property
    def max_score(self) -> float:
        """最高置信度 (空集返回 0.0)。"""
        return max((d.score for d in self.items), default=0.0)

    def __len__(self) -> int:
        return len(self.items)
