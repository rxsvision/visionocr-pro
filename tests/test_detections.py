"""统一检测结果模型 (core/detections.py) 测试

覆盖:
- from_gdino / from_yolo 转换器正确性
- to_legacy_dicts 与历史手工组装格式字节级等价 (下游兼容铁律)
- 空结果边界
- source 标记与只读属性
"""
from __future__ import annotations

from core.detections import VALID_SOURCES, Detection, DetectionSet
from core.qc_drawing import _bbox_area


def _sample_result():
    return {
        "boxes": [[10, 20, 50, 80], [0, 0, 100, 100]],
        "labels": ["划痕", "污渍"],
        "scores": [0.87, 0.6212],
    }


class TestFromConverters:
    def test_from_gdino_basic(self):
        ds = DetectionSet.from_gdino(_sample_result())
        assert len(ds) == 2
        assert ds.items[0].label == "划痕"
        assert ds.items[0].score == 0.87
        assert ds.items[0].source == "gdino"
        assert ds.items[1].box == [0, 0, 100, 100]

    def test_from_yolo_source_tag(self):
        ds = DetectionSet.from_yolo(_sample_result())
        assert all(d.source == "yolo" for d in ds.items)

    def test_source_markers_align_with_union(self):
        # source 标记与 Union ng_sources 命名体系对齐
        assert set(VALID_SOURCES) == {
            "gdino", "yolo", "patchcore", "dinov2"}

    def test_empty_result(self):
        ds = DetectionSet.from_gdino(
            {"boxes": [], "labels": [], "scores": []})
        assert len(ds) == 0
        assert ds.max_score == 0.0
        assert ds.to_legacy_dicts() == []

    def test_missing_keys_treated_as_empty(self):
        ds = DetectionSet.from_gdino({})
        assert len(ds) == 0

    def test_mismatched_lists_zip_truncates(self):
        ds = DetectionSet.from_gdino(
            {"boxes": [[0, 0, 10, 10]], "labels": ["a"], "scores": []})
        assert len(ds) == 0


class TestArea:
    def test_area_matches_bbox_area(self):
        ds = DetectionSet.from_gdino(_sample_result())
        for d in ds.items:
            assert d.area_px == round(_bbox_area(d.box), 1)

    def test_degenerate_box_area_zero(self):
        ds = DetectionSet.from_gdino(
            {"boxes": [[5, 5, 5, 5]], "labels": ["x"], "scores": [0.5]})
        assert ds.items[0].area_px == 0.0

    def test_inverted_box_matches_legacy_semantics(self):
        # 反转框 (负负得正) 与历史 _bbox_area 语义一致: 等价性优先于几何直觉
        ds = DetectionSet.from_gdino(
            {"boxes": [[50, 50, 10, 10]], "labels": ["x"], "scores": [0.5]})
        assert ds.items[0].area_px == round(_bbox_area([50, 50, 10, 10]), 1)


class TestLegacyCompat:
    """to_legacy_dicts 必须与 v1.5.0 手工组装格式字节级一致"""

    def test_byte_equivalence_with_legacy_comprehension(self):
        res = _sample_result()
        boxes, labels, scores = res["boxes"], res["labels"], res["scores"]
        # 历史实现 (run_detection v1.5.0 原逻辑)
        legacy = [
            {"box": b, "label": l, "score": s,
             "area_px": round(_bbox_area(b), 1)}
            for b, l, s in zip(boxes, labels, scores)
        ]
        new = DetectionSet.from_gdino(res).to_legacy_dicts()
        assert new == legacy

    def test_box_object_identity_preserved(self):
        # box 原样透传 (不拷贝/不转类型), 下游若持有引用不受影响
        boxes = [[1, 2, 3, 4]]
        ds = DetectionSet.from_gdino(
            {"boxes": boxes, "labels": ["a"], "scores": [0.9]})
        assert ds.to_legacy_dicts()[0]["box"] is boxes[0]

    def test_score_not_rounded(self):
        # 历史格式中逐框 score 不取整 (仅顶层 max_score 取整)
        ds = DetectionSet.from_gdino(_sample_result())
        assert ds.to_legacy_dicts()[1]["score"] == 0.6212


class TestReadOnlyProps:
    def test_max_score(self):
        ds = DetectionSet.from_gdino(_sample_result())
        assert ds.max_score == 0.87

    def test_len(self):
        ds = DetectionSet.from_gdino(_sample_result())
        assert len(ds) == 2

    def test_detection_is_dataclass(self):
        d = Detection(box=[0, 0, 1, 1], label="a", score=0.5,
                      area_px=1.0, source="gdino")
        assert d.source == "gdino"
