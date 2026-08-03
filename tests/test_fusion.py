"""分阶段融合 (v1.4.0 §5.5) 单元测试

纯逻辑测试: staged_fusion / fusion_stage / DriftMonitor /
calibrated_n_samples, 不依赖任何模型权重。
端到端验收走 scripts/eval_acceptance.py fusion55 模式。
"""
from __future__ import annotations

import pytest

from core.fusion import (CALIBRATED_SOURCES, DriftMonitor,
                         calibrated_n_samples, fusion_stage,
                         staged_fusion)


# ─── fusion_stage: 阶段边界 ─────────────────────────────────
class TestFusionStage:
    def test_none_is_stage1(self):
        assert fusion_stage(None) == 1

    @pytest.mark.parametrize("n", [0, 1, 5, 9])
    def test_small_n_stage1(self, n):
        assert fusion_stage(n) == 1

    @pytest.mark.parametrize("n", [10, 30, 49])
    def test_mid_n_stage2(self, n):
        assert fusion_stage(n) == 2

    @pytest.mark.parametrize("n", [50, 55, 500])
    def test_large_n_stage3(self, n):
        assert fusion_stage(n) == 3

    def test_custom_thresholds(self):
        cfg = {"stage2_min_cal": 5, "stage3_min_cal": 20}
        assert fusion_stage(5, cfg) == 2
        assert fusion_stage(4, cfg) == 1
        assert fusion_stage(20, cfg) == 3


# ─── staged_fusion: 判决逻辑 ────────────────────────────────
NCAL_BOTH = {"patchcore": 55, "dinov2": 55}


class TestStagedFusion:
    def test_no_ng_is_ok(self):
        r = staged_fusion([], NCAL_BOTH)
        assert r["verdict"] == "OK"
        assert r["review_required"] is False

    def test_mode_or_is_legacy_behavior(self):
        # v1.3.0 兼容: 任一源 NG 即 NG, 无 REVIEW
        r = staged_fusion(["patchcore"], NCAL_BOTH, {"mode": "or"})
        assert r["verdict"] == "NG"
        assert r["mode"] == "or"

    def test_stage1_low_cal_is_or(self):
        # n_cal=5 < 10 → 单源也直接 NG (保守零漏检)
        low = {"patchcore": 5, "dinov2": 5}
        r = staged_fusion(["patchcore"], low)
        assert r["stage"] == 1
        assert r["verdict"] == "NG"

    def test_stage2_both_calibrated_ng(self):
        r = staged_fusion(["patchcore", "dinov2"],
                          {"patchcore": 20, "dinov2": 20})
        assert r["stage"] == 2
        assert r["verdict"] == "NG"

    def test_stage2_single_calibrated_is_review(self):
        r = staged_fusion(["patchcore"],
                          {"patchcore": 20, "dinov2": 20})
        assert r["verdict"] == "REVIEW"
        assert r["review_required"] is True
        assert r["review_reasons"]

    def test_stage2_calibrated_plus_uncalibrated_is_ng(self):
        # 跨模态互证: 校准源 + GDINO/YOLO
        r = staged_fusion(["patchcore", "dino"],
                          {"patchcore": 20, "dinov2": 20})
        assert r["verdict"] == "NG"

    def test_stage2_uncalibrated_only_is_review(self):
        r = staged_fusion(["dino", "yolo"], NCAL_BOTH)
        assert r["verdict"] == "REVIEW"
        assert r["review_required"] is True

    def test_stage3_same_rule_plus_stage_mark(self):
        r = staged_fusion(["dinov2"], NCAL_BOTH)
        assert r["stage"] == 3
        assert r["verdict"] == "REVIEW"
        r2 = staged_fusion(["patchcore", "dinov2"], NCAL_BOTH)
        assert r2["stage"] == 3
        assert r2["verdict"] == "NG"

    def test_single_calibrated_source_fallback_or(self):
        # 仅一个校准源参与 → 一致投票不可行, 回退 OR (宁可误报不漏检)
        r = staged_fusion(["patchcore"], {"patchcore": 55, "dinov2": None})
        assert r["fallback_or"] is True
        assert r["verdict"] == "NG"

    def test_no_calibrated_source_fallback_or(self):
        # 无任何校准源 (n_cal=None → stage1) → OR
        r = staged_fusion(["dino"], {"patchcore": None, "dinov2": None})
        assert r["stage"] == 1
        assert r["verdict"] == "NG"

    def test_unknown_source_ignored(self):
        # 未知源不参与判决 (防御性)
        r = staged_fusion(["mystery_source"], NCAL_BOTH)
        assert r["verdict"] == "OK"

    def test_n_cal_reports_minimum(self):
        r = staged_fusion(["patchcore"],
                          {"patchcore": 80, "dinov2": 12})
        assert r["n_cal"] == 12
        assert r["stage"] == 2

    def test_calibrated_sources_contract(self):
        assert CALIBRATED_SOURCES == ("patchcore", "dinov2")


# ─── calibrated_n_samples ───────────────────────────────────
class _FakeCalibrator:
    def __init__(self, n, fitted=True):
        self.n_samples = n
        self.is_fitted = fitted


class _FakeEngine:
    def __init__(self, cal):
        self._np_calibrator = cal


class TestCalibratedNSamples:
    def test_fitted(self):
        assert calibrated_n_samples(_FakeEngine(_FakeCalibrator(55))) == 55

    def test_unfitted(self):
        assert calibrated_n_samples(
            _FakeEngine(_FakeCalibrator(55, fitted=False))) is None

    def test_no_calibrator(self):
        assert calibrated_n_samples(_FakeEngine(None)) is None

    def test_engine_without_attr(self):
        class E:
            pass
        assert calibrated_n_samples(E()) is None

    def test_zero(self):
        assert calibrated_n_samples(_FakeEngine(_FakeCalibrator(0))) is None


# ─── DriftMonitor ───────────────────────────────────────────
class TestDriftMonitor:
    def test_no_warning_below_min_samples(self):
        m = DriftMonitor(window=50, min_samples=30)
        out = None
        for _ in range(29):
            out = m.observe("p/patchcore", 99.0, 1.0, 0.10)
        assert out is None

    def test_warning_on_high_exceed_rate(self):
        m = DriftMonitor(window=50, min_samples=30)
        out = None
        for _ in range(35):   # 全部超阈值 (100% >> max(3*0.1, 0.15))
            out = m.observe("p/patchcore", 99.0, 1.0, 0.10)
        assert out is not None
        assert "漂移" in out

    def test_no_warning_at_normal_rate(self):
        m = DriftMonitor(window=100, min_samples=30)
        out = None
        # 5% 超阈值 (低于 max(3*0.10, 0.15) = 0.30)
        for i in range(60):
            score = 99.0 if i < 3 else 0.0
            out = m.observe("p/patchcore", score, 1.0, 0.10)
        assert out is None

    def test_stats(self):
        m = DriftMonitor(window=10, min_samples=5)
        for i in range(6):
            m.observe("k", 2.0 if i < 3 else 0.0, 1.0, 0.10)
        s = m.stats("k")
        assert s["n"] == 6
        assert abs(s["exceed_rate"] - 0.5) < 1e-9
        assert m.stats("missing") is None

    def test_window_cap(self):
        m = DriftMonitor(window=10, min_samples=5)
        for _ in range(30):
            m.observe("k", 0.0, 1.0, 0.10)
        assert m.stats("k")["n"] == 10
