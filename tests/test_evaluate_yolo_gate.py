"""finetune/evaluate_yolo.py 团标 §6.2 门控 (spec_gate) 单元测试

门控语义 (docs/spec_alignment.md):
- 主门控仅 mAP@50 (曲线下面积, 与 conf 无关, 不可操纵)
- P/R 为次级参考, 不参与 FAIL
- 单一测量口径: ultralytics val 默认参数 (避免门控基准漂移)
- 门槛数值参数化: standard=0.80, high=0.85
"""
import pytest

from finetune.evaluate_yolo import SPEC_TIERS, spec_gate


class TestStandardTier:
    def test_exact_boundary_passes(self):
        """恰好达标 (0.80) 判 PASS。"""
        passed, reasons = spec_gate({"map50": 0.80})
        assert passed is True
        assert reasons == []

    def test_below_boundary_fails(self):
        """单点未达标判 FAIL, 原因含阈值与档位。"""
        passed, reasons = spec_gate({"map50": 0.7999})
        assert passed is False
        assert len(reasons) == 1
        assert "0.80" in reasons[0] and "standard" in reasons[0]


class TestHighTier:
    def test_standard_pass_high_fail(self):
        """0.82 过标准档但不过高精度档 (阈值切换生效)。"""
        assert spec_gate({"map50": 0.82}, tier="standard")[0] is True
        passed, reasons = spec_gate({"map50": 0.82}, tier="high")
        assert passed is False
        assert "0.85" in reasons[0]

    def test_high_boundary_passes(self):
        passed, _ = spec_gate({"map50": 0.85}, tier="high")
        assert passed is True


class TestSecondaryReference:
    def test_pr_never_triggers_fail(self):
        """P/R 缺失或偏低均不影响主门控 (仅次级参考)。"""
        passed, _ = spec_gate({"map50": 0.90})
        assert passed is True
        passed, _ = spec_gate({"map50": 0.90, "p": 0.10, "r": 0.10})
        assert passed is True


class TestSafety:
    def test_missing_map50_fails_gracefully(self):
        """缺键不抛异常, 判 FAIL 并给出原因。"""
        passed, reasons = spec_gate({})
        assert passed is False
        assert "缺失" in reasons[0]

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError, match="未知 spec tier"):
            spec_gate({"map50": 0.9}, tier="ultra")

    def test_tiers_match_spec_values(self):
        """门槛数值与团标 §6.2 一致 (定稿后仅需改 SPEC_TIERS)。"""
        assert SPEC_TIERS["standard"] == {"map50": 0.80, "p": 0.85, "r": 0.85}
        assert SPEC_TIERS["high"] == {"map50": 0.85, "p": 0.90, "r": 0.90}
