"""NPCalibrator 单元测试 (Phase: NP校准)

验证目标:
1. FPR 保证: 留出正常样本经验误报率 ≤ epsilon + 容差
2. 校准概率单调性与取值域
3. 持久化往返一致
4. 边界情况: 样本不足 / 恒定分数 / 非法 epsilon
"""
import numpy as np
import pytest

from core.np_calibration import NPCalibrator


def _skewed_scores(rng, n):
    """模拟 PatchCore 分数分布: 右偏 (lognormal + 少量尾部)。"""
    return rng.lognormal(mean=-2.0, sigma=0.6, size=n)


class TestFPRGuarantee:
    """核心保证: 正常样本误报率 ≤ epsilon (有限样本, 分布无关)。"""

    @pytest.mark.parametrize("epsilon", [0.01, 0.02, 0.05, 0.10])
    def test_held_out_fpr_within_bound(self, epsilon):
        rng = np.random.default_rng(42)
        cal_scores = _skewed_scores(rng, 500)
        held_out = _skewed_scores(rng, 5000)

        calib = NPCalibrator(epsilon=epsilon)
        assert calib.fit(cal_scores)
        flags = [calib.decide(s) for s in held_out]
        empirical_fpr = sum(flags) / len(held_out)
        # 允许 +1% 采样容差 (5000样本的统计波动)
        assert empirical_fpr <= epsilon + 0.01, (
            f"FPR超限: eps={epsilon}, 实测={empirical_fpr:.4f}, "
            f"tau={calib.threshold:.4f}")

    def test_fpr_on_gaussian_marginal(self):
        """高斯分布下验证边际保证: 多次随机划分的平均FPR ≤ epsilon。

        说明: split-conformal 的 FPR≤ε 是对联合随机性的边际保证;
        单个固定校准集的条件 FPR 会波动 (尤其 n×ε 小时), 故测试
        用多随机划分取均值验证边际性质。
        """
        fprs = []
        for seed in range(12):
            rng = np.random.default_rng(seed)
            cal_scores = np.abs(rng.normal(5.0, 1.0, 300))
            held_out = np.abs(rng.normal(5.0, 1.0, 3000))
            calib = NPCalibrator(epsilon=0.05)
            assert calib.fit(cal_scores)
            fprs.append(float(np.mean([calib.decide(s) for s in held_out])))
        mean_fpr = float(np.mean(fprs))
        assert mean_fpr <= 0.06, f"边际FPR超限: {mean_fpr:.4f}"

    def test_threshold_monotone_in_epsilon(self):
        """epsilon 越大 → 阈值越低 (更宽松)。"""
        rng = np.random.default_rng(3)
        scores = _skewed_scores(rng, 400)
        taus = []
        for eps in [0.01, 0.05, 0.10, 0.20]:
            calib = NPCalibrator(epsilon=eps)
            calib.fit(scores)
            taus.append(calib.threshold)
        # 非递增
        assert all(taus[i] >= taus[i + 1] - 1e-12
                   for i in range(len(taus) - 1))


class TestCalibratedScores:
    def test_anomaly_confidence_range_and_monotone(self):
        rng = np.random.default_rng(11)
        calib = NPCalibrator(epsilon=0.02)
        assert calib.fit(_skewed_scores(rng, 200))

        grid = np.linspace(0, 5.0, 50)
        confs = [calib.anomaly_confidence(s) for s in grid]
        assert all(0.0 <= c <= 1.0 for c in confs)
        # 单调非降 (分数越高越异常)
        assert all(confs[i] <= confs[i + 1] + 1e-9
                   for i in range(len(confs) - 1))

    def test_extreme_scores(self):
        rng = np.random.default_rng(13)
        calib = NPCalibrator(epsilon=0.02)
        calib.fit(_skewed_scores(rng, 300))
        # 正常范围内分数 → 低异常置信; 远超范围 → 接近1
        typical = float(np.median(calib._sorted_scores))
        assert calib.anomaly_confidence(typical) < 0.7
        assert calib.anomaly_confidence(calib.threshold * 10 + 5) > 0.95

    def test_decide_consistent_with_threshold(self):
        rng = np.random.default_rng(17)
        calib = NPCalibrator(epsilon=0.05)
        calib.fit(_skewed_scores(rng, 150))
        assert calib.decide(calib.threshold + 1e-6) is True
        assert calib.decide(calib.threshold - 1e-6) is False

    def test_unfitted_raises(self):
        calib = NPCalibrator()
        with pytest.raises(RuntimeError):
            calib.decide(1.0)
        with pytest.raises(RuntimeError):
            calib.survival(1.0)


class TestEdgeCases:
    def test_insufficient_samples(self):
        calib = NPCalibrator()
        assert calib.fit([]) is False
        assert calib.fit([0.5]) is False
        assert calib.fit([0.5, 0.6]) is False
        assert not calib.is_fitted
        assert calib.to_dict() == {}

    def test_constant_scores(self):
        """全部相同分数 (退化分布): 不应崩溃。"""
        calib = NPCalibrator(epsilon=0.05)
        assert calib.fit([1.0] * 50)
        assert calib.threshold == pytest.approx(1.0)
        conf = calib.anomaly_confidence(2.0)
        assert 0.0 <= conf <= 1.0

    def test_negative_and_nan_filtered(self):
        calib = NPCalibrator(epsilon=0.05)
        scores = [0.1, 0.2, -0.5, float("nan"), 0.3, 0.4, float("inf")]
        assert calib.fit(scores)
        assert calib.n_samples == 4  # 负/nan/inf 被过滤

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            NPCalibrator(epsilon=0.0)
        with pytest.raises(ValueError):
            NPCalibrator(epsilon=1.0)
        with pytest.raises(ValueError):
            NPCalibrator(epsilon=-0.1)


class TestPersistence:
    def test_roundtrip(self):
        rng = np.random.default_rng(23)
        calib = NPCalibrator(epsilon=0.03)
        calib.fit(_skewed_scores(rng, 250))
        d = calib.to_dict()

        restored = NPCalibrator.from_dict(d)
        assert restored is not None
        assert restored.epsilon == pytest.approx(0.03)
        assert restored.threshold == pytest.approx(calib.threshold)
        assert restored.n_samples == calib.n_samples
        # 行为一致
        for s in [0.01, 0.1, 0.5, 1.0, 5.0]:
            assert restored.decide(s) == calib.decide(s)
            assert restored.anomaly_confidence(s) == pytest.approx(
                calib.anomaly_confidence(s), abs=1e-6)

    def test_from_dict_robust(self):
        assert NPCalibrator.from_dict({}) is None
        assert NPCalibrator.from_dict({"version": 999}) is None
        assert NPCalibrator.from_dict({"version": 1}) is None  # 缺字段
        # 样本不足的配置视为非法
        bad = {"version": 1, "epsilon": 0.02, "threshold": 1.0,
               "n_samples": 1, "scores_sorted": [1.0]}
        assert NPCalibrator.from_dict(bad) is None
