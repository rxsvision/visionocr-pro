"""DINOv2 异常检测引擎单元测试

逻辑层测试: 用合成特征 + mock _extract_features 覆盖
train/infer/save_bank/load_bank 全链路, 不依赖真实模型权重 (保持套件快速)。
真实模型验证走 scripts/smoke_dinov2_anomaly.py 与 scripts/eval_dinov2_anomaly.py。
"""
from __future__ import annotations

import numpy as np
import pytest

from engines.base import EngineState
from engines.vision.dinov2_anomaly import DINOv2AnomalyEngine


@pytest.fixture
def cfg():
    return {
        "device": "cpu",
        "qc": {"dinov2": {"input_size": 518, "pca_dim": 32,
                          "n_etalons": 4, "np_epsilon": 0.05}},
    }


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def make_engine(cfg, rng=None, n_images=15, patches_per_img=1369,
                anomaly_shift=0.0):
    """构造 mock 特征提取的引擎 (正常=标准高斯, 异常=均值偏移)。

    特征按图像名派生确定性种子, 与调用顺序/实例无关,
    保证 save/load 往返一致性测试的前提成立。
    """
    import hashlib

    eng = DINOv2AnomalyEngine(cfg)

    def fake_extract(image):
        if image == "__bad__":
            return None
        seed = int(hashlib.md5(str(image).encode()).hexdigest()[:8], 16)
        local = np.random.default_rng(seed)
        n = patches_per_img
        x = local.standard_normal((n, 384)).astype(np.float32)
        if anomaly_shift:
            x[:, :10] += anomaly_shift
        return x

    eng._extract_features = fake_extract
    eng._model = object()  # 占位, mock 路径不会触碰
    eng.state = EngineState.READY
    return eng


# ─── 配置解析 ────────────────────────────────────────────────
class TestConfig:
    def test_input_size_snapped_to_multiple_of_14(self, cfg):
        cfg["qc"]["dinov2"]["input_size"] = 500
        eng = DINOv2AnomalyEngine(cfg)
        assert eng._img_size == 490  # 500 // 14 * 14

    def test_defaults(self):
        eng = DINOv2AnomalyEngine({"device": "cpu"})
        assert eng._model_id == "facebook/dinov2-small"
        assert eng._img_size % 14 == 0
        assert eng._np_epsilon == 0.02

    def test_meta(self, cfg):
        eng = DINOv2AnomalyEngine(cfg)
        assert eng.meta.name == "dinov2_anomaly"
        assert eng.meta.license == "Apache-2.0"
        assert eng.meta.category == "vision"


# ─── 训练 + 推理链路 (mock 特征) ─────────────────────────────
class TestTrainInfer:
    def test_train_builds_bank_and_np_threshold(self, cfg, rng):
        eng = make_engine(cfg, rng)
        meta = eng.train([f"img_{i}.jpg" for i in range(15)])
        assert not meta.get("error")
        assert eng.has_bank
        # n_images = 建库图数 (15 张中留出 20%=3 张作 NP 校准集)
        assert meta["n_images"] == 12
        assert meta["n_etalons"] >= 1
        assert eng._calibrated_threshold is not None
        assert eng._np_calibrator is not None and \
            eng._np_calibrator.is_fitted

    def test_normal_scores_below_anomaly_scores(self, cfg, rng):
        eng = make_engine(cfg, rng)
        eng.train([f"img_{i}.jpg" for i in range(15)])

        normal_scores = [eng.infer(f"n{i}.jpg")["score"] for i in range(5)]
        eng_anom = make_engine(cfg, np.random.default_rng(7),
                               anomaly_shift=3.0)
        # 复用已训练的 PCA/GMM/阈值
        eng_anom._pca, eng_anom._gmm = eng._pca, eng._gmm
        eng_anom._calibrated_threshold = eng._calibrated_threshold
        eng_anom._np_calibrator = eng._np_calibrator
        anom_scores = [eng_anom.infer(f"a{i}.jpg")["score"]
                       for i in range(5)]
        assert min(anom_scores) > max(normal_scores), \
            f"异常分数未分离: {anom_scores} vs {normal_scores}"

    def test_infer_result_contract(self, cfg, rng):
        eng = make_engine(cfg, rng)
        eng.train([f"img_{i}.jpg" for i in range(15)])
        r = eng.infer("test.jpg")
        for key in ("score", "anomaly_map", "pred_label", "grid_size",
                    "threshold_used"):
            assert key in r
        assert r["pred_label"] in ("OK", "NG")
        assert r["anomaly_map"].shape == (37, 37)  # 1369 patches → 37x37
        assert r["anomaly_map"].min() >= 0 and r["anomaly_map"].max() <= 1.001
        # NP 校准附加字段
        assert "calibrated_score" in r and "np_p_value" in r
        assert 0.0 <= r["calibrated_score"] <= 1.0
        assert 0.0 <= r["np_p_value"] <= 1.0
        # 判定与阈值一致
        assert r["pred_label"] == ("NG" if r["score"] > r["threshold_used"]
                                   else "OK")

    def test_train_all_invalid_images(self, cfg, rng):
        eng = make_engine(cfg, rng)
        eng._extract_features = lambda image: None
        result = eng.train(["a.jpg", "b.jpg"])
        assert result.get("error") == "无有效图像"
        assert not eng.has_bank

    def test_infer_without_bank(self, cfg):
        eng = DINOv2AnomalyEngine(cfg)
        eng.state = EngineState.READY
        r = eng.infer("x.jpg")
        assert r["pred_label"] == "ERROR"
        assert "记忆库" in r["error"] or "注册" in r["error"]

    def test_infer_not_ready(self, cfg):
        eng = DINOv2AnomalyEngine(cfg)
        r = eng.infer("x.jpg")
        assert r["pred_label"] == "ERROR"


# ─── 持久化往返 ─────────────────────────────────────────────
class TestPersistence:
    def test_save_load_roundtrip_scores_identical(self, cfg, rng, tmp_path):
        eng = make_engine(cfg, rng)
        eng.train([f"img_{i}.jpg" for i in range(15)])
        probe = "probe.jpg"
        s1 = eng.infer(probe)["score"]

        bank = tmp_path / "bank.npz"
        eng.save_bank(bank, product_name="test")
        assert bank.exists()

        eng2 = make_engine(cfg, np.random.default_rng(42))
        assert eng2.load_bank(bank)
        assert eng2.has_bank
        assert eng2._np_calibrator is not None
        assert eng2._np_calibrator.is_fitted
        s2 = eng2.infer(probe)["score"]
        assert abs(s1 - s2) < 1e-3, f"往返分数漂移: {s1} vs {s2}"

    def test_load_missing_file(self, cfg, tmp_path):
        eng = DINOv2AnomalyEngine(cfg)
        assert eng.load_bank(tmp_path / "nope.npz") is False

    def test_save_without_bank_is_noop(self, cfg, tmp_path):
        eng = DINOv2AnomalyEngine(cfg)
        bank = tmp_path / "empty.npz"
        eng.save_bank(bank)
        assert not bank.exists()


# ─── 注册表集成 ─────────────────────────────────────────────
class TestRegistryIntegration:
    def test_manifest_contains_dinov2(self):
        from engines.registry import EngineRegistry
        assert ("engines.vision.dinov2_anomaly", "DINOv2AnomalyEngine") \
            in EngineRegistry.ENGINE_MANIFEST

    def test_unload_resets_state(self, cfg, rng):
        eng = make_engine(cfg, rng)
        eng.train([f"img_{i}.jpg" for i in range(15)])
        eng.unload()
        assert eng.state == EngineState.UNLOADED
        assert not eng.has_bank
        assert eng._np_calibrator is None


# ─── §5.1 PixOOD 思想借鉴 (P1 重初始化 / P4 局部NP) ─────────
class TestPixOODUpgrade:
    def test_component_logprob_matches_sklearn(self, cfg, rng):
        from sklearn.mixture import GaussianMixture
        eng = make_engine(cfg, rng)
        eng.train([f"img_{i}.jpg" for i in range(15)])
        X = rng.standard_normal((500, eng._pca.n_components_))
        ref = eng._gmm._estimate_weighted_log_prob(X)
        ours = eng._gmm_component_weighted_logprob(X)
        assert np.allclose(ref, ours, atol=1e-8)

    def test_reinit_dead_etalons_runs_and_keeps_bank(self, cfg, rng):
        # dead_weight_frac=3.0 → min_w=0.75, 强制每轮都有"死分量"重播种
        cfg["qc"]["dinov2"]["reinit_dead_etalons"] = True
        cfg["qc"]["dinov2"]["dead_weight_frac"] = 3.0
        cfg["qc"]["dinov2"]["reinit_rounds"] = 2
        eng = make_engine(cfg, rng)
        meta = eng.train([f"img_{i}.jpg" for i in range(15)])
        assert not meta.get("error")
        assert meta["reinit_dead"] is True
        assert eng.has_bank
        # 重初始化后推理链路正常
        r = eng.infer("probe.jpg")
        assert r["pred_label"] in ("OK", "NG")

    def test_per_etalon_np_stats_fitted(self, cfg, rng):
        cfg["qc"]["dinov2"]["per_etalon_np"] = True
        eng = make_engine(cfg, rng)
        meta = eng.train([f"img_{i}.jpg" for i in range(15)])
        assert meta["per_etalon_np"] is True
        K = eng._gmm.n_components
        assert eng._etalon_np_mu.shape == (K,)
        assert eng._etalon_np_sigma.shape == (K,)
        assert (eng._etalon_np_sigma > 0).all()

    def test_per_etalon_np_preserves_separation(self, cfg, rng):
        cfg["qc"]["dinov2"]["per_etalon_np"] = True
        eng = make_engine(cfg, rng)
        eng.train([f"img_{i}.jpg" for i in range(15)])
        normal_scores = [eng.infer(f"n{i}.jpg")["score"] for i in range(5)]
        eng_anom = make_engine(cfg, np.random.default_rng(7),
                               anomaly_shift=3.0)
        eng_anom._pca, eng_anom._gmm = eng._pca, eng._gmm
        eng_anom._calibrated_threshold = eng._calibrated_threshold
        eng_anom._np_calibrator = eng._np_calibrator
        eng_anom._etalon_np_mu = eng._etalon_np_mu
        eng_anom._etalon_np_sigma = eng._etalon_np_sigma
        anom_scores = [eng_anom.infer(f"a{i}.jpg")["score"]
                       for i in range(5)]
        assert min(anom_scores) > max(normal_scores), \
            f"局部归一化后异常分数未分离: {anom_scores} vs {normal_scores}"

    def test_per_etalon_np_persistence_roundtrip(self, cfg, rng, tmp_path):
        cfg["qc"]["dinov2"]["per_etalon_np"] = True
        eng = make_engine(cfg, rng)
        eng.train([f"img_{i}.jpg" for i in range(15)])
        probe = "probe.jpg"
        s1 = eng.infer(probe)["score"]
        bank = tmp_path / "bank.npz"
        eng.save_bank(bank, product_name="t")

        eng2 = make_engine(cfg, np.random.default_rng(42))
        assert eng2.load_bank(bank)
        assert eng2._etalon_np_mu is not None
        assert np.allclose(eng2._etalon_np_mu, eng._etalon_np_mu)
        assert np.allclose(eng2._etalon_np_sigma, eng._etalon_np_sigma)
        assert abs(eng2.infer(probe)["score"] - s1) < 1e-3

    def test_old_bank_without_stats_falls_back(self, cfg, rng, tmp_path):
        # 默认关闭 per_etalon_np → bank 内无统计键, 加载后走原始 NLL
        eng = make_engine(cfg, rng)
        eng.train([f"img_{i}.jpg" for i in range(15)])
        s1 = eng.infer("probe.jpg")["score"]
        bank = tmp_path / "bank.npz"
        eng.save_bank(bank, product_name="t")

        eng2 = make_engine(cfg, np.random.default_rng(42))
        assert eng2.load_bank(bank)
        assert eng2._etalon_np_mu is None
        assert abs(eng2.infer("probe.jpg")["score"] - s1) < 1e-3

    def test_unload_resets_etalon_stats(self, cfg, rng):
        cfg["qc"]["dinov2"]["per_etalon_np"] = True
        eng = make_engine(cfg, rng)
        eng.train([f"img_{i}.jpg" for i in range(15)])
        eng.unload()
        assert eng._etalon_np_mu is None
        assert eng._etalon_np_sigma is None
