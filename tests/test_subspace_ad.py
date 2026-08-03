"""SubspaceAD 快速换线引擎单元测试

逻辑层测试: 用合成低秩特征 + mock _extract_features 覆盖
train/infer/save_bank/load_bank 全链路 (含快速换线增广路径),
不依赖真实模型权重 (保持套件快速)。
真实模型验证走 scripts/eval_acceptance.py subspacead 模式。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from engines.base import EngineState
from engines.vision.subspace_ad import SubspaceADEngine


@pytest.fixture
def cfg():
    # 小网格 (56px → 4x4 patches) 加速测试; aug_count=10 + cal_frac=0.3
    # 使 1-shot 时 n_cal=3 恰好满足 NP 校准最小样本数
    return {
        "device": "cpu",
        "qc": {"subspacead": {
            "input_size": 56, "layers": [-4, -5], "pca_ev": 0.99,
            "aug_count": 10, "fast_max_images": 4, "cal_frac": 0.3,
            "np_epsilon": 0.05, "seed": 42,
        }},
    }


def make_engine(cfg, n_patches=16, dim=64, rank=8, anomaly_shift=0.0):
    """构造 mock 特征提取的引擎。

    正常特征 = rank 维低秩高斯 (PCA 子空间可完全解释, 残差≈0);
    anomaly_shift 沿随机固定方向注入子空间外偏移 → 残差升高。
    特征种子派生自图像内容/路径, 与实例无关 (save/load 往返前提)。
    """
    eng = SubspaceADEngine(cfg)
    base_rng = np.random.default_rng(7)
    basis = base_rng.standard_normal((rank, dim)).astype(np.float32)
    shift_dir = base_rng.standard_normal(dim).astype(np.float32)

    def fake_extract(image):
        if hasattr(image, "tobytes"):  # PIL (快速模式增广视图)
            seed = int(hashlib.md5(
                image.tobytes()[:64]).hexdigest()[:8], 16)
        else:
            if image == "__bad__":
                return None
            seed = int(hashlib.md5(str(image).encode()).hexdigest()[:8], 16)
        local = np.random.default_rng(seed)
        z = local.standard_normal((n_patches, rank)).astype(np.float32)
        x = z @ basis
        if anomaly_shift:
            x = x + anomaly_shift * shift_dir
        return x

    eng._extract_features = fake_extract
    eng._model = object()  # 占位, mock 路径不会触碰
    eng.state = EngineState.READY
    return eng


def write_pngs(tmp_path, n, size=56):
    """生成确定性合成 PNG (快速模式需要真实可打开的图像文件)。"""
    from PIL import Image
    rng = np.random.default_rng(2026)
    paths = []
    for i in range(n):
        arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        p = tmp_path / f"ok_{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    return paths


# ─── 配置解析 ────────────────────────────────────────────────
class TestConfig:
    def test_defaults(self):
        eng = SubspaceADEngine({"device": "cpu"})
        assert eng.meta.name == "subspace_ad"
        assert eng.meta.license == "Apache-2.0"
        assert eng._img_size == 448
        assert eng._layers == [-4, -5]
        assert eng._pca_ev == 0.99
        assert eng._aug_count == 30
        assert eng._fast_max == 4

    def test_input_size_snapped_to_multiple_of_14(self):
        cfg = {"qc": {"subspacead": {"input_size": 50}}}
        eng = SubspaceADEngine(cfg)
        assert eng._img_size == 42

    def test_meta_not_resident(self):
        # 快速换线通道按需加载, 不占常驻显存预算
        eng = SubspaceADEngine({"device": "cpu"})
        assert eng.meta.resident is False


# ─── 快速换线模式 ────────────────────────────────────────────
class TestFastMode:
    def test_one_shot_builds_bank_with_augmentation(self, cfg, tmp_path):
        eng = make_engine(cfg)
        paths = write_pngs(tmp_path, 1)
        meta = eng.train(paths)
        assert not meta.get("error")
        assert eng.has_bank
        assert meta["mode"] == "fast"
        assert meta["n_images"] == 1
        # aug_count=10, cal_frac=0.3 → 7 增广入池, 3 留作校准
        assert meta["n_augmented"] == 7
        assert eng._calibrated_threshold is not None
        assert eng._np_calibrator is not None and \
            eng._np_calibrator.is_fitted
        assert eng._np_calibrator.n_samples == 3

    def test_four_shot_still_fast(self, cfg, tmp_path):
        eng = make_engine(cfg)
        meta = eng.train(write_pngs(tmp_path, 4))
        assert meta["mode"] == "fast"
        assert meta["n_images"] == 4

    def test_five_shot_switches_to_standard_no_aug(self, cfg, tmp_path):
        eng = make_engine(cfg)
        paths = write_pngs(tmp_path, 5)
        calls = []
        orig = eng._extract_features

        def counting(image):
            calls.append(image)
            return orig(image)
        eng._extract_features = counting
        meta = eng.train(paths)
        assert meta["mode"] == "standard"
        assert meta["n_augmented"] == 0
        # 标准模式无增广: 建库 5 次 + 校准评分 5 次 (n<10 时两者同集)
        assert len(calls) == 10

    def test_one_shot_no_aug_has_no_threshold(self, cfg, tmp_path):
        cfg["qc"]["subspacead"]["aug_count"] = 0
        eng = make_engine(cfg)
        meta = eng.train(write_pngs(tmp_path, 1))
        assert eng.has_bank
        # 无增广 → 校准样本 0 → 阈值不可用
        assert eng._calibrated_threshold is None
        r = eng.infer(write_pngs(tmp_path, 1)[0])
        # 快速模式不给自主判定, 仅输出分数供人工复核
        assert r["pred_label"] == "REVIEW"
        assert r["review_required"] is True
        assert r["threshold_used"] is None

    def test_fast_mode_advisory_contract(self, cfg, tmp_path):
        """快速换线 = 辅助提示通道: 自校准偏乐观 (KolektorSDD 实测
        tau≈0.14 vs 真实正常件≈0.53), 禁止自主 OK/NG 判定。"""
        eng = make_engine(cfg)
        meta = eng.train(write_pngs(tmp_path, 1))
        assert meta["mode"] == "fast"
        r = eng.infer(write_pngs(tmp_path, 1)[0])
        assert r["pred_label"] == "REVIEW"
        assert r["review_required"] is True
        assert r["calibration_mode"] == "fast_selfcal"
        assert "score" in r and r["anomaly_map"] is not None


# ─── 标准模式 + PCA ─────────────────────────────────────────
class TestStandardMode:
    def test_train_standard_with_holdout(self, cfg):
        eng = make_engine(cfg)
        meta = eng.train([f"img_{i}.jpg" for i in range(15)])
        assert meta["mode"] == "standard"
        assert meta["n_images"] == 12  # 15 留 20%=3 张校准
        assert eng._np_calibrator is not None and \
            eng._np_calibrator.is_fitted

    def test_pca_k_selected_by_explained_variance(self, cfg):
        eng = make_engine(cfg)  # 合成特征秩=8
        meta = eng.train([f"img_{i}.jpg" for i in range(15)])
        # EV 0.99 截断: k 应远小于 dim=64, 接近真实秩 8
        assert meta["pca_k"] <= 12
        assert meta["pca_ev_achieved"] >= 0.99

    def test_train_empty_paths(self, cfg):
        eng = make_engine(cfg)
        assert eng.train([]).get("error")


# ─── 推理契约与分离性 ───────────────────────────────────────
class TestInfer:
    def test_result_contract(self, cfg):
        eng = make_engine(cfg)
        eng.train([f"img_{i}.jpg" for i in range(15)])
        r = eng.infer("probe.jpg")
        for key in ("score", "anomaly_map", "pred_label", "grid_size",
                    "threshold_used"):
            assert key in r
        assert r["pred_label"] in ("OK", "NG")
        assert r["anomaly_map"].shape == (4, 4)  # 56px → 4x4 grid
        assert r["anomaly_map"].min() >= 0 and r["anomaly_map"].max() <= 1.001
        assert "calibrated_score" in r and "np_p_value" in r

    def test_anomaly_scores_separate(self, cfg):
        eng = make_engine(cfg)
        eng.train([f"img_{i}.jpg" for i in range(15)])
        normal = [eng.infer(f"n{i}.jpg")["score"] for i in range(5)]

        eng_a = make_engine(cfg, anomaly_shift=3.0)
        eng_a._sub_mu, eng_a._sub_components = \
            eng._sub_mu, eng._sub_components
        eng_a._calibrated_threshold = eng._calibrated_threshold
        eng_a._np_calibrator = eng._np_calibrator
        anom = [eng_a.infer(f"a{i}.jpg")["score"] for i in range(5)]
        assert min(anom) > max(normal), \
            f"残差分数未分离: {anom} vs {normal}"
        assert all(r == "NG" for r in
                   [eng_a.infer(f"a{i}.jpg")["pred_label"]
                    for i in range(5)])

    def test_infer_without_bank(self, cfg):
        eng = SubspaceADEngine(cfg)
        eng.state = EngineState.READY
        r = eng.infer("x.jpg")
        assert r["pred_label"] == "ERROR"
        assert "注册" in r["error"]

    def test_infer_not_ready(self, cfg):
        eng = SubspaceADEngine(cfg)
        assert eng.infer("x.jpg")["pred_label"] == "ERROR"


# ─── 持久化往返 ─────────────────────────────────────────────
class TestPersistence:
    def test_save_load_roundtrip_scores_identical(self, cfg, tmp_path):
        eng = make_engine(cfg)
        eng.train([f"img_{i}.jpg" for i in range(15)])
        probe = "probe.jpg"
        s1 = eng.infer(probe)["score"]

        bank = tmp_path / "bank.npz"
        eng.save_bank(bank, product_name="test")
        assert bank.exists()

        eng2 = make_engine(cfg)
        assert eng2.load_bank(bank)
        assert eng2.has_bank
        assert eng2._np_calibrator is not None
        s2 = eng2.infer(probe)["score"]
        assert abs(s1 - s2) < 1e-3, f"往返分数漂移: {s1} vs {s2}"

    def test_load_missing_file(self, cfg, tmp_path):
        eng = SubspaceADEngine(cfg)
        assert eng.load_bank(tmp_path / "nope.npz") is False

    def test_save_without_bank_is_noop(self, cfg, tmp_path):
        eng = SubspaceADEngine(cfg)
        eng.save_bank(tmp_path / "empty.npz")
        assert not (tmp_path / "empty.npz").exists()


# ─── 注册表集成 ─────────────────────────────────────────────
class TestRegistryIntegration:
    def test_manifest_contains_subspace_ad(self):
        from engines.registry import EngineRegistry
        assert ("engines.vision.subspace_ad", "SubspaceADEngine") \
            in EngineRegistry.ENGINE_MANIFEST

    def test_unload_resets_state(self, cfg):
        eng = make_engine(cfg)
        eng.train([f"img_{i}.jpg" for i in range(15)])
        eng.unload()
        assert eng.state == EngineState.UNLOADED
        assert not eng.has_bank
        assert eng._np_calibrator is None
