"""校准协议 (v1.4.0 §6.2) 单元测试

纯逻辑测试: recalibrate_product / recalibrate_engine / format_report_md,
用假引擎替代真实模型 (不依赖 torch/权重)。
端到端验收走 scripts/eval_acceptance.py calibration 模式。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import anomaly_bank, calibration_protocol as cp
from core.np_calibration import NPCalibrator, recalibrate_engine


# ─── 假引擎 / 假注册表 ─────────────────────────────────────
class _FakeCalib:
    def __init__(self, n: int, tau: float):
        self.is_fitted = True
        self.n_samples = n
        self.threshold = tau


class FakeEngine:
    def __init__(self, score_map: dict, epsilon=0.10,
                 n_cal_before=3, tau_before=1.0):
        self._np_epsilon = epsilon
        self._calibrated_threshold = tau_before
        self._train_scores = [tau_before] * n_cal_before
        self._np_calibrator = _FakeCalib(n_cal_before, tau_before)
        self._score_map = score_map
        self.saved: list[tuple[Path, str]] = []

    def is_ready(self):
        return True

    def infer(self, path, **kwargs):
        s = self._score_map.get(str(path))
        if s is None:
            return {"score": 0, "pred_label": "ERROR", "error": "读取失败"}
        return {"score": s, "pred_label": "OK"}

    def save_bank(self, path, product_name=""):
        self.saved.append((Path(path), product_name))


class FakeRegistry:
    def __init__(self, engines: dict):
        self._engines = engines

    def get(self, name):
        return self._engines.get(name)


# ─── fixtures ──────────────────────────────────────────────
def _make_images(tmp_path: Path, n: int, prefix="cal") -> list[str]:
    paths = []
    for i in range(n):
        p = tmp_path / f"{prefix}_{i:03d}.png"
        p.write_bytes(b"\x89PNG-fake")
        paths.append(str(p))
    return paths


@pytest.fixture
def patched_bank(tmp_path, monkeypatch):
    """把 bank 加载/路径与校准根目录都重定向到 tmp_path。"""
    loaded = {"patchcore": True, "dinov2": True}
    monkeypatch.setattr(anomaly_bank, "load_product_bank",
                        lambda reg, p: loaded["patchcore"])
    monkeypatch.setattr(anomaly_bank, "load_product_bank_dinov2",
                        lambda reg, p: loaded["dinov2"])
    monkeypatch.setattr(anomaly_bank, "bank_path",
                        lambda p: tmp_path / "banks" / f"{p}.npz")
    monkeypatch.setattr(anomaly_bank, "bank_path_dinov2",
                        lambda p: tmp_path / "banks_dv" / f"{p}.npz")
    monkeypatch.setattr(cp, "_CAL_ROOT", tmp_path / "calibration")
    return loaded


def _registry_with(tmp_path, n_cal_imgs, eps=0.10, n_before=3):
    """构造双引擎 registry, 分数 = 索引线性序列 (确定性)。"""
    cal = _make_images(tmp_path, n_cal_imgs)
    scores = {p: 1.0 + 0.1 * i for i, p in enumerate(cal)}
    pc = FakeEngine(scores, epsilon=eps, n_cal_before=n_before)
    dv = FakeEngine(scores, epsilon=eps, n_cal_before=n_before)
    return cal, pc, dv, FakeRegistry(
        {"anomalib": pc, "dinov2_anomaly": dv})


# ─── 输入校验 ──────────────────────────────────────────────
class TestInputValidation:
    def test_too_few_cal_images(self, tmp_path, patched_bank):
        cal = _make_images(tmp_path, 2)
        r = cp.recalibrate_product(FakeRegistry({}), "P", cal)
        assert not r["ok"]
        assert "至少 3 张" in r["error"]

    def test_no_bank_at_all(self, tmp_path, patched_bank):
        patched_bank["patchcore"] = False
        patched_bank["dinov2"] = False
        cal = _make_images(tmp_path, 5)
        r = cp.recalibrate_product(FakeRegistry({}), "P", cal)
        assert not r["ok"]
        assert "无可用特征库" in r["error"]

    def test_nonexistent_files_filtered(self, tmp_path, patched_bank):
        cal = _make_images(tmp_path, 3) + [str(tmp_path / "ghost.png")]
        _, pc, dv, reg = _registry_with(tmp_path, 3)
        r = cp.recalibrate_product(reg, "P", cal)
        assert r["ok"]
        assert r["sources"]["patchcore"]["n_scored"] == 3


# ─── 重标定主流程 ──────────────────────────────────────────
class TestRecalibrateProduct:
    def test_recalibrate_updates_tau_and_stage(self, tmp_path, patched_bank):
        cal, pc, dv, reg = _registry_with(tmp_path, 30)
        r = cp.recalibrate_product(reg, "P", cal)

        assert r["ok"]
        # tau = NPCalibrator 在同样分数上的 split-conformal 分位数
        exp = NPCalibrator(epsilon=0.10)
        exp.fit([1.0 + 0.1 * i for i in range(30)])
        for src in ("patchcore", "dinov2"):
            s = r["sources"][src]
            assert s["status"] == "ok"
            assert s["n_cal"] == 30
            assert s["tau"] == pytest.approx(exp.threshold)
            assert s["tau_before"] == 1.0
        assert r["n_cal_min"] == 30
        assert r["stage_before"] == 1   # n_cal_before=3
        assert r["stage_after"] == 2    # 30 → Stage 2

    def test_stage3_with_50_images(self, tmp_path, patched_bank):
        cal, pc, dv, reg = _registry_with(tmp_path, 50)
        r = cp.recalibrate_product(reg, "P", cal)
        assert r["n_cal_min"] == 50
        assert r["stage_after"] == 3

    def test_bank_saved_with_new_calib(self, tmp_path, patched_bank):
        cal, pc, dv, reg = _registry_with(tmp_path, 12)
        cp.recalibrate_product(reg, "P", cal)
        assert pc.saved and pc.saved[0][1] == "P"
        assert pc.saved[0][0] == tmp_path / "banks" / "P.npz"
        assert dv.saved[0][0] == tmp_path / "banks_dv" / "P.npz"
        # 引擎内部状态已更新
        assert pc._np_calibrator.n_samples == 12
        assert pc._calibrated_threshold == pytest.approx(
            pc._np_calibrator.threshold)

    def test_partial_bank_only_one_source(self, tmp_path, patched_bank):
        patched_bank["dinov2"] = False
        cal, pc, dv, reg = _registry_with(tmp_path, 15)
        r = cp.recalibrate_product(reg, "P", cal)
        assert r["ok"]
        assert r["sources"]["patchcore"]["status"] == "ok"
        assert "无特征库" in r["sources"]["dinov2"]["status"]
        # n_cal_min 只看成功源
        assert r["n_cal_min"] == 15

    def test_failed_images_counted(self, tmp_path, patched_bank):
        cal, pc, dv, reg = _registry_with(tmp_path, 10)
        # 2 张图打分失败 (score_map 缺失)
        for p in cal[:2]:
            del pc._score_map[p]
        r = cp.recalibrate_product(reg, "P", cal)
        s = r["sources"]["patchcore"]
        assert s["n_failed"] == 2
        assert s["n_scored"] == 8
        assert s["n_cal"] == 8


# ─── 校准集落盘 ────────────────────────────────────────────
class TestPersistence:
    def test_session_dir_and_manifest(self, tmp_path, patched_bank):
        cal, pc, dv, reg = _registry_with(tmp_path, 10)
        r = cp.recalibrate_product(reg, "P", cal)
        session = Path(r["session_dir"])
        assert session.is_dir()
        assert session.parent == tmp_path / "calibration" / "P"
        copied = list(session.glob("cal_*.png"))
        assert len(copied) == 10
        m = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
        assert m["product"] == "P"
        assert m["n_cal_images"] == 10
        assert m["stage_after"] == r["stage_after"]
        assert len(m["copied_files"]) == 10

    def test_unsafe_product_name_sanitized(self, tmp_path, patched_bank):
        cal, pc, dv, reg = _registry_with(tmp_path, 5)
        r = cp.recalibrate_product(reg, "P/A:B", cal)
        assert r["ok"]
        assert (tmp_path / "calibration" / "P_A_B").is_dir()


# ─── NG 回归 ───────────────────────────────────────────────
class TestNGRegression:
    def test_recall_computed(self, tmp_path, patched_bank):
        cal, pc, dv, reg = _registry_with(tmp_path, 30)
        tau = NPCalibrator(0.10)
        tau.fit([1.0 + 0.1 * i for i in range(30)])
        t = tau.threshold
        # 4 张 NG: 3 张超阈值, 1 张低于
        ng = _make_images(tmp_path, 4, prefix="ng")
        ng_scores = {ng[0]: t + 1.0, ng[1]: t + 0.5, ng[2]: t + 0.1,
                     ng[3]: t - 0.5}
        pc._score_map.update(ng_scores)
        dv._score_map.update(ng_scores)
        r = cp.recalibrate_product(reg, "P", cal, ng_image_paths=ng)
        assert r["sources"]["patchcore"]["ng_recall"] == 0.75
        assert r["ng_regression"]["n_defects"] == 4
        assert r["ng_regression"]["union_recall"] == 0.75


# ─── recalibrate_engine (底层助手) ─────────────────────────
class TestRecalibrateEngine:
    def test_success_updates_state(self):
        eng = FakeEngine({}, tau_before=0.5, n_cal_before=3)
        scores = [1.0 + 0.05 * i for i in range(20)]
        res = recalibrate_engine(eng, scores)
        assert res["ok"]
        assert res["n"] == 20
        assert res["tau_before"] == 0.5
        exp = NPCalibrator(0.10)
        assert exp.fit(scores)
        assert res["tau"] == pytest.approx(exp.threshold)
        assert eng._calibrated_threshold == pytest.approx(exp.threshold)
        assert eng._np_calibrator.n_samples == 20
        assert eng._train_scores == scores

    def test_insufficient_scores_keep_state(self):
        eng = FakeEngine({}, tau_before=0.5, n_cal_before=3)
        res = recalibrate_engine(eng, [1.0, 2.0])
        assert not res["ok"]
        assert "样本不足" in res["error"]
        # 原状态不动
        assert eng._calibrated_threshold == 0.5
        assert eng._np_calibrator.n_samples == 3

    def test_epsilon_override(self):
        eng = FakeEngine({}, epsilon=0.10)
        scores = [1.0 + i for i in range(40)]
        res = recalibrate_engine(eng, scores, epsilon=0.05)
        assert res["epsilon"] == 0.05
        assert eng._np_calibrator.epsilon == 0.05


# ─── 报告 ──────────────────────────────────────────────────
class TestReport:
    def test_report_contains_key_fields(self, tmp_path, patched_bank):
        cal, pc, dv, reg = _registry_with(tmp_path, 30)
        r = cp.recalibrate_product(reg, "P", cal)
        md = cp.format_report_md(r)
        assert "校准协议验收报告" in md
        assert "PatchCore" in md and "DINOv2" in md
        assert "Stage" in md
        assert "ε" in md

    def test_report_error_path(self):
        md = cp.format_report_md({"ok": False, "error": "无可用特征库"})
        assert "无可用特征库" in md

    def test_report_stage1_hint(self, tmp_path, patched_bank):
        cal, pc, dv, reg = _registry_with(tmp_path, 5)
        r = cp.recalibrate_product(reg, "P", cal)
        md = cp.format_report_md(r)
        assert "Stage 1" in md
        assert "指引" in md
