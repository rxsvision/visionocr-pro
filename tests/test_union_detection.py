"""run_union_detection 四源 Union 检测测试 (审查项: 此前零覆盖)。

用假引擎 + 真 EngineRegistry 验证: OR 语义、分阶段融合 REVIEW 契约、
源门控 (prompt/YOLO 产品权重)、全跳过安全守卫、异常输入。
"""
import numpy as np
import pytest

from core.imutils import imwrite_unicode
from engines.base import BaseEngine, EngineMeta, EngineState
from engines.registry import EngineRegistry
from core.defect_detector import run_union_detection


# ─── 假引擎 ──────────────────────────────────────────────────

class _FakeCalibrator:
    def __init__(self, n_samples):
        self.is_fitted = True
        self.n_samples = n_samples
        self.threshold = 0.5
        self.epsilon = 0.1


class SurfaceEngine(BaseEngine):
    """模拟 PatchCore / DINOv2: pred_label + anomaly_map。"""

    def __init__(self, config, name, verdict="OK", n_cal=None):
        super().__init__(config)
        self._meta = EngineMeta(name=name, display_name=name,
                                category="vision", vram_gb=0.1)
        self.verdict = verdict
        self.has_bank = True
        if n_cal:
            self._np_calibrator = _FakeCalibrator(n_cal)

    @property
    def meta(self):
        return self._meta

    def load(self):
        self.state = EngineState.READY

    def unload(self):
        self.state = EngineState.UNLOADED

    def infer(self, image_path, **kw):
        return {
            "pred_label": self.verdict,
            "score": 0.9 if self.verdict == "NG" else 0.1,
            "anomaly_map": np.zeros((32, 32), dtype=np.float32),
            "threshold_used": 0.5,
        }


class DinoEngine(BaseEngine):
    """模拟 Grounding DINO: boxes/labels/scores。"""

    def __init__(self, config, detections=0):
        super().__init__(config)
        self._meta = EngineMeta(name="grounding_dino",
                                display_name="GDINO", category="vision",
                                vram_gb=0.1)
        self.detections = detections

    @property
    def meta(self):
        return self._meta

    def load(self):
        self.state = EngineState.READY

    def unload(self):
        self.state = EngineState.UNLOADED

    def infer(self, image_path, prompt="", threshold=0.3):
        boxes = [[10, 10, 60, 60]] * self.detections
        return {
            "boxes": boxes,
            "labels": ["scratch"] * self.detections,
            "scores": [0.9] * self.detections,
        }


class YoloEngine(BaseEngine):
    """模拟 YOLO: load_for_product 产品门控。"""

    def __init__(self, config, has_weights=False, count=0):
        super().__init__(config)
        self._meta = EngineMeta(name="yolo_defect", display_name="YOLO",
                                category="vision", vram_gb=0.1)
        self.has_weights = has_weights
        self.count = count

    @property
    def meta(self):
        return self._meta

    def load(self):
        self.state = EngineState.READY

    def unload(self):
        self.state = EngineState.UNLOADED

    def load_for_product(self, product_name):
        if self.has_weights and product_name:
            self.state = EngineState.READY
            return True
        return False

    def infer(self, image_path):
        boxes = [[5, 5, 20, 20]] * self.count
        return {"boxes": boxes, "labels": ["hole"] * self.count,
                "scores": [0.8] * self.count, "count": self.count}


# ─── fixtures ────────────────────────────────────────────────

@pytest.fixture
def image_path(tmp_path):
    img = np.full((120, 160, 3), 200, dtype=np.uint8)
    p = str(tmp_path / "sample.png")
    assert imwrite_unicode(p, img)
    return p


@pytest.fixture
def registry():
    reg = EngineRegistry({"vram": {"max_budget_gb": 8.0,
                                   "idle_unload_sec": 0}})
    yield reg
    reg.shutdown()


def _run(registry, image_path, prompt="", config=None, product_name=""):
    return run_union_detection(registry, image_path, prompt=prompt,
                               config=config, product_name=product_name)


# ─── 用例 ────────────────────────────────────────────────────

def test_all_ok(image_path, registry):
    registry.register(SurfaceEngine({}, "anomalib", verdict="OK"))
    registry.register(SurfaceEngine({}, "dinov2_anomaly", verdict="OK"))
    registry.register(DinoEngine({}, detections=0))
    r = _run(registry, image_path, prompt="划痕")
    assert r["verdict"] == "OK"
    assert r["ng_sources"] == []


def test_or_semantics_any_source_ng(image_path, registry):
    """Stage 1 (无校准) 保守 OR: 单源 NG → 终判 NG。"""
    registry.register(SurfaceEngine({}, "anomalib", verdict="NG"))
    registry.register(SurfaceEngine({}, "dinov2_anomaly", verdict="OK"))
    r = _run(registry, image_path)
    assert r["verdict"] == "NG"
    assert r["ng_sources"] == ["patchcore"]


def test_multi_source_ng(image_path, registry):
    registry.register(SurfaceEngine({}, "anomalib", verdict="NG"))
    registry.register(SurfaceEngine({}, "dinov2_anomaly", verdict="NG"))
    registry.register(DinoEngine({}, detections=2))
    r = _run(registry, image_path, prompt="划痕")
    assert r["verdict"] == "NG"
    assert set(r["ng_sources"]) == {"patchcore", "dinov2", "dino"}


def test_review_contract_single_calibrated_source(image_path, registry):
    """Stage 2: 双校准源参与但仅单源判 NG → REVIEW 黄牌 (不静默放行)。"""
    registry.register(SurfaceEngine({}, "anomalib", verdict="NG", n_cal=15))
    registry.register(SurfaceEngine({}, "dinov2_anomaly", verdict="OK",
                                    n_cal=15))
    r = _run(registry, image_path)
    assert r["verdict"] == "REVIEW"


def test_dual_calibrated_confirmation(image_path, registry):
    """Stage 2: 双校准源互证 → 自主 NG。"""
    registry.register(SurfaceEngine({}, "anomalib", verdict="NG", n_cal=15))
    registry.register(SurfaceEngine({}, "dinov2_anomaly", verdict="NG",
                                    n_cal=15))
    r = _run(registry, image_path)
    assert r["verdict"] == "NG"


def test_dino_skipped_without_prompt(image_path, registry):
    registry.register(DinoEngine({}, detections=5))
    r = _run(registry, image_path, prompt="")  # 空 prompt → 跳过 DINO
    assert r["verdict"] == "OK"
    assert "dino" not in r["ng_sources"]


def test_yolo_gated_by_product_weights(image_path, registry):
    """无产品权重 → YOLO 跳过, 即使 count>0 也不得贡献 NG。"""
    registry.register(YoloEngine({}, has_weights=False, count=3))
    r = _run(registry, image_path, product_name="产品A")
    assert "yolo" not in r["ng_sources"]


def test_yolo_active_with_weights(image_path, registry):
    registry.register(YoloEngine({}, has_weights=True, count=2))
    r = _run(registry, image_path, product_name="产品A")
    assert "yolo" in r["ng_sources"]
    assert r["verdict"] == "NG"


def test_all_engines_skipped_safe_guard(image_path, registry):
    """全引擎缺失: 不崩溃, verdict=OK 但结果可识别为空检测。"""
    r = _run(registry, image_path)
    assert r["verdict"] == "OK"
    assert r["ng_sources"] == []
    assert r["image"] is not None


def test_invalid_image_returns_error(registry):
    registry.register(SurfaceEngine({}, "anomalib"))
    r = _run(registry, "/nonexistent/路径.png")
    assert r["verdict"] == "ERROR"


def test_source_error_does_not_crash(image_path, registry):
    """单源推理报错 → 该源置 None, 其余源正常融合。"""

    class BrokenEngine(SurfaceEngine):
        def infer(self, image_path, **kw):
            raise RuntimeError("GPU OOM")

    registry.register(BrokenEngine({}, "anomalib", verdict="NG"))
    registry.register(SurfaceEngine({}, "dinov2_anomaly", verdict="NG"))
    r = _run(registry, image_path)
    assert r["patchcore"] is None
    assert "dinov2" in r["ng_sources"]
    assert r["verdict"] == "NG"


def test_fusion_mode_or_fallback(image_path, registry):
    """config fusion.mode=or → 回退 v1.3.0 纯 OR。"""
    registry.register(SurfaceEngine({}, "anomalib", verdict="NG", n_cal=15))
    registry.register(SurfaceEngine({}, "dinov2_anomaly", verdict="OK",
                                    n_cal=15))
    r = _run(registry, image_path,
             config={"qc": {"union": {"fusion": {"mode": "or"}}}})
    assert r["verdict"] == "NG"  # mode=or 下不产生 REVIEW
