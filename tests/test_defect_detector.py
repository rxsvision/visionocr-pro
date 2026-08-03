"""run_detection / run_anomaly_detection 主力生产路径聚焦测试。

覆盖缺口说明: 此前两函数仅有 run_union_detection 的间接覆盖
(run_anomaly_detection 实际零覆盖)。本文件用假引擎 + 真 EngineRegistry
验证: 判定契约、错误降级分支、尺寸过滤、阈值透传、ndarray 直通,
以及 v1.5.0 拆分后的向后兼容 re-export 面不变。
"""
import numpy as np
import pytest

import core.defect_detector as dd
from core.imutils import imwrite_unicode
from core.recipes import DEFAULT_PROMPT, translate_prompt
from engines.base import BaseEngine, EngineMeta, EngineState
from engines.registry import EngineRegistry
from core.defect_detector import run_detection, run_anomaly_detection


# ─── 假引擎 ──────────────────────────────────────────────────

class FakeDino(BaseEngine):
    """模拟 Grounding DINO: boxes/labels/scores 输出。"""

    def __init__(self, config, boxes=None, labels=None, scores=None,
                 fail_load=False, raise_load=False, error=None):
        super().__init__(config)
        self._meta = EngineMeta(name="grounding_dino", display_name="GDINO",
                                category="vision", vram_gb=0.1)
        self.boxes = boxes or []
        self.labels = labels or []
        self.scores = scores or []
        self.fail_load = fail_load
        self.raise_load = raise_load
        self.error = error
        self.received = None
        self.received_prompt = None
        self.received_threshold = None

    @property
    def meta(self):
        return self._meta

    def load(self):
        if self.raise_load:
            raise RuntimeError("CUDA 初始化失败")
        if self.fail_load:
            return  # 静默加载失败: 不进入 READY
        self.state = EngineState.READY

    def unload(self):
        self.state = EngineState.UNLOADED

    def infer(self, image, prompt="", threshold=0.3):
        self.received = image
        self.received_prompt = prompt
        self.received_threshold = threshold
        if self.error:
            return {"error": self.error}
        return {"boxes": self.boxes, "labels": self.labels,
                "scores": self.scores}


class FakeAnomalib(BaseEngine):
    """模拟 PatchCore: score/anomaly_map/pred_label 输出。"""

    def __init__(self, config, pred="OK", score=0.1, has_bank=True,
                 fail_load=False, raise_load=False, error=None):
        super().__init__(config)
        self._meta = EngineMeta(name="anomalib", display_name="PatchCore",
                                category="vision", vram_gb=0.1)
        self.pred = pred
        self.score = score
        self.has_bank = has_bank
        self.fail_load = fail_load
        self.raise_load = raise_load
        self.error = error
        self.received = None
        self.received_kwargs = None

    @property
    def meta(self):
        return self._meta

    def load(self):
        if self.raise_load:
            raise RuntimeError("CUDA 初始化失败")
        if self.fail_load:
            return  # 静默加载失败: 不进入 READY
        self.state = EngineState.READY

    def unload(self):
        self.state = EngineState.UNLOADED

    def infer(self, image, **kw):
        self.received = image
        self.received_kwargs = kw
        if self.error:
            return {"error": self.error}
        return {
            "pred_label": self.pred,
            "score": self.score,
            "anomaly_map": np.zeros((32, 32), dtype=np.float32),
            "threshold_used": 0.5,
            "grid_size": 4,
            "calibrated_score": self.score * 0.9,
            "np_p_value": 0.01,
        }


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


# ─── run_detection (Grounding DINO 主路径) ──────────────────

def test_run_detection_ng(image_path, registry):
    """有检测框 → NG, 契约字段完整 (count/max_score/area_px)。"""
    registry.register(FakeDino(
        {}, boxes=[[10, 10, 60, 60], [0, 0, 20, 10]],
        labels=["scratch", "dent"], scores=[0.9, 0.6]))
    r = run_detection(registry, image_path, prompt="划痕")
    assert r["verdict"] == "NG"
    assert r["count"] == 2
    assert r["max_score"] == 0.9
    assert len(r["detections"]) == 2
    assert r["detections"][0]["area_px"] == 2500.0
    assert isinstance(r["image"], np.ndarray)
    assert r["rejected_by_size"] == 0


def test_run_detection_ok_no_boxes(image_path, registry):
    registry.register(FakeDino({}))
    r = run_detection(registry, image_path, prompt="划痕")
    assert r["verdict"] == "OK"
    assert r["count"] == 0
    assert r["max_score"] == 0.0
    assert r["detections"] == []


def test_run_detection_empty_prompt_uses_default(image_path, registry):
    """空 prompt → DEFAULT_PROMPT, 且中文提示词翻译为英文后下发。"""
    eng = FakeDino({})
    registry.register(eng)
    run_detection(registry, image_path, prompt="")
    assert eng.received_prompt == translate_prompt(DEFAULT_PROMPT)


def test_run_detection_invalid_image(registry):
    registry.register(FakeDino({}))
    r = run_detection(registry, "/nonexistent/路径.png")
    assert r["verdict"] == "ERROR"
    assert r["image"] is None


def test_run_detection_engine_missing(registry, image_path):
    r = run_detection(registry, image_path)
    assert r["verdict"] == "ERROR"
    assert "未注册" in r["error"]


def test_run_detection_load_failure(image_path, registry):
    """静默加载失败 (未进入 READY) → ERROR, 不崩溃。"""
    registry.register(FakeDino({}, fail_load=True))
    r = run_detection(registry, image_path)
    assert r["verdict"] == "ERROR"
    assert "加载失败" in r["error"]


def test_run_detection_load_exception_degraded(image_path, registry):
    """load() 抛异常 → 优雅降级 ERROR, 不向调用方传播。"""
    registry.register(FakeDino({}, raise_load=True))
    r = run_detection(registry, image_path)
    assert r["verdict"] == "ERROR"
    assert "加载失败" in r["error"]


def test_run_detection_infer_error(image_path, registry):
    """引擎推理返回 error → ERROR, 但保留原图引用。"""
    registry.register(FakeDino({}, error="GPU OOM"))
    r = run_detection(registry, image_path)
    assert r["verdict"] == "ERROR"
    assert r["error"] == "GPU OOM"
    assert isinstance(r["image"], np.ndarray)


def test_run_detection_size_filter(image_path, registry):
    """size_cfg 启用 → 小框被拒, rejected_by_size 计数正确。"""
    registry.register(FakeDino(
        {}, boxes=[[10, 10, 60, 60], [0, 0, 5, 5]],
        labels=["scratch", "noise"], scores=[0.9, 0.5]))
    size_cfg = {"enabled": True, "min_area_px": 100}
    r = run_detection(registry, image_path, prompt="划痕",
                      size_cfg=size_cfg)
    assert r["count"] == 1
    assert r["rejected_by_size"] == 1
    assert r["verdict"] == "NG"
    assert r["detections"][0]["label"] == "scratch"


def test_run_detection_ndarray_passthrough(image_path, registry):
    """v1.5.0 单次解码契约: 引擎收到已解码 ndarray, 不重复读盘。"""
    eng = FakeDino({})
    registry.register(eng)
    run_detection(registry, image_path)
    assert isinstance(eng.received, np.ndarray)


def test_run_detection_threshold_forwarded(image_path, registry):
    eng = FakeDino({})
    registry.register(eng)
    run_detection(registry, image_path, prompt="划痕", threshold=0.42)
    assert eng.received_threshold == 0.42


# ─── run_anomaly_detection (PatchCore 主路径) ───────────────

def test_anomaly_ng(image_path, registry):
    """NG 路径: 热力图叠加 + 判定印章, 校准字段透传。"""
    registry.register(FakeAnomalib({}, pred="NG", score=0.9))
    r = run_anomaly_detection(registry, image_path)
    assert r["verdict"] == "NG"
    assert r["score"] == 0.9
    assert r["threshold_used"] == 0.5
    assert r["grid_size"] == 4
    assert r["calibrated_score"] == pytest.approx(0.81)
    assert r["np_p_value"] == 0.01
    assert isinstance(r["image"], np.ndarray)
    assert r["anomaly_map"] is not None


def test_anomaly_ok(image_path, registry):
    registry.register(FakeAnomalib({}, pred="OK", score=0.1))
    r = run_anomaly_detection(registry, image_path)
    assert r["verdict"] == "OK"
    assert r["score"] == 0.1


def test_anomaly_threshold_forwarded(image_path, registry):
    """显式 threshold → 透传引擎 kwargs。"""
    eng = FakeAnomalib({})
    registry.register(eng)
    run_anomaly_detection(registry, image_path, threshold=0.42)
    assert eng.received_kwargs.get("threshold") == 0.42


def test_anomaly_threshold_none_not_forwarded(image_path, registry):
    """threshold=None → 不下发 (引擎走配置默认)。"""
    eng = FakeAnomalib({})
    registry.register(eng)
    run_anomaly_detection(registry, image_path)
    assert "threshold" not in eng.received_kwargs


def test_anomaly_invalid_image(registry):
    registry.register(FakeAnomalib({}))
    r = run_anomaly_detection(registry, "/nonexistent/路径.png")
    assert r["verdict"] == "ERROR"
    assert r["image"] is None


def test_anomaly_engine_missing(registry, image_path):
    r = run_anomaly_detection(registry, image_path)
    assert r["verdict"] == "ERROR"
    assert "未注册" in r["error"]


def test_anomaly_load_failure(image_path, registry):
    registry.register(FakeAnomalib({}, fail_load=True))
    r = run_anomaly_detection(registry, image_path)
    assert r["verdict"] == "ERROR"
    assert "加载失败" in r["error"]


def test_anomaly_load_exception_degraded(image_path, registry):
    """load() 抛异常 → 优雅降级 ERROR, 不向调用方传播。"""
    registry.register(FakeAnomalib({}, raise_load=True))
    r = run_anomaly_detection(registry, image_path)
    assert r["verdict"] == "ERROR"
    assert "加载失败" in r["error"]


def test_anomaly_empty_bank(image_path, registry):
    """记忆库为空 → 明确引导建库, 不得静默判 OK。"""
    registry.register(FakeAnomalib({}, has_bank=False))
    r = run_anomaly_detection(registry, image_path)
    assert r["verdict"] == "ERROR"
    assert "记忆库" in r["error"]


def test_anomaly_infer_error(image_path, registry):
    registry.register(FakeAnomalib({}, error="CUDA 错误"))
    r = run_anomaly_detection(registry, image_path)
    assert r["verdict"] == "ERROR"
    assert r["error"] == "CUDA 错误"
    assert isinstance(r["image"], np.ndarray)


def test_anomaly_ndarray_passthrough(image_path, registry):
    """v1.5.0 单次解码契约: 引擎收到已解码 ndarray。"""
    eng = FakeAnomalib({})
    registry.register(eng)
    run_anomaly_detection(registry, image_path)
    assert isinstance(eng.received, np.ndarray)


# ─── v1.5.0 向后兼容 re-export 面守卫 ────────────────────────

def test_reexport_surface_unchanged():
    """v1.5.0 拆分后 re-export 面不得缺失 (外部 import 兼容)。"""
    compat = [
        # core.recipes
        "DEFAULT_PROMPT", "_RECIPES_DIR", "_recipe_path", "_safe_name",
        "delete_recipe", "list_recipes", "load_recipe", "save_recipe",
        "translate_prompt",
        # core.qc_drawing
        "_DEFECT_COLORS", "_bbox_area", "_draw_detections",
        "_filter_by_size", "_overlay_heatmap", "_pick_color",
        "draw_verdict_badge",
        # core.qc_persist
        "persist_qc_image", "save_qc_result",
    ]
    missing = [n for n in compat if not hasattr(dd, n)]
    assert missing == [], f"缺失向后兼容符号: {missing}"
