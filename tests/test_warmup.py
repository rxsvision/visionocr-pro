"""core/warmup.py 单元测试 - 预热策略契约 (核心检测优先, OCR 后台)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import warmup
from core.warmup import (_get_background_engines, _get_core_engines,
                         warmup_engines)


class FakeEngine:
    """可控假引擎: 记录 infer 调用, 可注入加载失败/dummy 推理异常。"""

    def __init__(self, name, ready=True, infer_raises=False,
                 fail_load=False):
        self.name = name
        self._ready = ready
        self.infer_raises = infer_raises
        self.fail_load = fail_load  # True: ensure_loaded 后仍不就绪
        self.infer_calls = 0

    def is_ready(self):
        return self._ready

    def infer(self, *a, **kw):
        self.infer_calls += 1
        if self.infer_raises:
            raise RuntimeError("记忆库为空 (预期业务报错)")
        return {}


class FakeRegistry:
    def __init__(self, engines: dict):
        self.engines = engines
        self.ensure_loaded_calls = []

    def get(self, name):
        return self.engines.get(name)

    def ensure_loaded(self, name):
        self.ensure_loaded_calls.append(name)
        eng = self.engines.get(name)
        if eng is None:
            raise KeyError(name)
        if not eng.fail_load:  # 模拟加载成功后就绪
            eng._ready = True
        return eng


def _full_registry(**overrides):
    names = ["anomalib", "dinov2_anomaly", "grounding_dino",
             "yolo_defect", "rapidocr", "scene_classifier", "barcode"]
    engines = {n: FakeEngine(n) for n in names}
    engines.update(overrides)
    return FakeRegistry(engines)


def _cfg(**union_overrides):
    cfg = {"ocr": {"default_engine": "rapidocr",
                   "fallback_engine": "rapidocr"},
           "qc": {"union": {"enable_patchcore": True,
                            "enable_dinov2": True,
                            "enable_dino": True}}}
    cfg["qc"]["union"].update(union_overrides)
    return cfg


class TestGetCoreEngines:
    def test_all_enabled_default(self):
        assert _get_core_engines(_cfg()) == [
            "anomalib", "dinov2_anomaly", "grounding_dino"]

    def test_respects_disabled_flags(self):
        core = _get_core_engines(_cfg(enable_dino=False,
                                      enable_dinov2=False))
        assert core == ["anomalib"]

    def test_empty_config_defaults_all_on(self):
        assert _get_core_engines({}) == [
            "anomalib", "dinov2_anomaly", "grounding_dino"]

    def test_yolo_never_in_core(self):
        """YOLO 产品门控按需加载, 永不入同步预热"""
        assert "yolo_defect" not in _get_core_engines(_cfg())


class TestGetBackgroundEngines:
    def test_ocr_first_then_scene_barcode(self):
        bg = _get_background_engines(_cfg(), "rapidocr")
        assert bg[0] == "rapidocr"
        assert bg == ["rapidocr", "scene_classifier", "barcode"]

    def test_scene_classifier_disabled(self):
        cfg = _cfg()
        cfg["ocr"]["scene_classifier"] = {"enabled": False}
        assert _get_background_engines(cfg, "rapidocr") == [
            "rapidocr", "barcode"]

    def test_dedup_when_primary_is_barcode(self):
        bg = _get_background_engines(_cfg(), "barcode")
        assert bg == ["barcode", "scene_classifier"]


class TestWarmupEngines:
    def test_core_loaded_sync_ocr_background_only(self):
        reg = _full_registry()
        report = warmup_engines(reg, _cfg())
        # 同步预热只含核心三源且顺序固定 (后台线程追加项不影响前 3 项)
        assert reg.ensure_loaded_calls[:3] == [
            "anomalib", "dinov2_anomaly", "grounding_dino"]
        assert "rapidocr" not in reg.ensure_loaded_calls[:3]
        assert report["ok"] is True
        assert set(report["core"]) == {
            "anomalib", "dinov2_anomaly", "grounding_dino"}
        assert all(e["ok"] for e in report["core"].values())
        # OCR 移入后台列表
        assert report["background"][0] == "rapidocr"

    def test_disabled_source_not_warmed(self):
        reg = _full_registry()
        warmup_engines(reg, _cfg(enable_dino=False))
        assert "grounding_dino" not in reg.ensure_loaded_calls

    def test_core_load_failure_marks_not_ok(self):
        bad = FakeEngine("anomalib", ready=False, fail_load=True)
        reg = _full_registry(anomalib=bad)
        report = warmup_engines(reg, _cfg())
        assert report["ok"] is False
        assert report["core"]["anomalib"]["ok"] is False
        # 其他核心源不受影响, 继续预热
        assert report["core"]["dinov2_anomaly"]["ok"] is True

    def test_dummy_infer_business_error_tolerated(self):
        """PatchCore 无记忆库时 dummy 推理报错, 不影响加载就绪判定"""
        eng = FakeEngine("anomalib", infer_raises=True)
        reg = _full_registry(anomalib=eng)
        report = warmup_engines(reg, _cfg())
        assert report["core"]["anomalib"]["ok"] is True
        assert eng.infer_calls == 1

    def test_core_engine_not_registered_tolerated(self):
        """引擎未注册 (依赖缺失被跳过) 时降级不崩溃"""
        reg = _full_registry()
        del reg.engines["grounding_dino"]
        report = warmup_engines(reg, _cfg())
        assert report["ok"] is False
        assert report["core"]["grounding_dino"]["ok"] is False
        assert report["core"]["anomalib"]["ok"] is True


class TestBackgroundStatus:
    def setup_method(self):
        warmup._background_status.clear()

    def test_background_thread_updates_status(self):
        reg = _full_registry()
        warmup._start_background_warmup(reg, ["rapidocr", "barcode"])
        import time
        deadline = time.time() + 5
        while time.time() < deadline:
            st = warmup.get_background_status()
            if st.get("rapidocr") == "ready" and st.get("barcode") == "ready":
                break
            time.sleep(0.05)
        assert warmup.get_background_status() == {
            "rapidocr": "ready", "barcode": "ready"}

    def test_background_fallback_on_primary_failure(self):
        bad = FakeEngine("rapidocr", ready=False, fail_load=True)
        fb = FakeEngine("ppocrv6", ready=False)
        reg = _full_registry(rapidocr=bad, ppocrv6=fb)
        warmup._start_background_warmup(reg, ["rapidocr"], fallback="ppocrv6")
        import time
        deadline = time.time() + 5
        while time.time() < deadline:
            st = warmup.get_background_status()
            if st.get("ppocrv6") == "ready":
                break
            time.sleep(0.05)
        assert warmup.get_background_status().get("rapidocr") == "failed"
        assert warmup.get_background_status().get("ppocrv6") == "ready"
