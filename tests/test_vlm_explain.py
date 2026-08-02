"""vlm_explain 单元测试 — 假引擎/假 registry (不需要真实 Ollama)"""
import numpy as np
import pytest

from core.imutils import imwrite_unicode
from core.vlm_explain import explain_union, DEFAULT_PROMPT


class FakeVLM:
    """模拟 OllamaEngine: 记录调用, 返回固定文本。"""

    def __init__(self, ready=True):
        self._ready = ready
        self.calls = []

    def is_ready(self):
        return self._ready

    def infer(self, image_path=None, prompt="", **kwargs):
        if not self._ready:
            return {"text": "", "confidence": 0.0, "engine": "ollama_vlm",
                    "error": "引擎未就绪"}
        self.calls.append({"image_path": image_path, "prompt": prompt,
                           "kwargs": kwargs})
        return {"text": f"划痕, 中等 (第{len(self.calls)}区)",
                "confidence": 1.0, "engine": "ollama_vlm"}


class FakeRegistry:
    def __init__(self, engine=None):
        self._engine = engine
        self.ensure_calls = []

    def get(self, name):
        return self._engine if name == "ollama_vlm" else None

    def ensure_loaded(self, name):
        self.ensure_calls.append(name)


def _make_union_result(with_map=True, verdict="NG", boxes=None):
    r = {"verdict": verdict, "ng_sources": ["patchcore"],
         "anomaly_map": None, "dino": None, "yolo": None}
    if with_map:
        m = np.full((120, 160), 0.1, dtype=np.float32)
        m[50:70, 80:110] = 1.0  # 一个亮斑
        r["anomaly_map"] = m
    if boxes:
        r["yolo"] = {"boxes": boxes, "scores": [0.9] * len(boxes),
                     "labels": ["x"] * len(boxes), "count": len(boxes)}
    return r


@pytest.fixture
def image_path(tmp_path):
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    p = str(tmp_path / "sample.png")
    assert imwrite_unicode(p, img)
    return p


@pytest.fixture
def config():
    return {"qc": {"vlm_explain": {"enabled": True, "max_rois": 3,
                                   "no_think": True, "max_tokens": 256}}}


class TestExplainUnion:
    def test_ok_verdict_early_return(self, image_path, config):
        vlm = FakeVLM()
        reg = FakeRegistry(vlm)
        out = explain_union(reg, image_path,
                            _make_union_result(verdict="OK"), config)
        assert "error" not in out
        assert vlm.calls == []
        assert "OK" in out["summary"]

    def test_disabled_by_config(self, image_path, config):
        config["qc"]["vlm_explain"]["enabled"] = False
        out = explain_union(FakeRegistry(FakeVLM()), image_path,
                            _make_union_result(), config)
        assert out.get("error")
        assert "禁用" in out["error"]

    def test_engine_missing(self, image_path, config):
        out = explain_union(FakeRegistry(None), image_path,
                            _make_union_result(), config)
        assert "未注册" in out["error"]

    def test_engine_not_ready(self, image_path, config):
        vlm = FakeVLM(ready=False)
        reg = FakeRegistry(vlm)
        out = explain_union(reg, image_path, _make_union_result(), config)
        assert out.get("error")
        assert "未就绪" in out["error"]
        assert reg.ensure_calls == ["ollama_vlm"]  # 尝试过自动加载

    def test_success_with_heatmap_roi(self, image_path, config):
        vlm = FakeVLM()
        out = explain_union(FakeRegistry(vlm), image_path,
                            _make_union_result(with_map=True), config)
        assert "error" not in out
        assert len(vlm.calls) == len(out["texts"]) >= 1
        assert len(out["crops"]) == len(out["texts"])
        assert "区域1" in out["summary"]
        # 提示词: 内置 + no_think 后缀
        assert DEFAULT_PROMPT.split("。")[0] in vlm.calls[0]["prompt"]
        assert vlm.calls[0]["prompt"].endswith("/no_think")
        assert vlm.calls[0]["kwargs"].get("max_tokens") == 256

    def test_box_roi_used(self, image_path, config):
        vlm = FakeVLM()
        res = _make_union_result(with_map=False,
                                 boxes=[(20, 30, 80, 90)])
        out = explain_union(FakeRegistry(vlm), image_path, res, config)
        assert "error" not in out
        assert len(out["texts"]) == 1
        assert out["rois"][0]["source"] == "det"

    def test_full_image_fallback(self, image_path, config):
        """NG 但无热力图无检测框 → 整图兜底。"""
        vlm = FakeVLM()
        out = explain_union(FakeRegistry(vlm), image_path,
                            _make_union_result(with_map=False), config)
        assert "error" not in out
        assert out["rois"][0]["source"] == "full"
        assert len(out["texts"]) == 1

    def test_bad_image_path(self, config):
        vlm = FakeVLM()
        out = explain_union(FakeRegistry(vlm), "不存在.png",
                            _make_union_result(), config)
        assert out.get("error")
        assert vlm.calls == []

    def test_vlm_error_recorded_per_roi(self, image_path, config):
        class FailVLM(FakeVLM):
            def infer(self, image_path=None, prompt="", **kwargs):
                return {"text": "", "error": "推理失败: timeout"}
        out = explain_union(FakeRegistry(FailVLM()), image_path,
                            _make_union_result(), config)
        assert "error" not in out  # 引擎就绪, 流程完成
        assert all("解释失败" in t for t in out["texts"])
