# -*- coding: utf-8 -*-
"""PP-OCRv6 常驻容器服务 (v1.3.0 P0-2) 单元测试 — 不依赖 Docker。

覆盖:
- worker.format_ocr_result 三种 OCRResult 形态
- server 模式推理协议 (成功/错误JSON/连接失败自愈/重启失败)
- _infer_docker 模式分发
- load() 降级决策 (server 失败 → run 模式, 状态仍 READY)
- unload() 停止容器
- _health_ok 探测
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import engines.ocr.ppocrv6 as pp
from engines.base import EngineState
from engines.ocr import _paddle_worker as worker


def _engine(cfg_extra=None):
    cfg = {"ocr": {"ppocrv6": cfg_extra or {}}}
    return pp.PPOCRv6Engine(cfg)


def _png(tmp_path):
    p = tmp_path / "t.png"
    p.write_bytes(b"\x89PNG fake bytes")
    return str(p)


# ─── format_ocr_result ────────────────────────────────────────
class TestFormatResult:
    def test_dict_like_page(self):
        page = {
            "rec_texts": ["ABC", "123"],
            "rec_polys": [[[0, 0], [10, 0], [10, 5], [0, 5]], None],
            "rec_scores": [0.9, 0.8],
        }
        out = worker.format_ocr_result([page])
        assert out["text"] == "ABC\n123"
        assert len(out["lines"]) == 2
        assert out["confidence"] == pytest.approx(0.85, abs=1e-3)
        assert out["lines"][0]["box"] == [[0.0, 0.0], [10.0, 0.0],
                                          [10.0, 5.0], [0.0, 5.0]]

    def test_object_page(self):
        class Page:
            rec_texts = ["X"]
            rec_polys = [[[1, 2], [3, 2], [3, 4], [1, 4]]]
            rec_scores = [0.7]
        out = worker.format_ocr_result([Page()])
        assert out["text"] == "X"
        assert out["lines"][0]["confidence"] == 0.7

    def test_legacy_list_tuple_page(self):
        page = [([0, 0, 5, 0, 5, 5, 0, 5], ("HELLO", 0.99))]
        out = worker.format_ocr_result([page])
        assert out["text"] == "HELLO"
        assert len(out["lines"][0]["box"]) == 4

    def test_empty(self):
        out = worker.format_ocr_result([])
        assert out == {"text": "", "lines": [], "confidence": 0.0}


# ─── server 模式推理协议 ──────────────────────────────────────
class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class TestInferServer:
    def test_success(self, tmp_path, monkeypatch):
        eng = _engine()
        eng._backend, eng._docker_mode = "docker", "server"
        calls = []

        def fake_post(url, files=None, timeout=None):
            calls.append((url, files, timeout))
            return _FakeResp({"text": "SN001", "lines": [
                {"text": "SN001", "box": [], "confidence": 0.97}],
                "confidence": 0.97})

        monkeypatch.setattr(pp.requests, "post", fake_post)
        r = eng._infer_server(_png(tmp_path))
        assert r["text"] == "SN001"
        assert r["engine"] == "ppocrv6"
        assert r["backend"] == "docker"
        assert "error" not in r
        # multipart field 名必须是 file (与 paddle_server.py 契约)
        assert "file" in calls[0][1]

    def test_error_json(self, tmp_path, monkeypatch):
        eng = _engine()
        eng._backend, eng._docker_mode = "docker", "server"
        monkeypatch.setattr(pp.requests, "post",
                            lambda *a, **k: _FakeResp({"error": "OOM"}))
        r = eng._infer_server(_png(tmp_path))
        assert r["error"] == "OOM"
        assert r["text"] == ""

    def test_connection_fail_then_restart_success(self, tmp_path, monkeypatch):
        eng = _engine()
        eng._backend, eng._docker_mode = "docker", "server"
        state = {"n": 0, "restarted": False}

        def fake_post(url, files=None, timeout=None):
            state["n"] += 1
            if state["n"] == 1:
                raise pp.requests.ConnectionError("refused")
            return _FakeResp({"text": "OK2", "lines": [], "confidence": 0.9})

        monkeypatch.setattr(pp.requests, "post", fake_post)
        monkeypatch.setattr(eng, "_stop_server",
                            lambda: state.__setitem__("restarted", True))
        monkeypatch.setattr(eng, "_start_server", lambda: True)
        r = eng._infer_server(_png(tmp_path))
        assert r["text"] == "OK2"
        assert state["restarted"] is True

    def test_connection_fail_restart_fail(self, tmp_path, monkeypatch):
        eng = _engine()
        eng._backend, eng._docker_mode = "docker", "server"
        monkeypatch.setattr(pp.requests, "post",
                            lambda *a, **k: (_ for _ in ()).throw(
                                pp.requests.ConnectionError("refused")))
        monkeypatch.setattr(eng, "_stop_server", lambda: None)
        monkeypatch.setattr(eng, "_start_server", lambda: False)
        r = eng._infer_server(_png(tmp_path))
        assert "error" in r and "重启失败" in r["error"]

    def test_missing_file(self):
        eng = _engine()
        eng._backend, eng._docker_mode = "docker", "server"
        r = eng._infer_server("不存在的路径.png")
        assert "error" in r


# ─── 模式分发 ─────────────────────────────────────────────────
class TestDispatch:
    def test_dispatch_server(self, monkeypatch):
        eng = _engine()
        eng._backend, eng._docker_mode = "docker", "server"
        monkeypatch.setattr(eng, "_infer_server", lambda p: {"m": "server"})
        monkeypatch.setattr(eng, "_infer_docker_run", lambda p: {"m": "run"})
        assert eng._infer_docker("x.png")["m"] == "server"

    def test_dispatch_run(self, monkeypatch):
        eng = _engine()
        eng._backend, eng._docker_mode = "docker", "run"
        monkeypatch.setattr(eng, "_infer_server", lambda p: {"m": "server"})
        monkeypatch.setattr(eng, "_infer_docker_run", lambda p: {"m": "run"})
        assert eng._infer_docker("x.png")["m"] == "run"


# ─── load() 降级决策 ──────────────────────────────────────────
class TestLoad:
    def test_server_success(self, monkeypatch):
        eng = _engine()
        monkeypatch.setattr(eng, "_check_docker", lambda: True)
        monkeypatch.setattr(eng, "_start_server", lambda: True)
        eng.load()
        assert eng.state == EngineState.READY
        assert eng._docker_mode == "server"

    def test_server_fail_fallback_run(self, monkeypatch):
        eng = _engine()
        monkeypatch.setattr(eng, "_check_docker", lambda: True)
        monkeypatch.setattr(eng, "_start_server", lambda: False)
        eng.load()
        assert eng.state == EngineState.READY
        assert eng._docker_mode == "run"

    def test_config_wiring(self, monkeypatch):
        eng = _engine({"port": 9999, "startup_timeout": 30,
                       "container_name": "custom-serve"})
        monkeypatch.setattr(eng, "_check_docker", lambda: True)
        monkeypatch.setattr(eng, "_start_server", lambda: True)
        eng.load()
        assert eng._port == 9999
        assert eng._startup_timeout == 30
        assert eng._container_name == "custom-serve"
        assert eng._server_url == "http://127.0.0.1:9999"


# ─── unload / health ─────────────────────────────────────────
class TestLifecycle:
    def test_unload_stops_container(self):
        eng = _engine()
        eng._backend, eng._docker_mode = "docker", "server"
        eng.state = EngineState.READY
        stopped = []
        eng._stop_server = lambda: stopped.append(True)
        eng.unload()
        assert stopped == [True]
        assert eng.state == EngineState.UNLOADED
        assert eng._docker_mode == ""

    def test_unload_run_mode_no_stop(self):
        eng = _engine()
        eng._backend, eng._docker_mode = "docker", "run"
        eng.state = EngineState.READY
        eng._stop_server = lambda: pytest.fail("run 模式不应停容器")
        eng.unload()
        assert eng.state == EngineState.UNLOADED

    def test_health_ok_true(self, monkeypatch):
        eng = _engine()
        monkeypatch.setattr(pp.requests, "get",
                            lambda url, timeout=None: _FakeResp(
                                {"status": "ok", "ready": True}))
        assert eng._health_ok() is True

    def test_health_ok_not_ready(self, monkeypatch):
        eng = _engine()
        monkeypatch.setattr(pp.requests, "get",
                            lambda url, timeout=None: _FakeResp(
                                {"status": "ok", "ready": False}))
        assert eng._health_ok() is False

    def test_health_ok_conn_refused(self, monkeypatch):
        eng = _engine()

        def boom(url, timeout=None):
            raise pp.requests.ConnectionError("refused")
        monkeypatch.setattr(pp.requests, "get", boom)
        assert eng._health_ok() is False
