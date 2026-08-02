"""OllamaEngine 图像预处理与配置测试 (不发网络请求)"""
from __future__ import annotations

import numpy as np
import pytest
import cv2

from engines.llm.ollama_provider import OllamaEngine


@pytest.fixture
def engine(sample_config):
    return OllamaEngine(sample_config)


def _write_img(path, w, h):
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def test_small_image_passthrough(engine, tmp_path):
    p = _write_img(tmp_path / "small.png", 640, 480)
    out = engine._prepare_image_bytes(str(p))
    assert out == p.read_bytes()  # 小图原样返回


def test_large_image_downscaled(engine, tmp_path):
    p = _write_img(tmp_path / "big.png", 4000, 2000)
    out = engine._prepare_image_bytes(str(p))
    assert len(out) < p.stat().st_size  # 明显变小 (JPEG)
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert max(decoded.shape[:2]) <= OllamaEngine.MAX_VLM_SIDE
    # 长边缩到上限, 比例保持
    assert decoded.shape[1] == OllamaEngine.MAX_VLM_SIDE
    assert abs(decoded.shape[0] - OllamaEngine.MAX_VLM_SIDE // 2) <= 1


def test_huge_linescan_image(engine, tmp_path):
    # 模拟线扫图 (宽高比极端): 15000x1000
    p = _write_img(tmp_path / "line.png", 15000, 1000)
    out = engine._prepare_image_bytes(str(p))
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[1] == OllamaEngine.MAX_VLM_SIDE


def test_non_image_returns_raw(engine, tmp_path):
    p = tmp_path / "notimg.bin"
    p.write_bytes(b"\x00\x01\x02garbage")
    out = engine._prepare_image_bytes(str(p))
    assert out == b"\x00\x01\x02garbage"


def test_host_env_override(engine, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11435")
    host = OllamaEngine._resolve_host({"host": "http://localhost:11434"})
    assert host == "http://127.0.0.1:11435"  # env 优先 + 自动补 scheme


def test_host_config_fallback(engine, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    host = OllamaEngine._resolve_host({"host": "http://localhost:11434/"})
    assert host == "http://localhost:11434"  # 去尾斜杠


def test_host_default(engine, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert OllamaEngine._resolve_host({}) == "http://localhost:11434"
