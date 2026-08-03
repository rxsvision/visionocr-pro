"""OCR 图像预处理管线测试 (审查项: High — OCR 精度链路)。"""
import os

import cv2
import numpy as np
import pytest

from core.imutils import imwrite_unicode
from core.image_preprocess import (check_image_quality, preprocess_for_ocr)


@pytest.fixture
def small_img_path(tmp_path):
    """低分辨率图 (触发 upscale)。"""
    img = np.full((200, 300, 3), 180, dtype=np.uint8)
    cv2.putText(img, "ABC123", (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 0), 2)
    p = str(tmp_path / "small.png")
    assert imwrite_unicode(p, img)
    return p


@pytest.fixture
def noisy_img_path(tmp_path):
    """带噪点图 (触发 denoise)。"""
    rng = np.random.default_rng(42)
    img = rng.integers(100, 200, (600, 800, 3), dtype=np.uint8)
    p = str(tmp_path / "noisy.png")
    assert imwrite_unicode(p, img)
    return p


def test_disabled_returns_original(small_img_path):
    out, meta = preprocess_for_ocr(small_img_path, {"enabled": False})
    assert out == small_img_path
    assert meta["steps"] == []


def test_invalid_path_returns_original(tmp_path):
    bad = str(tmp_path / "不存在.png")
    out, meta = preprocess_for_ocr(bad)
    assert out == bad


def test_upscale_small_image(small_img_path):
    cfg = {"denoise": False, "clahe": False, "sharpen": False,
           "upscale": True, "upscale_min_height": 800}
    out, meta = preprocess_for_ocr(small_img_path, cfg)
    assert out != small_img_path
    assert os.path.isfile(out)
    assert any(s.startswith("upscale") for s in meta["steps"])
    h_after = int(meta["size_after"].split("x")[1])
    assert h_after >= 400  # 200px 放大至少 2x
    os.unlink(out)


def test_downscale_huge_image(tmp_path):
    """超大图先缩小防 OOM (仅 downscale, 其余步骤关闭保持快速)。"""
    img = np.full((2000, 4500, 3), 128, dtype=np.uint8)
    p = str(tmp_path / "huge.png")
    assert imwrite_unicode(p, img)
    cfg = {"denoise": False, "clahe": False, "sharpen": False,
           "upscale": False}
    out, meta = preprocess_for_ocr(p, cfg)
    assert any(s.startswith("downscale") for s in meta["steps"])
    assert os.path.isfile(out)
    w_after = int(meta["size_after"].split("x")[0])
    assert w_after <= 4096
    os.unlink(out)


def test_binarize_outputs_grayscale(small_img_path):
    cfg = {"denoise": False, "clahe": False, "sharpen": False,
           "upscale": False, "binarize": True}
    out, meta = preprocess_for_ocr(small_img_path, cfg)
    assert "binarize" in meta["steps"]
    result = cv2.imread(out, cv2.IMREAD_UNCHANGED)
    assert result is not None and result.ndim == 2
    os.unlink(out)


def test_denoise_step_recorded(noisy_img_path):
    cfg = {"denoise": True, "clahe": False, "sharpen": False,
           "upscale": False}
    out, meta = preprocess_for_ocr(noisy_img_path, cfg)
    assert "denoise" in meta["steps"]
    assert os.path.isfile(out)
    os.unlink(out)


# ─── 质量预检 ────────────────────────────────────────────────

def test_quality_blur_detection(tmp_path):
    """纯色图 Laplacian 方差≈0 → 判模糊。"""
    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    p = str(tmp_path / "flat.png")
    assert imwrite_unicode(p, img)
    r = check_image_quality(p)
    assert r["blur"] is True
    assert r["ok"] is False


def test_quality_underexposed(tmp_path):
    img = np.full((300, 300, 3), 5, dtype=np.uint8)
    p = str(tmp_path / "dark.png")
    assert imwrite_unicode(p, img)
    r = check_image_quality(p)
    assert r["exposure"] == "underexposed"


def test_quality_normal_textured(tmp_path):
    rng = np.random.default_rng(7)
    img = rng.integers(60, 200, (300, 300, 3), dtype=np.uint8)
    p = str(tmp_path / "tex.png")
    assert imwrite_unicode(p, img)
    r = check_image_quality(p)
    assert r["exposure"] == "normal"
    assert r["blur"] is False
    assert r["ok"] is True


def test_quality_unreadable(tmp_path):
    r = check_image_quality(str(tmp_path / "missing.png"))
    assert r["ok"] is False
    assert r["exposure"] == "unreadable"
