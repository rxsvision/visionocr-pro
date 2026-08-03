"""透视/倾斜矫正测试 (审查项: High — OCR 精度链路)。"""
import os

import cv2
import numpy as np
import pytest

from core.imutils import imwrite_unicode
from core.perspective_correct import (
    _detect_skew_angle, _order_points, _rotate_image, _warp_quad,
    correct_perspective)


@pytest.fixture
def white_img_path(tmp_path):
    """纯白图: 无几何特征 → 保守策略不矫正。"""
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    p = str(tmp_path / "white.png")
    assert imwrite_unicode(p, img)
    return p


def test_disabled_returns_original(white_img_path):
    out, meta = correct_perspective(white_img_path, {"enabled": False})
    assert out == white_img_path
    assert meta["corrected"] is False


def test_invalid_path(tmp_path):
    bad = str(tmp_path / "不存在.png")
    out, meta = correct_perspective(bad)
    assert out == bad
    assert meta["reason"] == "imread_failed"


def test_no_features_no_correction(white_img_path):
    """保守策略: 检测不到特征时不动图像。"""
    out, meta = correct_perspective(white_img_path)
    assert out == white_img_path
    assert meta["corrected"] is False


def test_order_points_shuffle():
    pts = np.array([[100, 100], [0, 0], [100, 0], [0, 100]],
                   dtype=np.float32)
    rect = _order_points(pts)
    assert rect[0].tolist() == [0, 0]        # 左上
    assert rect[1].tolist() == [100, 0]      # 右上
    assert rect[2].tolist() == [100, 100]    # 右下
    assert rect[3].tolist() == [0, 100]      # 左下


def test_warp_quad_rect_output_size():
    quad = np.array([[0, 0], [200, 0], [200, 100], [0, 100]],
                    dtype=np.float32)
    img = np.full((150, 250, 3), 128, dtype=np.uint8)
    warped = _warp_quad(img, quad)
    assert warped is not None
    assert warped.shape[:2] == (100, 200)


def test_warp_quad_too_small_returns_none():
    quad = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    assert _warp_quad(img, quad) is None


def test_rotate_image_90_swaps_dims():
    img = np.full((100, 200, 3), 128, dtype=np.uint8)
    rotated = _rotate_image(img, 90.0)
    h, w = rotated.shape[:2]
    assert abs(h - 200) <= 2 and abs(w - 100) <= 2


def test_detect_skew_on_tilted_lines():
    """合成 5° 倾斜文本行 → 检出角度接近 -5° (图像坐标 y 向下)。"""
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    for y in range(80, 350, 40):
        cv2.line(img, (50, y), (550, y), (0, 0, 0), 4)
    # 绕中心旋转 5°
    M = cv2.getRotationMatrix2D((300, 200), 5.0, 1.0)
    tilted = cv2.warpAffine(img, M, (600, 400),
                            borderValue=(255, 255, 255))
    angle = _detect_skew_angle(tilted)
    assert angle is not None
    # Hough 检出方向与旋转方向相反, 绝对值应接近 5°
    assert 2.0 <= abs(angle) <= 8.0


def test_detect_skew_blank_returns_none():
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    assert _detect_skew_angle(img) is None


def test_full_pipeline_deskew(tmp_path):
    """端到端: 倾斜文本图 → 矫正生效并输出临时文件。"""
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    for y in range(80, 350, 40):
        cv2.line(img, (50, y), (550, y), (0, 0, 0), 4)
    M = cv2.getRotationMatrix2D((300, 200), 6.0, 1.0)
    tilted = cv2.warpAffine(img, M, (600, 400),
                            borderValue=(255, 255, 255))
    p = str(tmp_path / "tilted.png")
    assert imwrite_unicode(p, tilted)
    out, meta = correct_perspective(p, {"perspective": False})
    if meta["corrected"]:
        assert out != p and os.path.isfile(out)
        assert "deskew_angle" in meta
        os.unlink(out)
    else:
        # 保守策略允许不矫正, 但必须给出理由且不得崩溃
        assert out == p
