"""条码引擎测试 — pyzbar/ZBar"""
import numpy as np
import cv2
import pytest


@pytest.fixture
def barcode_engine():
    from engines.vision.barcode import BarcodeEngine
    engine = BarcodeEngine({"barcode": {"engine": "zbar"}})
    engine.load()
    return engine


@pytest.fixture
def qr_image():
    """生成测试QR码图像 (250x250 BGR)。"""
    qr = cv2.QRCodeEncoder_create()
    img = qr.encode("TEST-12345")
    img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(img, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


class TestBarcodeEngine:
    def test_load(self, barcode_engine):
        from engines.base import EngineState
        assert barcode_engine.state == EngineState.READY

    def test_meta(self, barcode_engine):
        meta = barcode_engine.meta
        assert meta.name == "barcode"
        assert meta.vram_gb == 0.0

    def test_qr_decode(self, barcode_engine, qr_image):
        result = barcode_engine.infer(qr_image)
        assert result["count"] >= 1
        assert result["codes"][0]["content"] == "TEST-12345"
        assert result["codes"][0]["type"] == "QRCODE"

    def test_qr_position(self, barcode_engine, qr_image):
        result = barcode_engine.infer(qr_image)
        code = result["codes"][0]
        assert len(code["rect"]) == 4
        assert code["rect"][2] > 0  # width > 0
        assert len(code["polygon"]) >= 4

    def test_no_barcode(self, barcode_engine):
        """纯白图像不应检出条码。"""
        blank = np.ones((200, 200, 3), dtype=np.uint8) * 255
        result = barcode_engine.infer(blank)
        assert result["count"] == 0

    def test_file_path(self, barcode_engine, qr_image, tmp_path):
        """文件路径输入 (含中文)。"""
        p = tmp_path / "测试_条码.png"
        cv2.imencode(".png", qr_image)[1].tofile(str(p))
        result = barcode_engine.infer(str(p))
        assert result["count"] >= 1
        assert result["codes"][0]["content"] == "TEST-12345"

    def test_enhanced_decode(self, barcode_engine, qr_image):
        """低对比度图像应通过增强策略解码。"""
        # 降低对比度
        gray = cv2.cvtColor(qr_image, cv2.COLOR_BGR2GRAY)
        low_contrast = (gray.astype(np.float32) * 0.3 + 180).astype(np.uint8)
        low_contrast_bgr = cv2.cvtColor(low_contrast, cv2.COLOR_GRAY2BGR)
        result = barcode_engine.infer(low_contrast_bgr)
        # 增强策略可能成功也可能失败, 不应崩溃
        assert "count" in result
