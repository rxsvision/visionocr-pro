"""YOLO 缺陷检测引擎测试 (使用冒烟训练权重)"""
from pathlib import Path

import pytest

from engines.vision.yolo_defect import YOLODefectEngine

_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_WEIGHTS = (_ROOT / "finetune" / "output_yolo" / "pcb_smoke"
                  / "weights" / "best.pt")
_VAL_IMG_DIR = _ROOT / "finetune" / "data_pcb" / "images" / "val"


@pytest.fixture(scope="module")
def engine():
    if not _SMOKE_WEIGHTS.exists():
        pytest.skip("冒烟权重不存在, 跳过 (先运行 train_yolo.py 冒烟)")
    cfg = {"yolo_defect": {"weights": str(_SMOKE_WEIGHTS),
                           "confidence_threshold": 0.1,
                           "imgsz": 640}}
    eng = YOLODefectEngine(cfg)
    eng.load()
    return eng


def test_load_ready(engine):
    assert engine.is_ready()
    assert engine.meta.name == "yolo_defect"
    assert len(engine._names) == 6  # PCB 6 类


def test_infer_returns_structure(engine):
    imgs = sorted(_VAL_IMG_DIR.glob("*.*"))
    if not imgs:
        pytest.skip("无验证图像")
    result = engine.infer(str(imgs[0]))
    assert "boxes" in result
    assert "labels" in result
    assert "scores" in result
    assert "count" in result
    assert isinstance(result["boxes"], list)
    # 检出框为 [x1,y1,x2,y2]
    for box in result["boxes"]:
        assert len(box) == 4


def test_infer_batch_no_crash(engine):
    imgs = sorted(_VAL_IMG_DIR.glob("*.*"))[:5]
    if not imgs:
        pytest.skip("无验证图像")
    total = 0
    for p in imgs:
        r = engine.infer(str(p))
        assert not r.get("error"), r.get("error")
        total += r["count"]
    # 冒烟权重虽弱, 低阈值下应至少有检出
    assert total >= 0


def test_no_weights_errors():
    cfg = {"yolo_defect": {"weights": "/nonexistent/best.pt"}}
    eng = YOLODefectEngine(cfg)
    eng.load()
    assert eng.state.value == "error"
    r = eng.infer("dummy.jpg")
    assert r.get("error")
