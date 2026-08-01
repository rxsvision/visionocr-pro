"""YOLO 产品门控测试 — 跨域误报防护"""
import shutil
from pathlib import Path

import pytest

from core import yolo_products
from core.yolo_products import (
    is_real_product, resolve_yolo_weights, list_yolo_products,
)
from engines.vision.yolo_defect import YOLODefectEngine

_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_WEIGHTS = (_ROOT / "finetune" / "output_yolo" / "pcb_smoke"
                  / "weights" / "best.pt")


# ─── is_real_product ────────────────────────────────────────

def test_placeholders_not_real():
    assert not is_real_product("")
    assert not is_real_product(None)
    assert not is_real_product("(新建)")
    assert not is_real_product("(自定义)")
    assert not is_real_product("   ")


def test_real_product_name():
    assert is_real_product("PCB")
    assert is_real_product("铝合金外壳")


# ─── resolve_yolo_weights ───────────────────────────────────

def test_resolve_no_product_returns_none():
    assert resolve_yolo_weights("") is None
    assert resolve_yolo_weights("(自定义)") is None


def test_resolve_missing_product_returns_none():
    # 未训练的产品 → None (即使存在其他产品权重也不回退)
    assert resolve_yolo_weights("不存在的产品XYZ") is None


def test_resolve_existing_product(tmp_path, monkeypatch):
    fake_dir = tmp_path / "yolo"
    fake_dir.mkdir()
    (fake_dir / "PCB.pt").write_bytes(b"x")
    monkeypatch.setattr(yolo_products, "_YOLO_DIR", fake_dir)
    assert resolve_yolo_weights("PCB") == fake_dir / "PCB.pt"


def test_resolve_sanitizes_unsafe_name(tmp_path, monkeypatch):
    fake_dir = tmp_path / "yolo"
    fake_dir.mkdir()
    (fake_dir / "a_b.pt").write_bytes(b"x")
    monkeypatch.setattr(yolo_products, "_YOLO_DIR", fake_dir)
    # "a/b" 含非法字符 → 安全化为 "a_b"
    assert resolve_yolo_weights("a/b") == fake_dir / "a_b.pt"


def test_list_yolo_products(tmp_path, monkeypatch):
    fake_dir = tmp_path / "yolo"
    fake_dir.mkdir()
    (fake_dir / "PCB.pt").write_bytes(b"x")
    (fake_dir / "金属盖.pt").write_bytes(b"x")
    monkeypatch.setattr(yolo_products, "_YOLO_DIR", fake_dir)
    assert list_yolo_products() == ["PCB", "金属盖"]


# ─── 引擎门控 load_for_product ──────────────────────────────

def test_load_for_product_no_context_returns_false():
    eng = YOLODefectEngine({"yolo_defect": {}})
    assert eng.load_for_product("") is False
    assert eng.load_for_product("(自定义)") is False
    assert not eng.is_ready()  # 不应误加载


def test_load_for_product_untrained_returns_false():
    eng = YOLODefectEngine({"yolo_defect": {}})
    assert eng.load_for_product("未训练产品") is False
    assert not eng.is_ready()


def test_load_for_product_with_weights(tmp_path, monkeypatch):
    # 真实加载权重需 ultralytics; 轻量环境 (CI) 无此包时一致 skip。
    pytest.importorskip("ultralytics")
    if not _SMOKE_WEIGHTS.exists():
        pytest.skip("冒烟权重不存在")
    fake_dir = tmp_path / "yolo"
    fake_dir.mkdir()
    shutil.copy(_SMOKE_WEIGHTS, fake_dir / "PCB.pt")
    monkeypatch.setattr(yolo_products, "_YOLO_DIR", fake_dir)

    eng = YOLODefectEngine({"yolo_defect": {"imgsz": 640}})
    assert eng.load_for_product("PCB") is True
    assert eng.is_ready()
    assert eng._loaded_product == "PCB"
    # 同一产品复用, 不重复加载
    assert eng.load_for_product("PCB") is True
