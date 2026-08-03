"""路径穿越防护 + 旧文件名回退兼容测试 (v1.4.1 安全修复)。"""
import pytest

from core import anomaly_bank, defect_detector, recipes


# ─── 清洗函数 ────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    "../../etc/passwd", "..\\..\\evil", "a/b\\c", "prod:v1",
    "name*with?special|chars<>", "....",
])
def test_safe_name_neutralizes_special_chars(raw):
    safe = anomaly_bank._safe_name(raw)
    assert not any(c in safe for c in '\\/:*?"<>|.')
    assert safe  # 不为空


def test_safe_name_empty_fallback():
    assert anomaly_bank._safe_name("   ") == "_"
    assert anomaly_bank._safe_name("") == "_"


# ─── anomaly_bank: 穿越拦截 + 旧文件回退 ─────────────────────

def test_bank_path_traversal_stays_inside_root(tmp_path, monkeypatch):
    banks = tmp_path / "banks"
    banks.mkdir()
    monkeypatch.setattr(anomaly_bank, "_BANKS_DIR", banks)
    p = anomaly_bank.bank_path("../../evil")
    assert p.resolve().is_relative_to(banks.resolve())
    assert not any(c in p.stem for c in '\\/:*?"<>|.')
    assert p.stem.endswith("evil")


def test_bank_path_legacy_fallback(tmp_path, monkeypatch):
    """v1.4.1 前含 . 的旧库文件仍可被定位。"""
    banks = tmp_path / "banks"
    banks.mkdir()
    monkeypatch.setattr(anomaly_bank, "_BANKS_DIR", banks)
    legacy = banks / "prod.v1.npz"
    legacy.write_bytes(b"dummy")
    assert anomaly_bank.bank_path("prod.v1") == legacy


def test_bank_path_prefers_sanitized_when_both_exist(tmp_path, monkeypatch):
    banks = tmp_path / "banks"
    banks.mkdir()
    monkeypatch.setattr(anomaly_bank, "_BANKS_DIR", banks)
    (banks / "prod_v1.npz").write_bytes(b"new")
    (banks / "prod.v1.npz").write_bytes(b"old")
    assert anomaly_bank.bank_path("prod.v1").name == "prod_v1.npz"


def test_bank_path_legacy_traversal_ignored(tmp_path, monkeypatch):
    """回退分支不得被用于越界访问。"""
    banks = tmp_path / "banks"
    banks.mkdir()
    monkeypatch.setattr(anomaly_bank, "_BANKS_DIR", banks)
    outside = tmp_path / "evil.npz"
    outside.write_bytes(b"x")
    p = anomaly_bank.bank_path("../evil")
    assert p.resolve().is_relative_to(banks.resolve())
    assert not p.exists()


def test_bank_path_subspace_and_dinov2(tmp_path, monkeypatch):
    sa = tmp_path / "sa"
    dv = tmp_path / "dv"
    sa.mkdir()
    dv.mkdir()
    monkeypatch.setattr(anomaly_bank, "_BANKS_SA_DIR", sa)
    monkeypatch.setattr(anomaly_bank, "_BANKS_DV_DIR", dv)
    assert anomaly_bank.bank_path_subspace("x/y").resolve().is_relative_to(sa.resolve())
    assert anomaly_bank.bank_path_dinov2("x/y").resolve().is_relative_to(dv.resolve())


# ─── 配方穿越拦截 + 回退 (实现位于 core.recipes, v1.5.0 拆分) ─

def test_recipe_path_traversal_raises(tmp_path, monkeypatch):
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    monkeypatch.setattr(recipes, "_RECIPES_DIR", recipes_dir)
    # 清洗后名称恒在根内, 越界输入也应落在根内而非抛到外部
    p = recipes._recipe_path("../../evil")
    assert p.resolve().is_relative_to(recipes_dir.resolve())


def test_recipe_legacy_fallback_and_load(tmp_path, monkeypatch):
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    monkeypatch.setattr(recipes, "_RECIPES_DIR", recipes_dir)
    legacy = recipes_dir / "prod.v1.json"
    legacy.write_text('{"name": "prod.v1", "prompt": "划痕"}', encoding="utf-8")
    assert recipes._recipe_path("prod.v1") == legacy
    data = defect_detector.load_recipe("prod.v1")
    assert data is not None and data["prompt"] == "划痕"


def test_save_recipe_sanitizes_new_name(tmp_path, monkeypatch):
    recipes_dir = tmp_path / "recipes"
    monkeypatch.setattr(recipes, "_RECIPES_DIR", recipes_dir)
    defect_detector.save_recipe("new.prod", "划痕")
    assert (recipes_dir / "new_prod.json").exists()
    data = defect_detector.load_recipe("new_prod")
    assert data["name"] == "new_prod"


def test_save_recipe_overwrites_legacy_in_place(tmp_path, monkeypatch):
    """旧文件原地覆写, 不产生重复条目。"""
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    monkeypatch.setattr(recipes, "_RECIPES_DIR", recipes_dir)
    (recipes_dir / "prod.v1.json").write_text("{}", encoding="utf-8")
    defect_detector.save_recipe("prod.v1", "凹陷")
    data = defect_detector.load_recipe("prod.v1")
    assert data["prompt"] == "凹陷"
    assert len(list(recipes_dir.glob("*.json"))) == 1


def test_delete_recipe_legacy(tmp_path, monkeypatch):
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    monkeypatch.setattr(recipes, "_RECIPES_DIR", recipes_dir)
    (recipes_dir / "prod.v1.json").write_text("{}", encoding="utf-8")
    assert defect_detector.delete_recipe("prod.v1") is True
    assert not (recipes_dir / "prod.v1.json").exists()
