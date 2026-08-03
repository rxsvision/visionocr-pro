"""安全加固测试 (v1.5.0): NPZ 反序列化硬化 + 批量审批信任链门控。"""
import json

import numpy as np
import pytest

from core.database import get_conn
from core.npz_io import load_npz_safe
from core.payment_store import (list_pending_review, mark_reviewed,
                                structurally_valid)


# ─── NPZ pickle 硬化 ─────────────────────────────────────────

def test_load_plain_arrays_without_pickle(tmp_path):
    """纯数值/字符串库: 默认 (禁 pickle) 可正常加载。"""
    p = tmp_path / "bank.npz"
    np.savez_compressed(str(p), bank=np.zeros((4, 8)),
                        product_name="产品A",
                        meta_n_samples=12, calibrated_threshold=0.5)
    data = load_npz_safe(p)
    assert data["bank"].shape == (4, 8)
    assert str(data["product_name"]) == "产品A"
    assert float(data["calibrated_threshold"]) == 0.5


def test_load_object_array_rejected_by_default(tmp_path):
    """含 pickle 对象的库默认拒绝加载 (防恶意反序列化)。"""
    p = tmp_path / "evil.npz"
    np.savez(str(p), obj=np.array({"payload": 1}, dtype=object))
    with pytest.raises(ValueError, match="拒绝加载"):
        load_npz_safe(p)


def test_load_object_array_legacy_optin(tmp_path):
    """显式 allow_legacy_pickle=True 时旧版库可回退加载。"""
    p = tmp_path / "legacy.npz"
    np.savez(str(p), obj=np.array({"payload": 1}, dtype=object))
    data = load_npz_safe(p, allow_legacy_pickle=True)
    assert data["obj"].item() == {"payload": 1}


# ─── 批量审批信任链门控 ──────────────────────────────────────

def test_structurally_valid_prefers_valid_flag():
    assert structurally_valid(
        {"structured_json": json.dumps({"valid": True}),
         "total_amount": None}) is True
    assert structurally_valid(
        {"structured_json": json.dumps({"valid": False}),
         "total_amount": 99999.0}) is False  # 高金额也拒绝


def test_structurally_valid_fallback_amount():
    """旧数据无 valid 字段 → 回退要求有总金额。"""
    assert structurally_valid(
        {"structured_json": "{}", "total_amount": 100.0}) is True
    assert structurally_valid(
        {"structured_json": None, "total_amount": None}) is False


def test_structurally_valid_bad_json():
    assert structurally_valid(
        {"structured_json": "{invalid", "total_amount": 50.0}) is True
    assert structurally_valid(
        {"structured_json": "{invalid", "total_amount": 0}) is False


def _insert_contract(conn, confidence, valid, amount):
    sj = json.dumps({"valid": valid}) if valid is not None else "{}"
    cur = conn.execute(
        """INSERT INTO contracts (file_path, title, confidence, reviewed,
                                  total_amount, structured_json)
           VALUES (?, ?, ?, 0, ?, ?)""",
        (f"/tmp/c{confidence}.pdf", f"合同-{confidence}", confidence,
         amount, sj))
    conn.commit()
    return int(cur.lastrowid)


def test_pending_review_exposes_structured_json(tmp_data_dir):
    conn = get_conn(tmp_data_dir)
    _insert_contract(conn, 0.9, False, None)
    rows = list_pending_review(conn)
    assert len(rows) == 1
    assert "structured_json" in rows[0]
    conn.close()


def test_batch_gate_high_conf_but_invalid_blocked(tmp_data_dir):
    """门控契约: 高置信 + 勾稽未过 → 不得进入可批准集合。"""
    conn = get_conn(tmp_data_dir)
    bad = _insert_contract(conn, 0.95, False, None)   # LLM 自报高分但无效
    good = _insert_contract(conn, 0.9, True, 1000.0)
    low = _insert_contract(conn, 0.4, True, 500.0)    # 低于阈值

    threshold = 0.8
    approved, blocked = [], []
    for r in list_pending_review(conn):
        if (r.get("confidence") or 0) < threshold:
            continue
        if structurally_valid(r):
            mark_reviewed(conn, r["id"])
            approved.append(r["id"])
        else:
            blocked.append(r["id"])

    assert approved == [good]
    assert blocked == [bad]
    assert low not in approved
    conn.close()
