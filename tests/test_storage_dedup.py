"""storage / dedup 数据层测试 (审查项: Medium)。"""
import json

from core.database import get_conn
from core.dedup import (check_duplicate, compute_sha256,
                        compute_sha256_bytes, is_duplicate, link_contract,
                        register_file)
from core.storage import Storage


# ─── Storage ─────────────────────────────────────────────────

def test_storage_creates_dirs(tmp_path):
    s = Storage(str(tmp_path / "data"))
    assert s.uploads.is_dir()
    assert s.results.is_dir()
    assert s.exports.is_dir()


def test_save_upload_copies_file(tmp_path):
    src = tmp_path / "合同.pdf"
    src.write_bytes(b"%PDF-fake")
    s = Storage(str(tmp_path / "data"))
    dest = s.save_upload(src, category="contract")
    assert dest.exists()
    assert dest.read_bytes() == b"%PDF-fake"
    assert dest.parent.name == "contract"


def test_save_result_writes_text(tmp_path):
    s = Storage(str(tmp_path / "data"))
    dest = s.save_result(json.dumps({"a": 1}), "out")
    assert dest.exists()
    assert json.loads(dest.read_text(encoding="utf-8")) == {"a": 1}


# ─── dedup ───────────────────────────────────────────────────

def test_sha256_file_matches_bytes(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello" * 10000)
    assert compute_sha256(f) == compute_sha256_bytes(b"hello" * 10000)


def test_register_and_check_duplicate(tmp_data_dir):
    conn = get_conn(tmp_data_dir)
    sha = compute_sha256_bytes(b"contract-A")
    assert check_duplicate(conn, sha) is None

    fid1 = register_file(conn, sha, "/tmp/a.pdf", "a.pdf", 10)
    fid2 = register_file(conn, sha, "/tmp/a_copy.pdf")  # 幂等
    assert fid1 == fid2
    rec = check_duplicate(conn, sha)
    assert rec is not None and rec["file_name"] == "a.pdf"
    conn.close()


def test_is_duplicate_one_step(tmp_path, tmp_data_dir):
    conn = get_conn(tmp_data_dir)
    f = tmp_path / "dup.pdf"
    f.write_bytes(b"same-content")
    ok, rec = is_duplicate(conn, f)
    assert ok is False and rec is None

    register_file(conn, compute_sha256(f), str(f))
    ok, rec = is_duplicate(conn, f)
    assert ok is True and rec is not None
    conn.close()


def test_link_contract_backfill(tmp_data_dir):
    conn = get_conn(tmp_data_dir)
    sha = compute_sha256_bytes(b"contract-B")
    register_file(conn, sha, "/tmp/b.pdf")
    link_contract(conn, sha, 42)
    rec = check_duplicate(conn, sha)
    assert rec["contract_id"] == 42

    # register_file 对已存在记录也回填 contract_id
    sha2 = compute_sha256_bytes(b"contract-C")
    fid = register_file(conn, sha2, "/tmp/c.pdf")
    register_file(conn, sha2, "/tmp/c.pdf", contract_id=7)
    rec2 = check_duplicate(conn, sha2)
    assert rec2["contract_id"] == 7 and rec2["id"] == fid
    conn.close()
