"""Phase 4 质检看板 (Datasette) 测试

不依赖网络/真实服务: 视图与元数据直接测 SQLite;
Datasette HTTP 层用内置 async client (0.65) 进程内测试。
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from core import qc_dashboard
from core.database import init_db
from core.defect_detector import persist_qc_image, save_qc_result

try:
    from datasette.app import Datasette
    _HAS_DATASETTE = True
except ImportError:
    _HAS_DATASETTE = False

needs_datasette = pytest.mark.skipif(
    not _HAS_DATASETTE, reason="datasette 未安装")


# ─── fixtures ────────────────────────────────────────────────
@pytest.fixture
def seeded_db(tmp_path):
    """含 5 条记录 (3 NG / 2 OK) 的 visionocr.db, 已建视图。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = init_db(data_dir)
    conn = sqlite3.connect(str(db_path))
    # 伪图: 供图片路由读取
    img = data_dir / "sample.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake-image-bytes")
    save_qc_result(conn, str(img), "NG",
                   [{"label": "划痕", "score": 0.9}], 0.91)
    save_qc_result(conn, str(img), "OK", [], 0.10)
    save_qc_result(conn, str(img), "NG",
                   [{"label": "脏污", "score": 0.8}], 0.85)
    save_qc_result(conn, str(data_dir / "已移动" / "不存在.png"), "NG", [], 0.77)
    save_qc_result(conn, str(img), "OK", [], 0.05)
    conn.close()
    qc_dashboard.ensure_views(db_path)
    return Path(db_path)


# ─── 视图层 ──────────────────────────────────────────────────
def test_ensure_views_idempotent(seeded_db):
    qc_dashboard.ensure_views(seeded_db)  # 二次调用不应报错
    conn = sqlite3.connect(str(seeded_db))
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view'")}
    conn.close()
    assert {"qc_daily_stats", "qc_ng_detail"} <= names


def test_daily_stats_view(seeded_db):
    conn = sqlite3.connect(str(seeded_db))
    rows = conn.execute(
        "SELECT day, total, ng_count, ng_rate_pct FROM qc_daily_stats"
    ).fetchall()
    conn.close()
    assert len(rows) == 1  # 全部今天
    _, total, ng_count, ng_rate = rows[0]
    assert total == 5
    assert ng_count == 3
    assert abs(ng_rate - 60.0) < 0.01


def test_ng_detail_view(seeded_db):
    conn = sqlite3.connect(str(seeded_db))
    rows = conn.execute(
        "SELECT id, image_url, defect_summary FROM qc_ng_detail"
    ).fetchall()
    conn.close()
    assert len(rows) == 3
    # 倒序 + 图片直链格式
    assert rows[0][0] > rows[-1][0]
    for _id, url, _ in rows:
        assert url == f"/-/qc-img/{_id}"
    assert "划痕" in rows[-1][2]


def test_ensure_views_missing_db(tmp_path):
    with pytest.raises(FileNotFoundError):
        qc_dashboard.ensure_views(tmp_path / "no.db")


# ─── 元数据 ──────────────────────────────────────────────────
def test_metadata_structure():
    meta = qc_dashboard.build_metadata()
    assert "VisionOCR" in meta["title"]
    tables = meta["databases"]["visionocr"]["tables"]
    assert {"qc_results", "qc_daily_stats", "qc_ng_detail"} <= set(tables)
    # datasette 要求表条目为 dict (内部调 .get("hidden")), 纯字符串会 500
    for name, entry in tables.items():
        assert isinstance(entry, dict), name
        assert entry.get("description")


def test_write_metadata_yaml(tmp_path):
    import yaml

    p = qc_dashboard.write_metadata_yaml(tmp_path)
    assert p.is_file()
    loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert loaded["title"] == qc_dashboard.build_metadata()["title"]


# ─── Datasette HTTP 层 (进程内 client) ──────────────────────
def _make_app(db_path: Path) -> Datasette:
    # plugins_dir 与 CLI `--plugins-dir dashboard/` 同一加载机制
    return Datasette([str(db_path)],
                     plugins_dir=str(qc_dashboard.PLUGINS_DIR))


def _run(ds, path: str):
    async def _go():
        await ds.invoke_startup()
        return await ds.client.get(path)
    return asyncio.run(_go())


@needs_datasette
def test_datasette_serves_table(seeded_db):
    ds = _make_app(seeded_db)
    resp = _run(ds, "/visionocr/qc_results.json?_shape=array")
    assert resp.status_code == 200
    assert len(resp.json()) == 5


@needs_datasette
def test_image_route_serves_file(seeded_db):
    ds = _make_app(seeded_db)
    resp = _run(ds, "/-/qc-img/1")
    assert resp.status_code == 200
    assert b"fake-image-bytes" in resp.content
    assert "png" in resp.headers["content-type"]


@needs_datasette
def test_image_route_missing_row(seeded_db):
    ds = _make_app(seeded_db)
    resp = _run(ds, "/-/qc-img/9999")
    assert resp.status_code == 404


@needs_datasette
def test_image_route_missing_file(seeded_db):
    ds = _make_app(seeded_db)
    resp = _run(ds, "/-/qc-img/4")  # 指向不存在文件那条
    assert resp.status_code == 404


# ─── 图片持久化 (防 Gradio 临时路径失效) ────────────────────


def test_persist_copies_with_hash_name(tmp_path):
    src = tmp_path / "gradio_tmp" / "upload_x123.png"
    src.parent.mkdir()
    src.write_bytes(b"\x89PNG\r\n\x1a\nABC")
    dest_dir = tmp_path / "data" / "qc_images"

    out = persist_qc_image(str(src), dest_dir)

    out_p = Path(out)
    assert out_p.parent == dest_dir
    assert out_p.is_file()
    assert out_p.read_bytes() == src.read_bytes()
    # 内容哈希命名: 16 位十六进制 + 小写扩展名
    assert len(out_p.stem) == 16
    assert all(c in "0123456789abcdef" for c in out_p.stem)
    assert out_p.suffix == ".png"


def test_persist_dedup_same_content(tmp_path):
    dest_dir = tmp_path / "qc_images"
    a = tmp_path / "a.png"
    b = tmp_path / "other_name.png"  # 同内容不同名 (Gradio 临时名随机)
    a.write_bytes(b"same-bytes")
    b.write_bytes(b"same-bytes")

    out_a = persist_qc_image(str(a), dest_dir)
    out_b = persist_qc_image(str(b), dest_dir)

    assert out_a == out_b
    assert len(list(dest_dir.iterdir())) == 1  # 不产生冗余副本


def test_persist_missing_file_passthrough(tmp_path):
    ghost = str(tmp_path / "不存在.png")
    assert persist_qc_image(ghost, tmp_path / "qc_images") == ghost


def test_persist_already_in_dest_passthrough(tmp_path):
    dest_dir = tmp_path / "qc_images"
    dest_dir.mkdir()
    img = dest_dir / "already.png"
    img.write_bytes(b"x")
    assert persist_qc_image(str(img), dest_dir) == str(img)
    assert len(list(dest_dir.iterdir())) == 1  # 不再复制自己


@needs_datasette
def test_gradio_temp_cleanup_regression(tmp_path):
    """原始 bug 复现: Gradio 临时文件在落库后被清理, 看板直链必须仍可用。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = init_db(data_dir)

    # 模拟 Gradio 上传: 临时目录中的图片
    tmp_upload = tmp_path / "gradio" / "tmpfile.png"
    tmp_upload.parent.mkdir()
    tmp_upload.write_bytes(b"\x89PNG\r\n\x1a\nPERSISTED-OK")

    conn = sqlite3.connect(str(db_path))
    stored = persist_qc_image(str(tmp_upload), data_dir / "qc_images")
    rowid = save_qc_result(conn, stored, "NG", [{"label": "划痕"}], 0.9)
    conn.close()
    qc_dashboard.ensure_views(db_path)

    # Gradio 清理临时文件 (bug 的触发条件)
    tmp_upload.unlink()

    ds = _make_app(Path(db_path))
    resp = _run(ds, f"/-/qc-img/{rowid}")
    assert resp.status_code == 200
    assert b"PERSISTED-OK" in resp.content
