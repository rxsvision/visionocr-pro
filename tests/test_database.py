"""core/database.py 单元测试"""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import init_db, get_conn, backup_db


class TestInitDb:
    def test_creates_db_file(self, tmp_data_dir):
        """init_db 创建数据库文件"""
        db_path = init_db(tmp_data_dir)
        assert db_path.exists()
        assert db_path.suffix == ".db"

    def test_all_tables_created(self, tmp_data_dir):
        """所有核心表存在"""
        db_path = init_db(tmp_data_dir)
        conn = sqlite3.connect(str(db_path))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        expected = {"contracts", "receivables", "collections",
                    "signer_map", "error_log", "risk_alert",
                    "ocr_results", "qc_results", "file_hashes"}
        assert expected.issubset(tables)

    def test_idempotent(self, tmp_data_dir):
        """重复调用不报错 (幂等)"""
        init_db(tmp_data_dir)
        init_db(tmp_data_dir)  # 第二次不应抛异常


class TestMigration:
    def test_adds_missing_columns(self, tmp_data_dir):
        """迁移为旧表补齐新列"""
        db_path = init_db(tmp_data_dir)
        conn = sqlite3.connect(str(db_path))
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(contracts)").fetchall()}
        conn.close()
        assert "contract_no" in cols
        assert "direction" in cols
        assert "reviewed" in cols


class TestBackup:
    def test_backup_creates_file(self, tmp_data_dir):
        """backup_db 创建备份文件"""
        init_db(tmp_data_dir)
        result = backup_db(tmp_data_dir)
        assert result is not None
        assert result.exists()
        assert "backups" in str(result)

    def test_backup_rotation(self, tmp_data_dir):
        """超过 max_backups 时清理旧备份"""
        init_db(tmp_data_dir)
        for _ in range(7):
            backup_db(tmp_data_dir, max_backups=3)
        backup_dir = Path(tmp_data_dir) / "backups"
        backups = list(backup_dir.glob("visionocr_*.db"))
        assert len(backups) <= 3

    def test_backup_no_db_returns_none(self, tmp_data_dir):
        """数据库不存在时返回 None"""
        result = backup_db(tmp_data_dir)
        assert result is None


class TestGetConn:
    def test_wal_mode(self, tmp_data_dir):
        """连接使用 WAL 日志模式"""
        init_db(tmp_data_dir)
        conn = get_conn(tmp_data_dir)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"
