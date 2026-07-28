"""SQLite 数据库初始化与访问"""
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ocr_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    image_path TEXT,
    engine TEXT,
    text_content TEXT,
    structured_json TEXT,
    confidence REAL
);

CREATE TABLE IF NOT EXISTS payments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id INTEGER,
    due_date TEXT,
    amount REAL,
    currency TEXT DEFAULT 'CNY',
    condition_text TEXT,
    penalty TEXT,
    status TEXT DEFAULT 'pending',
    reminded_7d INTEGER DEFAULT 0,
    reminded_3d INTEGER DEFAULT 0,
    reminded_1d INTEGER DEFAULT 0,
    source TEXT DEFAULT 'regex'
);

CREATE TABLE IF NOT EXISTS contracts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    file_path TEXT,
    title TEXT,
    parties TEXT,
    raw_text TEXT,
    structured_json TEXT
);

CREATE TABLE IF NOT EXISTS qc_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    image_path TEXT,
    barcode_content TEXT,
    barcode_type TEXT,
    verdict TEXT,
    anomaly_score REAL,
    defect_json TEXT
);

CREATE TABLE IF NOT EXISTS behavior_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    camera_id TEXT,
    event_type TEXT,
    confidence REAL,
    detail_json TEXT
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """轻量级 schema 迁移: 为旧库补齐新增列 (幂等, 不丢数据)。"""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(payments)").fetchall()}
    if "source" not in cols:
        conn.execute("ALTER TABLE payments ADD COLUMN source TEXT DEFAULT 'regex'")
    conn.commit()


def init_db(data_dir: str) -> Path:
    """初始化数据库, 返回 db 文件路径"""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    db_path = data_path / "visionocr.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    _migrate(conn)
    conn.close()
    return db_path


def get_conn(data_dir: str) -> sqlite3.Connection:
    db_path = Path(data_dir) / "visionocr.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn
