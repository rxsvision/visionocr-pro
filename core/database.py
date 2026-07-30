"""SQLite 数据库初始化与访问 (Phase 3A 应收回款数据模型)

数据模型设计 (收款方视角, 管应收回款):
- contracts   合同主数据: 编号/起止日期/总金额/我方主体/客户/签单人/方向
- receivables 应收计划: 每笔应收款 (原 payments, 增加 direction/our_party)
- collections 实收记录: 实际回款, 未收 = 应收 - 累计实收
- signer_map  签单人映射: 人名 -> 飞书/企微账号
- error_log   错误日志: 结构化错误定位 (阶段/错误码/字段/原文)
- risk_alert  风险预警: 合同风险/错误提示

兼容性: 保留旧 payments 表 (不删), 新增 receivables 作为主表;
        旧库通过 _migrate 平滑升级, 幂等不丢数据。
"""
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("visionocr.database")

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

-- 合同主数据 (Phase 3A 扩展)
CREATE TABLE IF NOT EXISTS contracts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    file_path TEXT,
    contract_no TEXT,                 -- 合同编号
    title TEXT,
    parties TEXT,                     -- 签约方概述 (兼容旧字段)
    our_party TEXT,                   -- 我方主体名称
    counterparty TEXT,                -- 对方主体 (客户)
    signer TEXT,                      -- 签单人/负责人
    start_date TEXT,                  -- 合同起始日期
    end_date TEXT,                    -- 合同终止日期
    total_amount REAL,               -- 合同总金额
    currency TEXT DEFAULT 'CNY',
    direction TEXT DEFAULT 'receivable',  -- receivable(应收/回款) | payable(应付)
    status TEXT DEFAULT 'active',     -- active | completed | terminated
    raw_text TEXT,
    structured_json TEXT,
    extract_source TEXT DEFAULT 'regex',  -- llm | regex
    confidence REAL DEFAULT 0.0,
    reviewed INTEGER DEFAULT 0        -- 是否已人工复核
);

-- 应收计划 (每笔应收款)
CREATE TABLE IF NOT EXISTS receivables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id INTEGER,
    due_date TEXT,                    -- 应收日期 (绝对日期; 相对日期留空)
    amount REAL,
    currency TEXT DEFAULT 'CNY',
    condition_text TEXT,              -- 收款条件原文
    penalty TEXT,                     -- 违约条款
    direction TEXT DEFAULT 'receivable',
    status TEXT DEFAULT 'pending',    -- pending | partial | collected | overdue
    source TEXT DEFAULT 'regex',      -- llm | regex
    reminded_7d INTEGER DEFAULT 0,
    reminded_3d INTEGER DEFAULT 0,
    reminded_1d INTEGER DEFAULT 0,
    reminded_overdue INTEGER DEFAULT 0
);

-- 实收记录 (实际回款)
CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT DEFAULT (datetime('now','localtime')),
    contract_id INTEGER,
    receivable_id INTEGER,            -- 可关联到具体应收条目 (可空)
    amount REAL,
    currency TEXT DEFAULT 'CNY',
    method TEXT,                      -- 银行转账/承兑/现金等
    note TEXT
);

-- 签单人映射: 人名 -> IM 账号
CREATE TABLE IF NOT EXISTS signer_map (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,                 -- 合同里的人名
    feishu_id TEXT,                   -- 飞书 open_id / 邮箱 / 手机号
    wecom_id TEXT,                    -- 企业微信 userid
    phone TEXT,
    note TEXT
);

-- 错误日志 (结构化错误定位)
CREATE TABLE IF NOT EXISTS error_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    stage TEXT,                       -- ingest | extract | store | notify | export
    error_code TEXT,                  -- 错误码, e.g. E_EXTRACT_JSON
    file_path TEXT,
    field TEXT,                       -- 出错字段
    message TEXT,
    context TEXT,                     -- 原始片段/上下文
    suggestion TEXT                   -- 建议动作
);

-- 风险预警 (合同风险/错误提示)
CREATE TABLE IF NOT EXISTS risk_alert (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    contract_id INTEGER,
    level TEXT,                       -- red | yellow
    rule TEXT,                        -- 命中规则, e.g. missing_penalty
    message TEXT,
    evidence TEXT                     -- 原文定位
);

-- 文件哈希 (重复检测)
CREATE TABLE IF NOT EXISTS file_hashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    sha256 TEXT UNIQUE,              -- 文件内容 SHA-256
    file_path TEXT,                  -- 首次入库路径
    file_name TEXT,                  -- 原始文件名
    contract_id INTEGER,             -- 关联合同 (可空, 入库后回填)
    file_size INTEGER                -- 字节数
);

-- 通知日志 (发送记录 + 重试追踪)
CREATE TABLE IF NOT EXISTS notification_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    channel TEXT,                    -- feishu | wecom | desktop
    recipient TEXT,                  -- 签单人名 / 默认联系人
    message TEXT,
    success INTEGER DEFAULT 0,      -- 0=失败/待重试, 1=成功
    attempts INTEGER DEFAULT 1,     -- 已尝试次数
    max_attempts INTEGER DEFAULT 3,
    next_retry_at TEXT,             -- 下次重试时间 (ISO)
    error TEXT,                     -- 最后一次错误信息
    receivable_id INTEGER,          -- 关联应收条目 (可空)
    contract_id INTEGER             -- 关联合同 (可空)
);

-- 审计追踪 (合同变更历史)
CREATE TABLE IF NOT EXISTS contract_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    contract_id INTEGER,
    action TEXT,                     -- create | update | review | reject | delete
    operator TEXT DEFAULT 'system',  -- 操作者 (UI用户/系统/调度器)
    field TEXT,                      -- 变更字段 (update时)
    old_value TEXT,
    new_value TEXT,
    note TEXT                        -- 备注
);

-- 旧表保留 (兼容): payments / qc_results / behavior_events
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

# 性能索引 (Phase 3F 审查后补充)
# 注意: 索引引用的列 (如 contracts.signer/reviewed) 在旧库中可能尚未存在,
# 必须在 _migrate 补齐列之后再创建, 否则 init_db 会因 "no such column" 崩溃。
_SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_receivables_contract ON receivables(contract_id);
CREATE INDEX IF NOT EXISTS idx_receivables_status ON receivables(status);
CREATE INDEX IF NOT EXISTS idx_collections_contract ON collections(contract_id);
CREATE INDEX IF NOT EXISTS idx_contracts_signer ON contracts(signer);
CREATE INDEX IF NOT EXISTS idx_contracts_reviewed ON contracts(reviewed);
CREATE INDEX IF NOT EXISTS idx_risk_alert_contract ON risk_alert(contract_id);
"""

# 平滑迁移: 为旧库已有表补齐新增列 (table -> [(column, ddl)])
_MIGRATIONS = {
    "contracts": [
        ("contract_no", "ALTER TABLE contracts ADD COLUMN contract_no TEXT"),
        ("our_party", "ALTER TABLE contracts ADD COLUMN our_party TEXT"),
        ("counterparty", "ALTER TABLE contracts ADD COLUMN counterparty TEXT"),
        ("signer", "ALTER TABLE contracts ADD COLUMN signer TEXT"),
        ("start_date", "ALTER TABLE contracts ADD COLUMN start_date TEXT"),
        ("end_date", "ALTER TABLE contracts ADD COLUMN end_date TEXT"),
        ("total_amount", "ALTER TABLE contracts ADD COLUMN total_amount REAL"),
        ("currency", "ALTER TABLE contracts ADD COLUMN currency TEXT DEFAULT 'CNY'"),
        ("direction", "ALTER TABLE contracts ADD COLUMN direction TEXT DEFAULT 'receivable'"),
        ("status", "ALTER TABLE contracts ADD COLUMN status TEXT DEFAULT 'active'"),
        ("extract_source", "ALTER TABLE contracts ADD COLUMN extract_source TEXT DEFAULT 'regex'"),
        ("confidence", "ALTER TABLE contracts ADD COLUMN confidence REAL DEFAULT 0.0"),
        ("reviewed", "ALTER TABLE contracts ADD COLUMN reviewed INTEGER DEFAULT 0"),
        ("updated_by", "ALTER TABLE contracts ADD COLUMN updated_by TEXT DEFAULT ''"),
        ("updated_at", "ALTER TABLE contracts ADD COLUMN updated_at TEXT DEFAULT ''"),
        ("version", "ALTER TABLE contracts ADD COLUMN version INTEGER DEFAULT 1"),
    ],
    "payments": [
        ("source", "ALTER TABLE payments ADD COLUMN source TEXT DEFAULT 'regex'"),
    ],
    "ocr_results": [
        ("image_hash", "ALTER TABLE ocr_results ADD COLUMN image_hash TEXT"),
        ("scene", "ALTER TABLE ocr_results ADD COLUMN scene TEXT"),
        ("roi_count", "ALTER TABLE ocr_results ADD COLUMN roi_count INTEGER DEFAULT 1"),
        ("raw_text", "ALTER TABLE ocr_results ADD COLUMN raw_text TEXT"),
        ("corrected_text", "ALTER TABLE ocr_results ADD COLUMN corrected_text TEXT"),
        ("confidence_threshold", "ALTER TABLE ocr_results ADD COLUMN confidence_threshold REAL DEFAULT 0.75"),
        ("verdict", "ALTER TABLE ocr_results ADD COLUMN verdict TEXT DEFAULT 'OK'"),
        ("ng_regions", "ALTER TABLE ocr_results ADD COLUMN ng_regions TEXT"),
        ("corrections", "ALTER TABLE ocr_results ADD COLUMN corrections TEXT"),
        ("elapsed_sec", "ALTER TABLE ocr_results ADD COLUMN elapsed_sec REAL"),
    ],
}


def _migrate(conn: sqlite3.Connection) -> None:
    """轻量级 schema 迁移: 为旧库已有表补齐新增列 (幂等, 不丢数据)。"""
    for table, columns in _MIGRATIONS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col, ddl in columns:
            if col not in existing:
                conn.execute(ddl)
    conn.commit()


def init_db(data_dir: str) -> Path:
    """初始化数据库, 返回 db 文件路径"""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    db_path = data_path / "visionocr.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    _migrate(conn)
    conn.executescript(_SCHEMA_INDEXES)  # 索引须在补列后创建
    conn.close()
    return db_path


# 已初始化的 db 路径缓存 (避免每次 get_conn 重复执行 DDL)
_initialized_dbs: set[str] = set()


def get_conn(data_dir: str) -> sqlite3.Connection:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    db_path = data_path / "visionocr.db"
    db_str = str(db_path)

    is_new = not db_path.exists() or db_str not in _initialized_dbs
    conn = sqlite3.connect(db_str, timeout=10)
    conn.row_factory = sqlite3.Row
    # M6 修复: WAL 模式 + busy_timeout, 减少并发锁冲突
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    if is_new:
        conn.executescript(_SCHEMA)
        _migrate(conn)
        conn.executescript(_SCHEMA_INDEXES)  # 索引须在补列后创建
        _initialized_dbs.add(db_str)
    return conn


def log_ocr_audit(data_dir: str, record: dict) -> None:
    """将 OCR 识别结果写入审计日志 (生产追溯)。

    Args:
        data_dir: 数据目录 (含 visionocr.db)
        record: {
            image_path, image_hash, engine, scene, roi_count,
            raw_text, corrected_text, confidence, confidence_threshold,
            verdict, ng_regions (list), corrections (list), elapsed_sec
        }
    """
    import json
    try:
        conn = get_conn(data_dir)
        conn.execute(
            """INSERT INTO ocr_results
               (image_path, image_hash, engine, scene, roi_count,
                raw_text, corrected_text, text_content, confidence,
                confidence_threshold, verdict, ng_regions, corrections,
                elapsed_sec, structured_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.get("image_path", ""),
                record.get("image_hash", ""),
                record.get("engine", ""),
                record.get("scene", ""),
                record.get("roi_count", 1),
                record.get("raw_text", ""),
                record.get("corrected_text", ""),
                record.get("corrected_text", ""),  # text_content 兼容旧字段
                record.get("confidence", 0.0),
                record.get("confidence_threshold", 0.75),
                record.get("verdict", "OK"),
                json.dumps(record.get("ng_regions", []), ensure_ascii=False),
                json.dumps(record.get("corrections", []), ensure_ascii=False),
                record.get("elapsed_sec", 0.0),
                record.get("structured_json", "{}"),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        # 审计日志写入失败不应阻断主流程
        logger.warning("[AuditLog] 写入失败 (不影响识别): %s", e)


def backup_db(data_dir: str, max_backups: int = 5) -> Path | None:
    """使用 SQLite backup API 创建数据库热备份 (WAL 安全)。

    策略: 启动时调用, 保留最近 max_backups 份, 按时间戳轮转。
    返回备份文件路径; 数据库不存在或备份失败时返回 None。
    """
    import shutil
    from datetime import datetime

    data_path = Path(data_dir)
    db_path = data_path / "visionocr.db"
    if not db_path.exists():
        logger.debug("[Backup] 数据库不存在, 跳过备份")
        return None

    backup_dir = data_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"visionocr_{ts}.db"

    try:
        src = sqlite3.connect(str(db_path), timeout=10)
        dst = sqlite3.connect(str(backup_path))
        src.backup(dst)
        dst.close()
        src.close()
        logger.info("[Backup] 已创建: %s (%.1f KB)",
                    backup_path.name, backup_path.stat().st_size / 1024)
    except Exception as e:
        logger.warning("[Backup] 备份失败 (不影响运行): %s", e)
        # 清理可能的残缺文件
        if backup_path.exists():
            backup_path.unlink()
        return None

    # 轮转: 只保留最近 max_backups 份
    backups = sorted(backup_dir.glob("visionocr_*.db"), reverse=True)
    for old in backups[max_backups:]:
        old.unlink()
        logger.debug("[Backup] 已清理旧备份: %s", old.name)

    return backup_path
