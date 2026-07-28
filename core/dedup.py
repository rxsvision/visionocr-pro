"""文件重复检测 (Phase 5: 批量自动化)

基于 SHA-256 内容哈希, 在合同入库前检测是否已存在相同文件。
策略: 重复文件默认跳过 (不重复入库), 但返回警告信息供 UI 展示。
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Optional


def compute_sha256(file_path: str | Path) -> str:
    """计算文件 SHA-256 (分块读取, 支持大文件)。"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """计算字节数据的 SHA-256。"""
    return hashlib.sha256(data).hexdigest()


def check_duplicate(conn: sqlite3.Connection, sha256: str) -> Optional[dict]:
    """查询该哈希是否已入库。返回已有记录 dict 或 None。"""
    row = conn.execute(
        "SELECT * FROM file_hashes WHERE sha256 = ?", (sha256,)
    ).fetchone()
    return dict(row) if row else None


def register_file(conn: sqlite3.Connection, sha256: str, file_path: str,
                  file_name: str = "", file_size: int = 0,
                  contract_id: Optional[int] = None) -> int:
    """注册文件哈希, 返回 file_hash_id。若已存在则返回已有 id。"""
    existing = check_duplicate(conn, sha256)
    if existing:
        # 已存在: 若 contract_id 为空则回填
        if contract_id and not existing.get("contract_id"):
            conn.execute(
                "UPDATE file_hashes SET contract_id = ? WHERE id = ?",
                (contract_id, existing["id"]),
            )
            conn.commit()
        return existing["id"]

    cur = conn.execute(
        """INSERT INTO file_hashes (sha256, file_path, file_name, file_size, contract_id)
           VALUES (?, ?, ?, ?, ?)""",
        (sha256, file_path, file_name or Path(file_path).name, file_size, contract_id),
    )
    conn.commit()
    return int(cur.lastrowid)


def link_contract(conn: sqlite3.Connection, sha256: str, contract_id: int) -> None:
    """将已注册的文件哈希关联到合同 id。"""
    conn.execute(
        "UPDATE file_hashes SET contract_id = ? WHERE sha256 = ?",
        (contract_id, sha256),
    )
    conn.commit()


def is_duplicate(conn: sqlite3.Connection, file_path: str | Path) -> tuple[bool, Optional[dict]]:
    """一步式检测: 计算哈希 + 查重。

    Returns:
        (is_dup, existing_record)
        - is_dup=True 时 existing_record 包含首次入库信息
        - is_dup=False 时 existing_record=None
    """
    sha = compute_sha256(file_path)
    existing = check_duplicate(conn, sha)
    return existing is not None, existing
