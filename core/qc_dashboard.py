"""Phase 4 质检看板 — Datasette 只读浏览层 (本机)

设计决策:
- 少造轮子: 直接用 Apache-2.0 的 Datasette 服务现有 visionocr.db,
  不自研 Web 框架/前端。
- 只读: datasette serve 默认只读, 看板不修改检测数据。
- 增值层只有两样:
  1. SQL 视图 (qc_daily_stats / qc_ng_detail) — 日 NG 率趋势与 NG 明细;
  2. dashboard/qc_image_plugin.py — 按行号回显原始检测图。
- 纯本机: 默认绑定 127.0.0.1, 不做鉴权 (与产线内网部署假设一致)。

用法:
    python scripts/qc_dashboard.py [--port 8901] [--no-open]
"""
from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
import sys
import webbrowser
from pathlib import Path

logger = logging.getLogger("visionocr.qc_dashboard")

_ROOT = Path(__file__).resolve().parents[1]
PLUGINS_DIR = _ROOT / "dashboard"

# ─── 统计视图 ────────────────────────────────────────────────
# 注: 视图不落数据, 只改查询口径; IF NOT EXISTS 保证幂等。
_VIEWS = [
    # 按日统计: 总数 / NG 数 / NG 率 / 平均分
    """
    CREATE VIEW IF NOT EXISTS qc_daily_stats AS
    SELECT
        date(created_at)                            AS day,
        COUNT(*)                                    AS total,
        SUM(CASE WHEN verdict = 'NG' THEN 1 ELSE 0 END) AS ng_count,
        ROUND(100.0 * SUM(CASE WHEN verdict = 'NG' THEN 1 ELSE 0 END)
              / COUNT(*), 2)                        AS ng_rate_pct,
        ROUND(AVG(anomaly_score), 4)                AS avg_score
    FROM qc_results
    GROUP BY date(created_at)
    ORDER BY day DESC
    """,
    # NG 明细: 倒序 + 图片直链 (datasette 会把 URL 文本渲染为可点击链接)
    """
    CREATE VIEW IF NOT EXISTS qc_ng_detail AS
    SELECT
        id,
        created_at,
        image_path,
        '/-/qc-img/' || id                          AS image_url,
        anomaly_score,
        barcode_content,
        substr(defect_json, 1, 200)                 AS defect_summary
    FROM qc_results
    WHERE verdict = 'NG'
    ORDER BY id DESC
    """,
]


def ensure_views(db_path: Path | str) -> None:
    """在目标库上创建统计视图 (幂等)。"""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"数据库不存在: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        for ddl in _VIEWS:
            conn.execute(ddl)
        conn.commit()
    finally:
        conn.close()
    logger.info("[看板] 统计视图就绪: %s", db_path)


# ─── Datasette 元数据 ────────────────────────────────────────
def build_metadata() -> dict:
    """看板元数据 (标题与中文表说明)。

    注意: datasette 要求每个表条目为 dict (内部会调用 .get("hidden")),
    纯字符串会导致首页与表页面 500。
    """
    return {
        "title": "VisionOCR Pro 质检看板",
        "description": "工业外观检测结果浏览与统计 (只读 · 本机)",
        "databases": {
            "visionocr": {
                "tables": {
                    "qc_results": {
                        "description": (
                            "全部检测记录: 每次检测一行 "
                            "(判定/异常分/缺陷 JSON/条码)"
                        )
                    },
                    "qc_daily_stats": {
                        "description": "按日统计: 总数 / NG 数 / NG 率 / 平均分"
                    },
                    "qc_ng_detail": {
                        "description": (
                            "NG 明细 (倒序), image_url 可点击查看原图"
                        )
                    },
                }
            }
        },
    }


def write_metadata_yaml(data_dir: Path | str) -> Path:
    """把元数据写到 data 目录, 返回文件路径。"""
    import yaml

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "datasette_metadata.yaml"
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(build_metadata(), f, allow_unicode=True,
                       sort_keys=False)
    return meta_path


# ─── 启动 ────────────────────────────────────────────────────
def datasette_available() -> bool:
    try:
        import datasette  # noqa: F401
        return True
    except ImportError:
        return False


def launch_dashboard(db_path: Path | str, port: int = 8901,
                     host: str = "127.0.0.1",
                     open_browser: bool = True) -> subprocess.Popen:
    """启动 Datasette 只读看板 (前台子进程, Ctrl+C 退出)。

    返回 Popen 句柄 (供测试/上层管理)。datasette 未安装时抛 RuntimeError。
    """
    db_path = Path(db_path)
    if not datasette_available():
        raise RuntimeError(
            "未安装 datasette, 请先执行: pip install datasette")
    if db_path.name != "visionocr.db":
        logger.warning("[看板] 数据库文件名不是 visionocr.db, "
                       "图片直链路由将不可用")
    ensure_views(db_path)
    meta_path = write_metadata_yaml(db_path.parent)
    cmd = [
        sys.executable, "-m", "datasette", "serve", str(db_path),
        "--host", host, "--port", str(port),
        "--metadata", str(meta_path),
        "--plugins-dir", str(PLUGINS_DIR),
        "--setting", "sql_time_limit_ms", "5000",
    ]
    url = f"http://{host}:{port}/"
    print(f"[看板] 启动 Datasette: {url} (Ctrl+C 退出)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 无浏览器环境不致命
            pass
    # 强制 UTF-8: datasette CLI 用平台默认编码 (zh-CN Windows = GBK/cp936)
    # 读取 --metadata 与插件源码, 不加 PYTHONUTF8 会把中文元数据读坏。
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(cmd, cwd=str(_ROOT), env=env)
