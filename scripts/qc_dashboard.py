"""质检看板启动器 (Phase 4)

用法:
    python scripts/qc_dashboard.py [--db 路径] [--port 8901] [--no-open]

默认读取 config.yaml 的 data_dir/visionocr.db, 只读服务在本机 127.0.0.1。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.config import load_config  # noqa: E402
from core.qc_dashboard import launch_dashboard  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="VisionOCR Pro 质检看板")
    ap.add_argument("--db", default="",
                    help="visionocr.db 路径 (默认取 config data_dir)")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true",
                    help="不自动打开浏览器")
    args = ap.parse_args()

    if args.db:
        db_path = Path(args.db)
    else:
        config = load_config()
        db_path = Path(config.get("data_dir", "data")) / "visionocr.db"

    try:
        proc = launch_dashboard(db_path, port=args.port, host=args.host,
                                open_browser=not args.no_open)
        proc.wait()
    except RuntimeError as e:
        print(f"[看板] {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[看板] {e} — 请先运行主程序产生检测记录, 或用 --db 指定")
        sys.exit(1)


if __name__ == "__main__":
    main()
