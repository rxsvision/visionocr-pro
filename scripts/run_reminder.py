"""VisionOCR 提醒扫描独立入口 (Windows 计划任务 / cron 调用)

用法:
  python scripts/run_reminder.py

Windows 计划任务注册 (PowerShell Admin):
  python scripts/run_reminder.py --register
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from core.scheduler import run_reminder_check, register_windows_task


def main():
    cfg = load_config()

    if "--register" in sys.argv:
        cmd = register_windows_task(cfg)
        print("请在 PowerShell (管理员) 中执行以下命令注册系统计划任务:")
        print()
        print(cmd)
        return

    results = run_reminder_check(cfg)
    print(f"提醒扫描完成: 触发 {len(results)} 条")
    for r in results:
        status = "已送达" if r.get("delivered") else "发送失败"
        print(f"  [{r['level']}] {r['message']} ({status})")


if __name__ == "__main__":
    main()
