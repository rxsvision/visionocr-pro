"""定时调度器 (Phase 5: 批量自动化)

双保险策略:
- APScheduler BackgroundScheduler: 应用运行期间, 按配置时间自动执行 check_reminders。
- Windows Task Scheduler 注册脚本: 应用未运行时, 由系统级计划任务兜底执行。

配置 (config.yaml → scheduler):
  enabled: true
  reminder_time: "09:00"     # 每日提醒扫描时间 (HH:MM)
  catch_up: true             # 启动时补跑: 若上次运行距今 >24h, 立即执行一次
  status_file: "data/scheduler_state.json"
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("visionocr.scheduler")

_scheduler = None  # 全局单例
_lock = threading.Lock()


# ─── 状态持久化 ──────────────────────────────────────────────
def _state_path(config: dict) -> Path:
    rel = config.get("scheduler", {}).get("status_file", "data/scheduler_state.json")
    return Path(rel)


def _read_state(config: dict) -> dict:
    p = _state_path(config)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_state(config: dict, state: dict) -> None:
    p = _state_path(config)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── 核心任务 ────────────────────────────────────────────────
def run_reminder_check(config: dict) -> list[dict]:
    """执行一次到期提醒扫描, 更新 last_run 时间戳。"""
    from core.database import get_conn
    from core.payment_store import check_reminders

    data_dir = config.get("data_dir", "data")
    conn = get_conn(data_dir)
    try:
        fired = check_reminders(conn, do_notify=True, config=config)
        logger.info("提醒扫描完成: 触发 %d 条", len(fired))
    finally:
        conn.close()

    # 更新状态
    state = _read_state(config)
    state["last_reminder_run"] = datetime.now().isoformat(timespec="seconds")
    state["last_fired_count"] = len(fired)
    _write_state(config, state)
    return fired


def run_retry_notifications(config: dict) -> int:
    """重试失败的通知 (由调度器每5分钟调用)。"""
    from core.database import get_conn
    from core.notifier import retry_failed_notifications

    data_dir = config.get("data_dir", "data")
    conn = get_conn(data_dir)
    try:
        return retry_failed_notifications(conn, config)
    finally:
        conn.close()


# ─── APScheduler 启动/停止 ──────────────────────────────────
def start_scheduler(config: dict) -> bool:
    """启动应用内定时调度器。返回是否成功启动。"""
    global _scheduler
    scfg = config.get("scheduler", {}) or {}
    if not scfg.get("enabled", True):
        logger.info("调度器已禁用 (scheduler.enabled=false)")
        return False

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning("APScheduler 未安装, 应用内调度不可用。"
                       "请 pip install apscheduler 或使用 Windows 计划任务兜底。")
        return False

    with _lock:
        if _scheduler and _scheduler.running:
            logger.debug("调度器已在运行, 跳过重复启动")
            return True

        _scheduler = BackgroundScheduler(daemon=True)

        # 解析提醒时间
        time_str = scfg.get("reminder_time", "09:00")
        try:
            hour, minute = map(int, time_str.split(":"))
        except ValueError:
            hour, minute = 9, 0

        _scheduler.add_job(
            run_reminder_check,
            trigger=CronTrigger(hour=hour, minute=minute),
            args=[config],
            id="daily_reminder",
            replace_existing=True,
            misfire_grace_time=3600,  # 错过1小时内仍补跑
        )

        # 通知重试: 每5分钟检查失败通知并重发
        from apscheduler.triggers.interval import IntervalTrigger
        _scheduler.add_job(
            run_retry_notifications,
            trigger=IntervalTrigger(minutes=5),
            args=[config],
            id="notification_retry",
            replace_existing=True,
            misfire_grace_time=300,
        )

        _scheduler.start()
        logger.info("调度器已启动: 每日 %02d:%02d 提醒扫描 + 每5min通知重试", hour, minute)

    # 启动时补跑逻辑
    if scfg.get("catch_up", True):
        _maybe_catch_up(config)

    return True


def stop_scheduler() -> None:
    """停止调度器 (应用退出时调用)。"""
    global _scheduler
    with _lock:
        if _scheduler and _scheduler.running:
            _scheduler.shutdown(wait=False)
            logger.info("调度器已停止")
        _scheduler = None


def _maybe_catch_up(config: dict) -> None:
    """若上次运行距今 >24h, 在后台线程立即补跑一次。"""
    state = _read_state(config)
    last_run = state.get("last_reminder_run", "")
    if last_run:
        try:
            last_dt = datetime.fromisoformat(last_run)
            if datetime.now() - last_dt < timedelta(hours=24):
                logger.debug("距上次运行不足24h, 无需补跑")
                return
        except ValueError:
            pass
    # 需要补跑
    logger.info("检测到超过24h未执行提醒扫描, 启动补跑...")
    t = threading.Thread(target=run_reminder_check, args=(config,), daemon=True)
    t.start()


# ─── Windows 计划任务注册 ───────────────────────────────────
def register_windows_task(config: dict, python_exe: Optional[str] = None) -> str:
    """生成 Windows Task Scheduler 注册命令 (需管理员权限执行)。

    返回可直接在 PowerShell (Admin) 中执行的命令字符串。
    应用关闭后, 系统级计划任务仍可兜底执行提醒扫描。
    """
    import os
    if python_exe is None:
        python_exe = sys.executable

    project_root = Path(__file__).parent.parent.resolve()
    script_path = project_root / "scripts" / "run_reminder.py"
    task_name = "VisionOCR_ReminderCheck"
    time_str = config.get("scheduler", {}).get("reminder_time", "09:00")

    # 生成独立执行脚本
    script_content = f'''"""VisionOCR 提醒扫描独立入口 (Windows 计划任务调用)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import load_config
from core.scheduler import run_reminder_check
if __name__ == "__main__":
    cfg = load_config()
    results = run_reminder_check(cfg)
    print(f"触发 {{len(results)}} 条提醒")
'''
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_content, encoding="utf-8")

    cmd = (
        f'Register-ScheduledTask -TaskName "{task_name}" '
        f'-Action (New-ScheduledTaskAction -Execute "{python_exe}" '
        f'-Argument "{script_path}") '
        f'-Trigger (New-ScheduledTaskTrigger -Daily -At "{time_str}") '
        f'-Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable '
        f'-ExecutionTimeLimit (New-TimeSpan -Minutes 10)) '
        f'-Description "VisionOCR Pro 每日回款提醒扫描 (应用外兜底)" '
        f'-RunLevel Highest'
    )
    return cmd


def get_scheduler_status(config: dict) -> dict:
    """获取调度器运行状态 (供 UI 展示)。"""
    state = _read_state(config)
    running = _scheduler is not None and _scheduler.running
    return {
        "apscheduler_running": running,
        "last_run": state.get("last_reminder_run", "从未执行"),
        "last_fired": state.get("last_fired_count", 0),
        "enabled": config.get("scheduler", {}).get("enabled", True),
        "reminder_time": config.get("scheduler", {}).get("reminder_time", "09:00"),
    }
