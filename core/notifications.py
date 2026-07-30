"""系统通知 (付款提醒等)

安全: 所有动态文本经过 _sanitize() 清洗, 防止命令注入 (C1 修复)。
"""
import logging
import platform
import re
import subprocess

logger = logging.getLogger("visionocr.notifications")


def _sanitize(text: str, max_len: int = 200) -> str:
    """移除可用于命令注入的字符, 截断长度。"""
    # 移除引号、反引号、$、\、换行等危险字符
    text = re.sub(r'["\'`$\\;\n\r]', '', text)
    return text[:max_len]


def notify(title: str, message: str) -> None:
    """跨平台桌面通知"""
    system = platform.system()
    title = _sanitize(title, 100)
    message = _sanitize(message, 500)
    try:
        if system == "Windows":
            # PowerShell toast — 动态文本已清洗, 不含引号/$/反引号
            ps = (
                '[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, '
                'ContentType = WindowsRuntime] | Out-Null; '
                '$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(0); '
                '$text = $template.GetElementsByTagName("text"); '
                f'$text[0].AppendChild($template.CreateTextNode("{title}")) | Out-Null; '
                f'$text[1].AppendChild($template.CreateTextNode("{message}")) | Out-Null; '
                '$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("VisionOCR"); '
                '$notifier.Show([Windows.UI.Notifications.ToastNotification]::new($template))'
            )
            subprocess.run(["powershell", "-Command", ps], capture_output=True, timeout=5)
        elif system == "Linux":
            subprocess.run(["notify-send", title, message], capture_output=True, timeout=5)
        elif system == "Darwin":
            # osascript: 动态文本已清洗, 不含引号/反斜杠
            script = f'display notification "{message}" with title "{title}"'
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except Exception:
        logger.warning("[Notify] %s: %s", title, message)
