"""系统通知 (付款提醒等)"""
import platform
import subprocess


def notify(title: str, message: str) -> None:
    """跨平台桌面通知"""
    system = platform.system()
    try:
        if system == "Windows":
            # PowerShell toast notification
            ps = (
                f'[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, '
                f'ContentType = WindowsRuntime] | Out-Null; '
                f'$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(0); '
                f'$text = $template.GetElementsByTagName("text"); '
                f'$text[0].AppendChild($template.CreateTextNode("{title}")) | Out-Null; '
                f'$text[1].AppendChild($template.CreateTextNode("{message}")) | Out-Null; '
                f'$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("VisionOCR"); '
                f'$notifier.Show([Windows.UI.Notifications.ToastNotification]::new($template))'
            )
            subprocess.run(["powershell", "-Command", ps], capture_output=True, timeout=5)
        elif system == "Linux":
            subprocess.run(["notify-send", title, message], capture_output=True, timeout=5)
        elif system == "Darwin":
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
                capture_output=True, timeout=5,
            )
    except Exception:
        print(f"[Notify] {title}: {message}")
