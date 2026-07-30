"""VisionOCR Pro - 通用视觉识别与检测平台
启动: python app.py → http://localhost:7860
"""
import atexit
import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 确保项目根目录在 sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ─── 日志系统 ─────────────────────────────────────────────────────────────────

LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "visionocr.log"
LOG_FORMAT = "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(verbose: bool = False) -> None:
    """配置全局日志: 控制台(INFO) + 文件轮转(DEBUG, 10MB×5)。"""
    LOG_DIR.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # 控制台: 工人/工程师看得到的输出
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATEFMT))
    root.addHandler(console)

    # 文件: 排障用, 保留最近 50MB 日志
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATEFMT))
    root.addHandler(file_handler)

    # 降低第三方库噪音
    logging.getLogger("gradio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


logger = logging.getLogger("visionocr.app")

# ─── 启动流程 ─────────────────────────────────────────────────────────────────


def _startup_checks(cfg: dict) -> None:
    """启动前环境检查, 失败时抛出明确异常。"""
    # 检查数据目录可写
    data_dir = Path(cfg.get("data_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    test_file = data_dir / ".write_test"
    try:
        test_file.write_text("ok")
        test_file.unlink()
    except OSError as e:
        raise RuntimeError(f"数据目录不可写: {data_dir} ({e})")


def _log_engine_health(registry) -> None:
    """启动后记录引擎健康状态。"""
    engines = registry.list_engines()

    ready, error, unloaded = [], [], []
    for eng in engines:
        name = eng.get("name", "?")
        state = eng.get("state", "unknown")
        if state == "ready":
            ready.append(name)
        elif state == "error":
            error.append(name)
        else:
            unloaded.append(name)

    logger.info("引擎健康: %d 就绪 / %d 异常 / %d 未加载",
                len(ready), len(error), len(unloaded))
    if ready:
        logger.info("  就绪: %s", ", ".join(ready))
    if error:
        logger.warning("  异常: %s", ", ".join(error))
    if unloaded:
        logger.debug("  未加载: %s", ", ".join(unloaded))

    if not ready and not unloaded:
        logger.error("无可用引擎! 请检查模型文件和依赖安装。")


def main():
    setup_logging(verbose="--verbose" in sys.argv or "-v" in sys.argv)
    logger.info("VisionOCR Pro 启动中 ...")

    try:
        from core.config import load_config
        cfg = load_config()
        logger.info("配置加载完成 (data_dir=%s)", cfg.get("data_dir", "data"))
    except FileNotFoundError as e:
        logger.critical("配置文件缺失: %s", e)
        logger.critical("请确认 config.yaml 存在于: %s", ROOT / "config.yaml")
        _fatal_exit("配置文件缺失", str(e))
        return
    except Exception as e:
        logger.critical("配置解析失败: %s", e)
        _fatal_exit("配置解析失败", str(e))
        return

    try:
        _startup_checks(cfg)
    except RuntimeError as e:
        logger.critical("环境检查失败: %s", e)
        _fatal_exit("环境检查失败", str(e))
        return

    # HuggingFace 镜像
    import os
    hf_mirror = cfg.get("hf_mirror", "")
    if hf_mirror:
        os.environ.setdefault("HF_ENDPOINT", hf_mirror)

    try:
        from core.database import init_db, backup_db
        init_db(cfg["data_dir"])
        backup_db(cfg["data_dir"])
        logger.info("数据库初始化完成")
    except Exception as e:
        logger.critical("数据库初始化失败: %s", e)
        _fatal_exit("数据库初始化失败", str(e))
        return

    try:
        from engines.registry import EngineRegistry
        registry = EngineRegistry(cfg)
        registry.register_all()
        _log_engine_health(registry)
    except Exception as e:
        logger.critical("引擎注册失败: %s", e)
        _fatal_exit("引擎注册失败", str(e))
        return

    # 注入 registry 到各面板
    try:
        from ui.tab_settings import set_registry as set_settings_registry
        from ui.tab_ocr import set_registry as set_ocr_registry
        from ui.tab_contract import set_registry as set_contract_registry
        from ui.tab_qc import set_registry as set_qc_registry
        set_settings_registry(registry)
        set_ocr_registry(registry)
        set_contract_registry(registry)
        set_qc_registry(registry)
    except Exception as e:
        logger.critical("UI 模块加载失败: %s", e)
        _fatal_exit("UI 模块加载失败", str(e))
        return

    # 启动定时调度器
    try:
        from core.scheduler import start_scheduler, stop_scheduler
        start_scheduler(cfg)
        atexit.register(stop_scheduler)
        logger.info("定时调度器已启动")
    except Exception as e:
        logger.warning("调度器启动失败 (非致命): %s", e)

    # 引擎预热: 加载默认 OCR 引擎 + dummy 推理, 消除首次操作延迟
    try:
        from core.warmup import warmup_engines
        warmup_report = warmup_engines(registry, cfg)
        if warmup_report["ocr"].get("ok"):
            logger.info("系统就绪 (预热 %.1fs), 可开始操作",
                        warmup_report["total_sec"])
        else:
            logger.warning("预热未完成, 首次操作可能较慢")
    except Exception as e:
        logger.warning("预热跳过 (非致命): %s", e)

    # 启动 Gradio
    try:
        from ui.main import create_app, THEME, CSS
        app = create_app(cfg, registry)
        port = cfg.get("server_port", 7860)
        logger.info("Web UI 启动: http://localhost:%d", port)
        app.launch(
            server_name=cfg.get("server_name", "127.0.0.1"),
            server_port=port,
            share=False,
            inbrowser=True,
            theme=THEME,
            css=CSS,
        )
    except OSError as e:
        if "address already in use" in str(e).lower() or "10048" in str(e):
            logger.critical("端口 %d 已被占用", cfg.get("server_port", 7860))
            _fatal_exit("端口被占用",
                        f"端口 {cfg.get('server_port', 7860)} 已被其他程序使用。\n"
                        f"请关闭占用程序, 或在 config.yaml 中修改 server_port。")
        else:
            logger.critical("启动失败: %s", e)
            _fatal_exit("启动失败", str(e))
    except Exception as e:
        logger.critical("未预期错误:\n%s", traceback.format_exc())
        _fatal_exit("未预期错误", str(e))


def _fatal_exit(title: str, detail: str) -> None:
    """致命错误: 打印用户可读信息, 保持窗口不关闭。"""
    msg = (
        f"\n{'='*60}\n"
        f"  VisionOCR Pro 启动失败\n"
        f"  原因: {title}\n"
        f"  详情: {detail}\n"
        f"{'='*60}\n"
        f"  日志文件: {LOG_FILE}\n"
        f"  按 Enter 退出 ...\n"
    )
    print(msg, file=sys.stderr)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    sys.exit(1)


if __name__ == "__main__":
    main()
