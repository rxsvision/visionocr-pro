"""VisionOCR Pro - 通用视觉识别与检测平台
启动: python app.py → http://localhost:7860
"""
import atexit
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.database import init_db
from core.scheduler import start_scheduler, stop_scheduler
from engines.registry import EngineRegistry
from ui.main import create_app, THEME, CSS


def main():
    cfg = load_config()

    # HuggingFace 镜像: 国内网络不稳定时通过 hf-mirror.com 加速下载
    import os
    hf_mirror = cfg.get("hf_mirror", "")
    if hf_mirror:
        os.environ.setdefault("HF_ENDPOINT", hf_mirror)

    init_db(cfg["data_dir"])
    registry = EngineRegistry(cfg)
    registry.register_all()

    # 注入 registry 到各面板
    from ui.tab_settings import set_registry as set_settings_registry
    from ui.tab_ocr import set_registry as set_ocr_registry
    from ui.tab_contract import set_registry as set_contract_registry
    from ui.tab_qc import set_registry as set_qc_registry
    set_settings_registry(registry)
    set_ocr_registry(registry)
    set_contract_registry(registry)
    set_qc_registry(registry)

    # 启动定时调度器 (提醒自动化)
    start_scheduler(cfg)
    atexit.register(stop_scheduler)

    app = create_app(cfg, registry)
    app.launch(
        server_name=cfg.get("server_name", "127.0.0.1"),
        server_port=cfg.get("server_port", 7860),
        share=False,
        inbrowser=True,
        theme=THEME,
        css=CSS,
    )


if __name__ == "__main__":
    main()
