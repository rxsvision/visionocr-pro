"""配置加载"""
from pathlib import Path
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return _defaults()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # 路径解析为绝对路径
    root = path.parent
    for key in ("models_dir", "data_dir"):
        val = cfg.get(key, key.replace("_dir", "s"))
        p = Path(val)
        cfg[key] = str(p if p.is_absolute() else root / p)
    # export.dir 嵌套路径解析
    export_cfg = cfg.get("export", {})
    if isinstance(export_cfg, dict) and export_cfg.get("dir"):
        ep = Path(export_cfg["dir"])
        export_cfg["dir"] = str(ep if ep.is_absolute() else root / ep)
    return cfg


def _defaults() -> dict:
    root = DEFAULT_CONFIG_PATH.parent
    return {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "models_dir": str(root / "models"),
        "data_dir": str(root / "data"),
        "company": {"name": "", "aliases": []},
        "export": {"dir": str(root / "exports"), "excel_summary": True},
        "model_source": "huggingface",
        "device": "auto",
        "vram": {"max_budget_gb": 12, "idle_unload_sec": 300, "quantization": "q4"},
        "ocr": {"default_engine": "auto"},
        "llm": {
            "routing": {"policy": "local_first_cloud_fallback",
                        "confidence_threshold": 0.6,
                        "escalate_on_validation_fail": True},
            "ollama": {"model": "qwen3-vl:8b", "host": "http://localhost:11434", "timeout": 600},
            "api": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat",
                    "api_key": "", "timeout": 120},
        },
        "camera": {"type": "opencv", "index": 0},
        "barcode": {"engine": "zbar"},
        "qc": {"anomaly_algorithm": "dinomaly", "confidence_threshold": 0.5},
        "behavior": {"enabled": False},
    }
