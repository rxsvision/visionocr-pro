"""配置加载 - 支持环境变量替换 (${VAR} / ${VAR:-default})"""
import os
import re
from pathlib import Path
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# 匹配 ${VAR_NAME} 或 ${VAR_NAME:-default_value}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _resolve_env_vars(value):
    """递归替换配置值中的 ${ENV_VAR} 引用。"""
    if isinstance(value, str):
        def _replacer(m):
            var_name = m.group(1)
            default = m.group(2) if m.group(2) is not None else ""
            return os.environ.get(var_name, default)
        return _ENV_PATTERN.sub(_replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并: override 覆盖 base (字典递归合并, 非字典直接覆盖)。"""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: Path | str | None = None, profile: str | None = None) -> dict:
    # 尝试加载 .env 文件 (可选依赖)
    try:
        from dotenv import load_dotenv
        env_file = DEFAULT_CONFIG_PATH.parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
    except ImportError:
        pass

    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        cfg = _defaults()
    else:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # 环境变量替换
    cfg = _resolve_env_vars(cfg)

    # Profile 覆盖 (部署环境分层: gpu-full / cpu-only / edge-jetson)
    if profile:
        profile_path = DEFAULT_CONFIG_PATH.parent / "profiles" / f"{profile}.yaml"
        if profile_path.exists():
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_cfg = yaml.safe_load(f) or {}
            profile_cfg = _resolve_env_vars(profile_cfg)
            cfg = _deep_merge(cfg, profile_cfg)
        else:
            import logging
            logging.getLogger("visionocr.config").warning(
                "Profile '%s' 不存在: %s (使用默认配置)", profile, profile_path)

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

    # Schema 校验 (类型/取值范围/枚举; 失败抛 ConfigValidationError 启动期快速暴露)
    try:
        from core.config_schema import validate_config
        cfg = validate_config(cfg)
    except ImportError:
        import logging
        logging.getLogger("visionocr.config").warning(
            "pydantic 未安装, 跳过配置 schema 校验")
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
        "vram": {"max_budget_gb": 12, "idle_unload_sec": 1800, "quantization": "q4"},
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
        "qc": {"anomaly_algorithm": "dinomaly", "confidence_threshold": 0.5,
               "union": {"enable_patchcore": True, "enable_dino": True,
                         "enable_yolo": True}},
        "yolo_defect": {"weights": "", "confidence_threshold": 0.25,
                        "imgsz": 1280},
        "behavior": {"enabled": False},
    }
