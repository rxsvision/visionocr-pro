"""pytest 共享 fixtures"""
import sys
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_data_dir(tmp_path):
    """临时数据目录 (含 visionocr.db)"""
    return str(tmp_path / "data")


@pytest.fixture
def sample_config(tmp_path):
    """最小可用配置 dict"""
    return {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "models_dir": str(tmp_path / "models"),
        "data_dir": str(tmp_path / "data"),
        "company": {"name": "测试公司", "aliases": ["测试"]},
        "export": {"dir": str(tmp_path / "exports")},
        "model_source": "local",
        "device": "cpu",
        "vram": {"max_budget_gb": 12, "idle_unload_sec": 300},
        "ocr": {"default_engine": "auto", "confidence_threshold": 0.75},
        "llm": {
            "routing": {"policy": "local_first_cloud_fallback",
                        "confidence_threshold": 0.6},
            "ollama": {"model": "qwen3-vl:8b", "host": "http://localhost:11434"},
        },
        "camera": {"type": "opencv", "index": 0},
    }
