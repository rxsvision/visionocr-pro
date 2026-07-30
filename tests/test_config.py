"""core/config.py 单元测试"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config


class TestLoadConfig:
    def test_missing_file_returns_defaults(self, tmp_path):
        """配置文件不存在时返回默认值"""
        cfg = load_config(tmp_path / "nonexist.yaml")
        assert cfg["server_port"] == 7860
        assert cfg["device"] == "auto"

    def test_loads_yaml(self, tmp_path):
        """正常 YAML 加载"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "server_port: 8080\ndevice: cuda\n", encoding="utf-8")
        cfg = load_config(yaml_file)
        assert cfg["server_port"] == 8080
        assert cfg["device"] == "cuda"

    def test_paths_resolved_absolute(self, tmp_path):
        """相对路径解析为绝对路径"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("data_dir: mydata\n", encoding="utf-8")
        cfg = load_config(yaml_file)
        assert Path(cfg["data_dir"]).is_absolute()
        assert "mydata" in cfg["data_dir"]


class TestEnvVarSubstitution:
    def test_env_var_resolved(self, tmp_path, monkeypatch):
        """${VAR:-default} 环境变量替换"""
        monkeypatch.setenv("TEST_PORT", "9999")
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            'server_port: "${TEST_PORT:-7860}"\n', encoding="utf-8")
        cfg = load_config(yaml_file)
        assert cfg["server_port"] == "9999"

    def test_env_var_default_fallback(self, tmp_path, monkeypatch):
        """环境变量未设置时使用默认值"""
        monkeypatch.delenv("UNSET_VAR_XYZ", raising=False)
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            'device: "${UNSET_VAR_XYZ:-cpu}"\n', encoding="utf-8")
        cfg = load_config(yaml_file)
        assert cfg["device"] == "cpu"
