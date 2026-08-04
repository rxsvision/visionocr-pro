"""core/config_schema.py 单元测试 - 配置 schema 校验契约"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config, DEFAULT_CONFIG_PATH
from core.config_schema import AppConfig, ConfigValidationError, validate_config


def _write(tmp_path, text: str) -> Path:
    f = tmp_path / "config.yaml"
    f.write_text(text, encoding="utf-8")
    return f


class TestValidateConfig:
    def test_real_config_yaml_passes(self):
        """仓库自带 config.yaml 必须通过 schema 校验 (防 schema 与实际配置脱节)"""
        cfg = load_config(DEFAULT_CONFIG_PATH)
        assert cfg["server_port"] == 7860

    def test_defaults_pass(self, tmp_path):
        """配置文件缺失时的内置默认值必须通过校验"""
        cfg = load_config(tmp_path / "nonexist.yaml")
        assert validate_config(cfg) is cfg

    def test_empty_dict_passes(self):
        """空配置 = 全部默认值, 合法"""
        assert validate_config({}) == {}

    def test_invalid_port_type(self, tmp_path):
        with pytest.raises(ConfigValidationError, match="server_port"):
            load_config(_write(tmp_path, "server_port: not_a_number\n"))

    def test_port_out_of_range(self, tmp_path):
        with pytest.raises(ConfigValidationError, match="server_port"):
            load_config(_write(tmp_path, "server_port: 999999\n"))

    def test_invalid_device_enum(self, tmp_path):
        with pytest.raises(ConfigValidationError, match="device"):
            load_config(_write(tmp_path, "device: tpu\n"))

    def test_invalid_routing_policy(self, tmp_path):
        with pytest.raises(ConfigValidationError, match="routing"):
            load_config(_write(tmp_path,
                               "llm:\n  routing:\n    policy: yolo_mode\n"))

    def test_np_epsilon_range(self, tmp_path):
        with pytest.raises(ConfigValidationError, match="np_epsilon"):
            load_config(_write(tmp_path,
                               "qc:\n  patchcore:\n    np_epsilon: 1.5\n"))

    def test_confidence_threshold_range(self, tmp_path):
        with pytest.raises(ConfigValidationError, match="confidence_threshold"):
            load_config(_write(tmp_path,
                               "ocr:\n  confidence_threshold: 2.0\n"))

    def test_multiple_errors_aggregated(self, tmp_path):
        """多处错误一次性全部报出 (不做逐条挤牙膏)"""
        with pytest.raises(ConfigValidationError) as ei:
            load_config(_write(tmp_path,
                               "server_port: abc\ndevice: tpu\n"))
        msg = str(ei.value)
        assert "server_port" in msg and "device" in msg
        assert "2 处" in msg

    def test_numeric_string_coerced(self, tmp_path, monkeypatch):
        """环境变量替换产生的数值字符串通过 lax 模式转型校验"""
        monkeypatch.setenv("TEST_SCHEMA_PORT", "8090")
        cfg = load_config(_write(
            tmp_path, 'server_port: "${TEST_SCHEMA_PORT:-7860}"\n'))
        # 原值保留字符串形态 (向后兼容), 但校验已通过
        assert cfg["server_port"] in ("8090", 8090)

    def test_unknown_keys_preserved(self, tmp_path):
        """未建模键原样保留 (extra=allow, 前向兼容用户自定义)"""
        cfg = load_config(_write(
            tmp_path, "my_custom_section:\n  foo: 1\ndevice: cpu\n"))
        assert cfg["my_custom_section"] == {"foo": 1}

    def test_appconfig_defaults_complete(self):
        """AppConfig 可用纯默认值实例化 (与 _defaults() 对齐)"""
        m = AppConfig()
        assert m.server_port == 7860
        assert m.qc.union.enable_dinov2 is True
        assert m.vram.quantization == "q4"
