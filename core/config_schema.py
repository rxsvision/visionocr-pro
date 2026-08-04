"""配置 schema 校验 (pydantic v2) - 启动期暴露配置错误

策略:
- `load_config()` 返回的 dict 先经 `validate_config()` 校验 (类型/取值范围/枚举)。
- 所有模型 `extra="allow"`: 未建模的键原样保留, 前向兼容用户自定义配置。
- 数值字段容忍字符串 (环境变量替换后可能是 "7860"), pydantic lax 模式自动转型。
- 校验失败抛 ConfigValidationError, 聚合全部问题一次报出 (启动期快速失败)。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger("visionocr.config")


class ConfigValidationError(ValueError):
    """配置校验失败 (聚合全部问题)。"""


class _M(BaseModel):
    """配置子模型基类: 允许未建模键原样通过。"""
    model_config = ConfigDict(extra="allow")


# ─── 顶层杂项 ────────────────────────────────────────────────────────────────

class CompanyCfg(_M):
    name: str = ""
    aliases: List[str] = []


class ExportCfg(_M):
    dir: str = "exports"
    enabled: List[str] = []
    excel_summary: bool = True


class NotifyCfg(_M):
    desktop_fallback: bool = True


class SchedulerCfg(_M):
    enabled: bool = True
    reminder_time: str = "09:00"
    catch_up: bool = True


class VramCfg(_M):
    max_budget_gb: float = Field(default=12, gt=0)
    idle_unload_sec: int = Field(default=1800, ge=0)
    quantization: Literal["q4", "q5", "q8", "fp16"] = "q4"


# ─── OCR ─────────────────────────────────────────────────────────────────────

class SceneClassifierCfg(_M):
    enabled: bool = True
    confidence_threshold: float = Field(default=0.7, ge=0, le=1)


class PreprocessCfg(_M):
    enabled: bool = True
    upscale_factor: float = Field(default=2.0, gt=0)


class PpOcrCfg(_M):
    gpu: bool = True
    timeout: int = Field(default=120, gt=0)
    port: int = Field(default=8686, gt=0, lt=65536)
    startup_timeout: int = Field(default=120, gt=0)


class OcrCfg(_M):
    default_engine: str = "rapidocr"
    fallback_engine: str = "rapidocr"
    confidence_threshold: float = Field(default=0.75, ge=0, le=1)
    scene_classifier: SceneClassifierCfg = SceneClassifierCfg()
    preprocess: PreprocessCfg = PreprocessCfg()
    ppocrv6: PpOcrCfg = PpOcrCfg()


# ─── LLM ─────────────────────────────────────────────────────────────────────

class RoutingCfg(_M):
    policy: Literal["local_only", "local_first_cloud_fallback",
                    "cloud_first"] = "local_first_cloud_fallback"
    confidence_threshold: float = Field(default=0.6, ge=0, le=1)
    escalate_on_validation_fail: bool = True


class OllamaCfg(_M):
    model: str = "qwen3-vl:8b"
    host: str = "http://localhost:11434"
    timeout: int = Field(default=600, gt=0)


class LlmApiCfg(_M):
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    api_key: str = ""
    timeout: int = Field(default=120, gt=0)


class LlmCfg(_M):
    routing: RoutingCfg = RoutingCfg()
    ollama: OllamaCfg = OllamaCfg()
    api: LlmApiCfg = LlmApiCfg()


# ─── 相机 / 3D ───────────────────────────────────────────────────────────────

class CameraCfg(_M):
    type: Literal["opencv", "gigevision", "hikvision"] = "hikvision"
    index: int = Field(default=0, ge=0)
    trigger: Literal["software", "hardware"] = "software"
    exposure_us: int = Field(default=5000, gt=0)


class SizectorCfg(_M):
    enabled: bool = True
    index: int = Field(default=0, ge=0)
    working_mode: Literal["fastest", "fast", "standard", "precise",
                          "super_precise", "dynamic"] = "precise"
    timeout_ms: int = Field(default=5000, gt=0)
    mock: bool = False


# ─── 质检 ────────────────────────────────────────────────────────────────────

class NpEngineCfg(_M):
    """共享 NP 校准语义的引擎段 (patchcore/dinov2/subspacead)。"""
    np_epsilon: float = Field(default=0.10, gt=0, lt=1)


class UnionFusionCfg(_M):
    mode: Literal["staged", "or"] = "staged"
    stage2_min_cal: int = Field(default=10, ge=0)
    stage3_min_cal: int = Field(default=50, ge=0)
    drift_window: int = Field(default=200, gt=0)


class UnionCfg(_M):
    enable_patchcore: bool = True
    enable_dino: bool = True
    enable_yolo: bool = True
    enable_dinov2: bool = True
    fusion: UnionFusionCfg = UnionFusionCfg()


class DefectSizeCfg(_M):
    enabled: bool = True
    min_area_px: int = Field(default=100, ge=0)
    max_area_px: int = Field(default=500000, ge=0)
    mode: Literal["bbox", "contour"] = "bbox"


class VlmExplainCfg(_M):
    enabled: bool = True
    max_rois: int = Field(default=3, ge=1)
    pad_frac: float = Field(default=0.25, ge=0)
    rel_thresh: float = Field(default=0.45, ge=0, le=1)
    max_tokens: int = Field(default=512, gt=0)


class QcCfg(_M):
    anomaly_algorithm: Literal["patchcore", "dinomaly", "visualad"] = "patchcore"
    grounding_dino_model: Literal["tiny", "base"] = "tiny"
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    auto_ng_alert: bool = True
    patchcore: NpEngineCfg = NpEngineCfg()
    dinov2: NpEngineCfg = NpEngineCfg()
    subspacead: NpEngineCfg = NpEngineCfg()
    union: UnionCfg = UnionCfg()
    defect_size: DefectSizeCfg = DefectSizeCfg()
    vlm_explain: VlmExplainCfg = VlmExplainCfg()


class YoloDefectCfg(_M):
    weights: str = ""
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    imgsz: int = Field(default=1280, gt=0)


class BehaviorCfg(_M):
    enabled: bool = False
    fatigue_threshold: float = Field(default=0.7, ge=0, le=1)


class BarcodeCfg(_M):
    engine: Literal["zbar", "dynamsoft"] = "zbar"


# ─── 顶层 ────────────────────────────────────────────────────────────────────

class AppConfig(_M):
    server_name: str = "127.0.0.1"
    server_port: int = Field(default=7860, gt=0, lt=65536)
    ui_password: str = ""
    models_dir: str = "models"
    data_dir: str = "data"
    company: CompanyCfg = CompanyCfg()
    export: ExportCfg = ExportCfg()
    notify: NotifyCfg = NotifyCfg()
    scheduler: SchedulerCfg = SchedulerCfg()
    model_source: Literal["huggingface", "modelscope", "local"] = "huggingface"
    hf_mirror: str = ""
    device: Literal["cuda", "cpu", "auto"] = "auto"
    vram: VramCfg = VramCfg()
    ocr: OcrCfg = OcrCfg()
    llm: LlmCfg = LlmCfg()
    camera: CameraCfg = CameraCfg()
    sizector: SizectorCfg = SizectorCfg()
    default_scenes: List[str] = []
    barcode: BarcodeCfg = BarcodeCfg()
    qc: QcCfg = QcCfg()
    yolo_defect: YoloDefectCfg = YoloDefectCfg()
    behavior: BehaviorCfg = BehaviorCfg()


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """校验配置 dict, 通过则原样返回; 失败抛 ConfigValidationError (聚合全部问题)。"""
    try:
        AppConfig.model_validate(cfg)
    except ValidationError as e:
        lines = []
        for err in e.errors():
            loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
            lines.append(f"  - {loc}: {err.get('msg')} (输入值: {err.get('input')!r})")
        detail = "\n".join(lines)
        logger.error("config.yaml 校验失败 (%d 处):\n%s", len(lines), detail)
        raise ConfigValidationError(
            f"config.yaml 校验失败 ({len(lines)} 处):\n{detail}") from e
    return cfg
