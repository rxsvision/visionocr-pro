"""引擎注册表 + LRU 显存管理

职责:
1. 注册/发现所有引擎
2. 按名称获取引擎实例
3. LRU 策略自动加载/卸载, 保证不超显存预算
4. 空闲超时自动卸载
"""
import logging
import threading
import time
from collections import OrderedDict
from typing import Optional

from engines.base import BaseEngine, EngineState

logger = logging.getLogger("visionocr.registry")


class EngineRegistry:
    def __init__(self, config: dict):
        self.config = config
        vram_cfg = config.get("vram", {})
        self.max_budget_gb: float = vram_cfg.get("max_budget_gb", 12.0)
        self.idle_unload_sec: int = vram_cfg.get("idle_unload_sec", 300)

        self._engines: dict[str, BaseEngine] = {}
        # LRU: 最近使用在前
        self._lru: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    # ─── 注册 ───────────────────────────────────────────────
    def register(self, engine: BaseEngine) -> None:
        name = engine.meta.name
        if name in self._engines:
            raise ValueError(f"Engine '{name}' already registered")
        self._engines[name] = engine

    # 引擎清单: (module_path, class_name)
    ENGINE_MANIFEST = [
        ("engines.ocr.ovisocr2", "OvisOCR2Engine"),
        ("engines.ocr.paddleocr_vl", "PaddleOCRVLEngine"),
        # ("engines.ocr.hunyuan_ocr", "HunyuanOCREngine"),  # 模块未实现, 待接入
        ("engines.ocr.rapidocr", "RapidOCREngine"),
        ("engines.ocr.unlimited_ocr", "UnlimitedOCREngine"),
        ("engines.ocr.mineru", "MinerUEngine"),
        ("engines.ocr.scene_classifier", "SceneClassifierEngine"),
        ("engines.vision.rfdetr", "RFDETREngine"),
        ("engines.vision.yolo26", "YOLO26Engine"),
        ("engines.vision.grounding_dino", "GroundingDINOEngine"),
        ("engines.vision.sam3", "SAM3Engine"),
        ("engines.vision.anomalib_engine", "AnomalibEngine"),
        ("engines.vision.barcode", "BarcodeEngine"),
        ("engines.pose.rtmpose", "RTMPoseEngine"),
        ("engines.pose.ctrgcn", "CTRGCNEngine"),
        ("engines.pose.fatigue", "FatigueEngine"),
        ("engines.llm.ollama_provider", "OllamaEngine"),
        ("engines.llm.api_provider", "APIEngine"),
    ]

    def register_all(self) -> None:
        """延迟导入并注册所有引擎, 缺失依赖的引擎静默跳过"""
        import importlib

        t0 = time.time()
        ok_count = 0
        skip_count = 0
        for module_path, class_name in self.ENGINE_MANIFEST:
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                self.register(cls(self.config))
                ok_count += 1
            except Exception as e:
                skip_count += 1
                logger.debug("跳过 %s: %s", class_name, e)
        elapsed = time.time() - t0
        logger.info("注册完成: %d 引擎, %d 跳过 (耗时 %.2fs)",
                    ok_count, skip_count, elapsed)

    # ─── 获取 ───────────────────────────────────────────────
    def get(self, name: str) -> Optional[BaseEngine]:
        return self._engines.get(name)

    def list_engines(self, category: str = "") -> list[dict]:
        result = []
        for eng in self._engines.values():
            if category and eng.meta.category != category:
                continue
            result.append({
                "name": eng.meta.name,
                "display_name": eng.meta.display_name,
                "category": eng.meta.category,
                "state": eng.state.value,
                "vram_gb": eng.meta.vram_gb,
                "license": eng.meta.license,
                "description": eng.meta.description,
            })
        return result

    # ─── LRU 加载/卸载 ─────────────────────────────────────
    def ensure_loaded(self, name: str) -> BaseEngine:
        """确保引擎已加载, 必要时驱逐 LRU 尾部引擎腾出显存"""
        engine = self._engines.get(name)
        if engine is None:
            raise KeyError(f"Engine '{name}' not registered")

        with self._lock:
            # 双重检查: 锁内再次确认状态 (防止并发重复加载)
            if engine.is_ready():
                self._touch(name)
                return engine

            needed = engine.meta.vram_gb
            # 驱逐直到有足够空间
            while self._used_vram() + needed > self.max_budget_gb and self._lru:
                evict_name, _ = self._lru.popitem(last=True)
                evict_eng = self._engines[evict_name]
                if evict_eng.is_ready():
                    logger.info("LRU 驱逐 %s 以释放显存", evict_name)
                    evict_eng.unload()

            engine.state = EngineState.LOADING
            engine.load()
            self._touch(name)
        return engine

    def unload(self, name: str) -> None:
        engine = self._engines.get(name)
        if engine and engine.is_ready():
            engine.unload()
            self._lru.pop(name, None)

    def unload_all(self) -> None:
        with self._lock:
            for name, eng in self._engines.items():
                if eng.is_ready():
                    eng.unload()
            self._lru.clear()

    # ─── 内部 ───────────────────────────────────────────────
    def _touch(self, name: str):
        self._lru.pop(name, None)
        self._lru[name] = time.time()

    def _used_vram(self) -> float:
        return sum(
            eng.meta.vram_gb
            for eng in self._engines.values()
            if eng.is_ready()
        )

    def status(self) -> dict:
        return {
            "max_budget_gb": self.max_budget_gb,
            "used_gb": round(self._used_vram(), 2),
            "loaded": [n for n, e in self._engines.items() if e.is_ready()],
            "registered": len(self._engines),
        }
