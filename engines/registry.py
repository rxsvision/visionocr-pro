"""引擎注册表 + LRU 显存管理

职责:
1. 注册/发现所有引擎
2. 按名称获取引擎实例
3. LRU 策略自动加载/卸载, 保证不超显存预算
   (常驻引擎 resident=True 永不驱逐, v1.3.0)
4. 空闲超时自动卸载 (后台线程, idle_unload_sec 生效, v1.3.0)

并发模型 (v1.5.0 重构, 修复 C-1):
- per-engine 锁: 模型加载/卸载只锁该引擎自身, 不再用全局锁阻塞
  其他引擎的加载与推理调度 (此前全局锁在 load() 期间持有数秒)。
- infer 租约: 推理期间持有租约, 驱逐/空闲卸载/显式卸载均跳过
  有活跃租约的引擎, 杜绝"推理中引擎被卸载"崩溃。
- 锁序约定: 引擎槽锁 → 簿记锁 (self._lock); 持 self._lock 时
  禁止再获取引擎槽锁, 避免死锁。
"""
import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Optional

from engines.base import BaseEngine, EngineState

logger = logging.getLogger("visionocr.registry")


class _EngineSlot:
    """单引擎并发槽: 串行化该引擎的 load/unload, 并跟踪推理租约。"""

    __slots__ = ("cv", "in_use")

    def __init__(self):
        self.cv = threading.Condition(threading.Lock())
        self.in_use = 0  # 活跃 infer 租约数


class EngineRegistry:
    def __init__(self, config: dict):
        self.config = config
        vram_cfg = config.get("vram", {})
        self.max_budget_gb: float = vram_cfg.get("max_budget_gb", 12.0)
        self.idle_unload_sec: int = vram_cfg.get("idle_unload_sec", 300)

        self._engines: dict[str, BaseEngine] = {}
        self._slots: dict[str, _EngineSlot] = {}
        # LRU: 最近使用在前 (常驻引擎不入队, 天然豁免驱逐/空闲卸载)
        self._lru: OrderedDict[str, float] = OrderedDict()
        # 簿记锁: 仅保护 _lru 读写, 禁止在持锁期间做模型加载/卸载
        self._lock = threading.Lock()

        # v1.3.0: 空闲超时卸载后台线程 (此前 idle_unload_sec 为死配置)
        self._stop_idle = threading.Event()
        self._idle_thread: Optional[threading.Thread] = None
        if self.idle_unload_sec > 0:
            self._idle_thread = threading.Thread(
                target=self._idle_unload_loop, daemon=True,
                name="registry-idle-unload")
            self._idle_thread.start()

    # ─── 注册 ───────────────────────────────────────────────
    def register(self, engine: BaseEngine) -> None:
        name = engine.meta.name
        if name in self._engines:
            raise ValueError(f"Engine '{name}' already registered")
        self._engines[name] = engine
        self._slots[name] = _EngineSlot()

    # 引擎清单: (module_path, class_name)
    ENGINE_MANIFEST = [
        ("engines.ocr.ppocrv6", "PPOCRv6Engine"),
        ("engines.ocr.ovisocr2", "OvisOCR2Engine"),
        ("engines.ocr.paddleocr_vl", "PaddleOCRVLEngine"),
        # ("engines.ocr.hunyuan_ocr", "HunyuanOCREngine"),  # 模块未实现, 待接入
        ("engines.ocr.rapidocr", "RapidOCREngine"),
        ("engines.ocr.unlimited_ocr", "UnlimitedOCREngine"),
        ("engines.ocr.mineru", "MinerUEngine"),
        ("engines.ocr.scene_classifier", "SceneClassifierEngine"),
        ("engines.vision.rfdetr", "RFDETREngine"),
        ("engines.vision.yolo_defect", "YOLODefectEngine"),
        # ("engines.vision.yolo26", "YOLO26Engine"),  # 边缘轻量检测 stub, 待选型
        ("engines.vision.grounding_dino", "GroundingDINOEngine"),
        ("engines.vision.sam3", "SAM3Engine"),
        ("engines.vision.anomalib_engine", "AnomalibEngine"),
        ("engines.vision.dinov2_anomaly", "DINOv2AnomalyEngine"),
        ("engines.vision.subspace_ad", "SubspaceADEngine"),
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
            except ImportError as e:
                # 依赖缺失属预期跳过 (如 GPU 库未安装), 仅记 debug
                skip_count += 1
                logger.debug("跳过 %s (依赖缺失): %s", class_name, e)
            except Exception as e:
                # 非预期异常 (引擎初始化 bug/配置错误), 需引起注意
                skip_count += 1
                logger.warning("引擎 %s 注册失败: %s", class_name, e)
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

    # ─── infer 租约 ─────────────────────────────────────────
    def acquire_lease(self, name: str) -> None:
        """登记一次推理占用 (须与 release_lease 配对)。"""
        slot = self._slots.get(name)
        if slot is None:
            return
        with slot.cv:
            slot.in_use += 1

    def release_lease(self, name: str) -> None:
        slot = self._slots.get(name)
        if slot is None:
            return
        with slot.cv:
            slot.in_use = max(0, slot.in_use - 1)
            slot.cv.notify_all()

    @contextmanager
    def lease(self, name: str):
        """推理租约上下文: ensure_loaded + 持有租约, 期间引擎不会被卸载。

        用法:
            with registry.lease("anomalib") as engine:
                result = engine.infer(path)
        """
        engine = self.ensure_loaded(name)
        self.acquire_lease(name)
        try:
            yield engine
        finally:
            self.release_lease(name)

    # ─── LRU 加载/卸载 ─────────────────────────────────────
    def ensure_loaded(self, name: str) -> BaseEngine:
        """确保引擎已加载, 必要时驱逐 LRU 尾部引擎腾出显存。

        并发: 仅持有该引擎自身的槽锁; 慢速 load() 不阻塞其他引擎。
        """
        engine = self._engines.get(name)
        if engine is None:
            raise KeyError(f"Engine '{name}' not registered")

        # 快路径: 已就绪, 无需加槽锁
        if engine.is_ready():
            self._touch(name)
            return engine

        slot = self._slots[name]
        with slot.cv:
            # 双重检查: 槽锁内再次确认 (防止并发重复加载)
            if engine.is_ready():
                self._touch(name)
                return engine

            needed = engine.meta.vram_gb
            # 驱逐直到有足够空间 (常驻引擎 resident=True 永不驱逐;
            # 有活跃推理租约的引擎跳过, 防推理中被卸载)
            excluded = {name}
            while self._used_vram() + needed > self.max_budget_gb:
                victim = self._pick_eviction_victim(excluded)
                if victim is None:
                    logger.warning("显存预算不足但无可驱逐引擎 "
                                   "(常驻/在用引擎受保护), 仍需 %.1fGB", needed)
                    break
                if not self._try_unload(victim):
                    excluded.add(victim)  # 在用: 跳过换下一个
                    continue
                logger.info("LRU 驱逐 %s 以释放显存", victim)

            engine.state = EngineState.LOADING
            try:
                engine.load()
            except Exception:
                engine.state = EngineState.ERROR
                raise
            self._touch(name)
        return engine

    def unload(self, name: str) -> None:
        """显式卸载 (UI 面板触发)。有活跃推理租约时跳过并告警。"""
        engine = self._engines.get(name)
        if not engine or not engine.is_ready():
            return
        if not self._try_unload(name):
            logger.warning("卸载 %s 被跳过: 推理租约活跃中", name)

    def unload_all(self) -> None:
        for name, eng in list(self._engines.items()):
            if eng.is_ready():
                if not self._try_unload(name):
                    logger.warning("unload_all 跳过 %s: 推理租约活跃中", name)

    # ─── 内部 ───────────────────────────────────────────────
    def _touch(self, name: str):
        with self._lock:
            engine = self._engines.get(name)
            # 常驻引擎不入 LRU 队列 → 永不驱逐/空闲卸载 (v1.3.0)
            if engine is not None and getattr(engine.meta, "resident", False):
                self._lru.pop(name, None)
                return
            self._lru.pop(name, None)
            self._lru[name] = time.time()

    def _pick_eviction_victim(self, excluded: set[str]) -> Optional[str]:
        """按最久未使用顺序, 挑选第一个非常驻、已加载、无租约的驱逐候选。"""
        with self._lock:
            names = list(self._lru.keys())
        # OrderedDict 尾部为最近使用 → 从头部开始即最久未使用
        for name in names:
            if name in excluded:
                continue
            eng = self._engines.get(name)
            if eng is None:
                continue
            if getattr(eng.meta, "resident", False):
                continue
            if not eng.is_ready():
                continue
            slot = self._slots.get(name)
            if slot is not None and slot.in_use > 0:
                continue  # 推理中: 不可驱逐
            return name
        return None

    def _try_unload(self, name: str) -> bool:
        """在该引擎槽锁内安全卸载。返回 False 表示有活跃租约被跳过。"""
        eng = self._engines.get(name)
        slot = self._slots.get(name)
        if slot is None:
            return True
        with slot.cv:
            if slot.in_use > 0:
                return False
            if eng is not None and eng.is_ready():
                try:
                    eng.unload()
                except Exception as e:
                    logger.warning("卸载 %s 失败: %s", name, e)
        with self._lock:
            self._lru.pop(name, None)
        return True

    def _idle_unload_loop(self):
        """后台线程: 卸载空闲超时引擎 (v1.3.0 让 idle_unload_sec 生效)。

        常驻引擎不在 LRU 队列中, 天然豁免。
        有活跃推理租约的引擎跳过并续期, 下轮再试 (v1.5.0)。
        """
        interval = max(5, min(60, self.idle_unload_sec // 10))
        while not self._stop_idle.wait(interval):
            now = time.time()
            with self._lock:
                expired = [n for n, t in list(self._lru.items())
                           if now - t > self.idle_unload_sec]
                for name in expired:
                    self._lru.pop(name, None)
            for name in expired:
                eng = self._engines.get(name)
                if eng is None or not eng.is_ready():
                    continue
                if self._try_unload(name):
                    logger.info("空闲超时 (%ds), 卸载 %s",
                                self.idle_unload_sec, name)
                else:
                    # 推理中: 续期, 避免下一轮立即重试抖动
                    logger.debug("空闲卸载 %s 跳过: 推理租约活跃", name)
                    self._touch(name)

    def shutdown(self):
        """停止空闲卸载线程并卸载所有已加载引擎 (进程退出清理)。

        常驻引擎也在此卸载 — 常驻豁免的是"运行期"驱逐/空闲卸载,
        进程退出时仍应释放资源 (如 PP-OCRv6 常驻容器)。
        有活跃租约的引擎最多等待 30s (v1.5.0)。
        """
        self._stop_idle.set()
        deadline = time.time() + 30
        for name, slot in self._slots.items():
            with slot.cv:
                while slot.in_use > 0:
                    remain = deadline - time.time()
                    if remain <= 0:
                        logger.warning("shutdown: %s 租约未释放, 强制继续", name)
                        break
                    slot.cv.wait(timeout=remain)
        for name, eng in list(self._engines.items()):
            if eng.is_ready():
                try:
                    eng.unload()
                except Exception as e:
                    logger.warning("退出卸载 %s 失败: %s", name, e)

    def _used_vram(self) -> float:
        return sum(
            eng.meta.vram_gb
            for eng in self._engines.values()
            if eng.is_ready()
        )

    def status(self) -> dict:
        with self._lock:
            loaded = [n for n, e in self._engines.items() if e.is_ready()]
            return {
                "max_budget_gb": self.max_budget_gb,
                "used_gb": round(self._used_vram(), 2),
                "loaded": loaded,
                "resident": [n for n, e in self._engines.items()
                             if e.is_ready()
                             and getattr(e.meta, "resident", False)],
                "in_use": [n for n, s in self._slots.items()
                           if s.in_use > 0],
                "idle_unload_sec": self.idle_unload_sec,
                "registered": len(self._engines),
            }
