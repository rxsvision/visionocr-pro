"""EngineRegistry 并发模型测试 (v1.5.0: per-engine 锁 + infer 租约)。"""
import threading
import time

import pytest

from engines.base import BaseEngine, EngineMeta, EngineState
from engines.registry import EngineRegistry


class FakeEngine(BaseEngine):
    """可控假引擎: 记录 load/unload 次数, 可注入加载延迟。"""

    def __init__(self, config, name, vram=1.0, load_delay=0.0,
                 resident=False):
        super().__init__(config)
        self._meta = EngineMeta(name=name, display_name=name,
                                category="test", vram_gb=vram,
                                resident=resident)
        self.load_calls = 0
        self.unload_calls = 0
        self._load_delay = load_delay

    @property
    def meta(self):
        return self._meta

    def load(self):
        self.load_calls += 1
        if self._load_delay:
            time.sleep(self._load_delay)
        self.state = EngineState.READY

    def infer(self, *a, **k):
        return {"ok": True}

    def unload(self):
        self.unload_calls += 1
        self.state = EngineState.UNLOADED


@pytest.fixture
def make_registry():
    regs = []

    def _make(budget=12.0, idle=0):
        reg = EngineRegistry({"vram": {"max_budget_gb": budget,
                                        "idle_unload_sec": idle}})
        regs.append(reg)
        return reg

    yield _make
    for reg in regs:
        reg.shutdown()


# ─── 基础 ────────────────────────────────────────────────────

def test_register_and_get(make_registry):
    reg = make_registry()
    eng = FakeEngine({}, "a")
    reg.register(eng)
    assert reg.get("a") is eng
    with pytest.raises(ValueError):
        reg.register(FakeEngine({}, "a"))


def test_ensure_loaded_unknown_raises(make_registry):
    reg = make_registry()
    with pytest.raises(KeyError):
        reg.ensure_loaded("nope")


def test_ensure_loaded_loads_once(make_registry):
    reg = make_registry()
    eng = FakeEngine({}, "a")
    reg.register(eng)
    reg.ensure_loaded("a")
    reg.ensure_loaded("a")
    assert eng.load_calls == 1
    assert eng.is_ready()


# ─── 并发加载: per-engine 锁串行化 ───────────────────────────

def test_concurrent_ensure_loaded_single_load(make_registry):
    reg = make_registry()
    eng = FakeEngine({}, "slow", load_delay=0.3)
    reg.register(eng)
    results = []

    def worker():
        results.append(reg.ensure_loaded("slow"))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert eng.load_calls == 1
    assert all(r is eng for r in results)


def test_load_failure_sets_error_state(make_registry):
    class FailEngine(FakeEngine):
        def load(self):
            self.load_calls += 1
            raise RuntimeError("boom")

    reg = make_registry()
    eng = FailEngine({}, "fail")
    reg.register(eng)
    with pytest.raises(RuntimeError):
        reg.ensure_loaded("fail")
    assert eng.state == EngineState.ERROR


# ─── LRU 驱逐 ────────────────────────────────────────────────

def test_lru_eviction_frees_budget(make_registry):
    reg = make_registry(budget=2.5)
    a = FakeEngine({}, "a", vram=2.0)
    b = FakeEngine({}, "b", vram=2.0)
    reg.register(a)
    reg.register(b)
    reg.ensure_loaded("a")
    reg.ensure_loaded("b")  # 预算不足 → 驱逐 a
    assert a.unload_calls == 1
    assert not a.is_ready()
    assert b.is_ready()


def test_resident_engine_never_evicted(make_registry):
    reg = make_registry(budget=2.5)
    a = FakeEngine({}, "a", vram=2.0, resident=True)
    b = FakeEngine({}, "b", vram=2.0)
    reg.register(a)
    reg.register(b)
    reg.ensure_loaded("a")
    reg.ensure_loaded("b")  # 无可驱逐对象, 告警但仍加载
    assert a.is_ready()
    assert b.is_ready()


# ─── infer 租约 ──────────────────────────────────────────────

def test_lease_blocks_eviction(make_registry):
    reg = make_registry(budget=2.5)
    a = FakeEngine({}, "a", vram=2.0)
    b = FakeEngine({}, "b", vram=2.0)
    reg.register(a)
    reg.register(b)
    reg.ensure_loaded("a")
    reg.acquire_lease("a")
    try:
        reg.ensure_loaded("b")  # a 有租约不可驱逐
        assert a.is_ready(), "有活跃租约的引擎不得被驱逐"
        assert a.unload_calls == 0
    finally:
        reg.release_lease("a")


def test_lease_context_manager(make_registry):
    reg = make_registry()
    eng = FakeEngine({}, "a")
    reg.register(eng)
    with reg.lease("a") as engine:
        assert engine is eng
        assert engine.is_ready()
        assert reg.status()["in_use"] == ["a"]
    assert reg.status()["in_use"] == []


def test_lease_released_on_exception(make_registry):
    reg = make_registry()
    eng = FakeEngine({}, "a")
    reg.register(eng)
    with pytest.raises(RuntimeError):
        with reg.lease("a"):
            raise RuntimeError("infer crashed")
    assert reg.status()["in_use"] == []


def test_unload_skipped_while_leased(make_registry):
    reg = make_registry()
    eng = FakeEngine({}, "a")
    reg.register(eng)
    reg.ensure_loaded("a")
    reg.acquire_lease("a")
    reg.unload("a")
    assert eng.is_ready(), "显式卸载必须跳过有租约的引擎"
    reg.release_lease("a")
    reg.unload("a")
    assert not eng.is_ready()
    assert eng.unload_calls == 1


def test_idle_unload_respects_lease(make_registry):
    """空闲超时卸载: 无租约→卸载; 有租约→跳过并续期。"""
    reg = make_registry(idle=1)
    leased = FakeEngine({}, "leased")
    reg.register(leased)
    reg.ensure_loaded("leased")
    reg.acquire_lease("leased")
    try:
        time.sleep(7)  # 等待空闲卸载线程至少跑一轮 (interval >= 5s)
        assert leased.is_ready(), "租约活跃期间空闲卸载不得生效"
    finally:
        reg.release_lease("leased")
