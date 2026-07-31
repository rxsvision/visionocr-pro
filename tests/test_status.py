"""infer_stats + status + 结构化日志 测试"""
import json
import logging

from core import infer_stats
from core.status import collect_status, format_status_markdown


class _FakeMeta:
    def __init__(self, name):
        self.name = name


class _FakeEngine:
    def __init__(self, name):
        self.meta = _FakeMeta(name)


def setup_function():
    infer_stats.reset()


# ─── infer_stats ────────────────────────────────────────────

def test_record_and_stats():
    infer_stats.record("rapidocr", 0.10)
    infer_stats.record("rapidocr", 0.20)
    s = infer_stats.get_stats()
    assert s["rapidocr"]["count"] == 2
    assert s["rapidocr"]["last_ms"] == 200.0
    assert s["rapidocr"]["avg_ms"] == 150.0


def test_timer_records_on_success():
    with infer_stats.Timer("anomalib"):
        pass
    assert infer_stats.get_stats()["anomalib"]["count"] == 1


def test_timer_skips_on_exception():
    try:
        with infer_stats.Timer("grounding_dino"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert "grounding_dino" not in infer_stats.get_stats()


def test_reset_clears():
    infer_stats.record("x", 0.01)
    infer_stats.reset()
    assert infer_stats.get_stats() == {}


# ─── status 聚合 ────────────────────────────────────────────

class _FakeRegistry:
    def list_engines(self, category=""):
        return [
            {"name": "rapidocr", "display_name": "RapidOCR",
             "category": "ocr", "state": "ready", "vram_gb": 0.0},
            {"name": "anomalib", "display_name": "PatchCore",
             "category": "vision", "state": "unloaded", "vram_gb": 2.0},
        ]

    def status(self):
        return {"max_budget_gb": 12.0, "used_gb": 0.0,
                "loaded": ["rapidocr"], "registered": 2}


def test_collect_status_summary():
    infer_stats.record("rapidocr", 0.09)
    data = collect_status(_FakeRegistry())
    assert data["summary"]["ready"] == 1
    assert data["summary"]["unloaded"] == 1
    assert data["summary"]["total"] == 2
    names = {e["name"] for e in data["engines"]}
    assert names == {"rapidocr", "anomalib"}
    # 耗时统计已合并
    rc = next(e for e in data["engines"] if e["name"] == "rapidocr")
    assert rc["count"] == 1


def test_format_markdown_renders():
    infer_stats.record("rapidocr", 0.09)
    md = format_status_markdown(_FakeRegistry())
    assert "RapidOCR" in md
    assert "就绪" in md
    assert "预算" in md


# ─── 结构化日志 ─────────────────────────────────────────────

def test_json_formatter():
    from app import JsonFormatter
    fmt = JsonFormatter()
    record = logging.LogRecord(
        "visionocr.test", logging.INFO, __file__, 1,
        "引擎就绪 %s", ("rapidocr",), None)
    line = fmt.format(record)
    obj = json.loads(line)
    assert obj["level"] == "INFO"
    assert obj["logger"] == "visionocr.test"
    assert obj["msg"] == "引擎就绪 rapidocr"
