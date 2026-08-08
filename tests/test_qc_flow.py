"""ui.qc_flow 单元测试 (Union/DINO/3D融合 编排逻辑)

覆盖 finding ui-hotspot-validation-gap 的修复: UI 编排层抽离后的
参数转发、结果装配、判定文案与落库路径。仅依赖 requirements-test.txt
最小依赖集 (不 import gradio), 可在 CI 最小环境下运行。
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ui import qc_flow
from ui.qc_flow import (
    assemble_dino_view, assemble_union_view, format_fusion_display,
    persist_qc_result, resolve_fusion_mode, resolve_product_name,
)
from core.database import get_conn

ROOT = Path(__file__).parent.parent


# ─── 参数转发 ────────────────────────────────────────────────

class TestResolveFusionMode:
    def test_and(self):
        assert resolve_fusion_mode("AND (高精确)") == "and"

    def test_depth_only(self):
        assert resolve_fusion_mode("仅深度") == "depth_only"

    def test_or_default(self):
        assert resolve_fusion_mode("OR (高召回, 推荐)") == "or"

    def test_none_defaults_or(self):
        assert resolve_fusion_mode(None) == "or"
        assert resolve_fusion_mode("") == "or"


class TestResolveProductName:
    def test_new_placeholder_maps_empty(self):
        assert resolve_product_name("(新建)") == ""

    def test_none_maps_empty(self):
        assert resolve_product_name(None) == ""

    def test_empty_stays_empty(self):
        assert resolve_product_name("") == ""

    def test_real_product_passthrough(self):
        assert resolve_product_name("门把手") == "门把手"


# ─── Union 结果装配 ──────────────────────────────────────────

def _union_result_four_sources():
    """四源 (PatchCore+DINOv2+DINO+YOLO) 全激活的 Union 结果样例。"""
    return {
        "verdict": "NG",
        "ng_sources": ["PatchCore", "DINO", "YOLO", "DINOv2"],
        "patchcore": {"score": 5.0},   # 无界量纲, 不应进入 max_score
        "dinov2": {"score": 3.2},      # 无界量纲, 不应进入 max_score
        "dino": {"max_score": 0.71,
                 "detections": [{"label": "划痕", "score": 0.71,
                                 "box": [10.4, 20.6, 30.5, 40.9]}]},
        "yolo": {"max_score": 0.82,
                 "boxes": [[1, 2, 3, 4]], "labels": ["dent"],
                 "scores": [0.82]},
        "fusion": {"mode": "staged", "stage": 2, "n_cal": 32},
        "image": "annotated.png",
    }


class TestAssembleUnionView:
    def test_row_order_and_numbering(self):
        view = assemble_union_view(_union_result_four_sources())
        # 行序: PatchCore → DINOv2 → DINO → YOLO, 编号连续
        labels = [r[1] for r in view.table]
        assert labels == ["[PatchCore] 表面异常", "[DINOv2] 表面异常",
                          "[DINO] 划痕", "[YOLO] dent"]
        assert [r[0] for r in view.table] == ["1", "2", "3", "4"]
        assert view.count_str == "4"

    def test_max_score_excludes_unbounded_sources(self):
        # patchcore 5.0 / dinov2 3.2 是无界分数, max_score 只取 dino/yolo
        view = assemble_union_view(_union_result_four_sources())
        assert view.max_score == pytest.approx(0.82)
        assert view.score_str == "82.00%"

    def test_unbounded_only_shows_placeholder_score(self):
        """仅距离分源 (patchcore/dinov2) 触发时不显示误导性的 0.00%。"""
        view = assemble_union_view(
            {"verdict": "NG", "ng_sources": ["PatchCore", "DINOv2"],
             "patchcore": {"score": 26.7}, "dinov2": {"score": 138.8}})
        assert view.score_str == "— (仅距离分源触发)"
        assert view.max_score == 0.0  # 落库数值语义不变

    def test_ok_verdict_keeps_percent_display(self):
        """全源未触发 (OK) 时保持百分比显示, 不出现占位文案。"""
        view = assemble_union_view({"verdict": "OK", "ng_sources": []})
        assert view.score_str == "0.00%"

    def test_detections_schema_for_persist(self):
        view = assemble_union_view(_union_result_four_sources())
        sources = [d["source"] for d in view.detections]
        assert sources == ["patchcore", "dinov2", "dino", "yolo"]
        assert view.detections[2]["label"] == "划痕"
        assert view.detections[3]["box"] == [1, 2, 3, 4]

    def test_ng_verdict_lists_sources(self):
        view = assemble_union_view(_union_result_four_sources())
        assert view.verdict_str == ("✗ NG (触发源: PatchCore+DINO+YOLO"
                                    "+DINOv2)")

    def test_ok_verdict(self):
        view = assemble_union_view({"verdict": "OK", "ng_sources": []})
        assert view.verdict_str == "✓ OK (合格)"
        assert view.table == []
        assert view.count_str == "0"

    def test_review_verdict_single_source(self):
        view = assemble_union_view(
            {"verdict": "REVIEW", "ng_sources": ["YOLO"],
             "yolo": {"max_score": 0.6, "boxes": [], "labels": [],
                      "scores": []}})
        assert "REVIEW 待人工复核" in view.verdict_str
        assert "YOLO" in view.verdict_str

    def test_status_fusion_or_mode(self):
        view = assemble_union_view(
            {"verdict": "OK", "ng_sources": [], "fusion": {"mode": "or"}})
        assert "融合: 纯OR (v1.3.0)" in view.status

    def test_status_fusion_staged_without_n_cal(self):
        view = assemble_union_view(
            {"verdict": "OK", "ng_sources": [],
             "fusion": {"mode": "staged", "stage": 1, "n_cal": None}})
        assert "阶段1" in view.status
        assert "n_cal=—" in view.status

    def test_status_product_default(self):
        view = assemble_union_view({"verdict": "OK", "ng_sources": []})
        assert "产品: 默认" in view.status
        view2 = assemble_union_view({"verdict": "OK", "ng_sources": []},
                                    product="门把手")
        assert "产品: 门把手" in view2.status


# ─── Grounding DINO 结果装配 ─────────────────────────────────

class TestAssembleDinoView:
    def _dino_result(self):
        return {
            "verdict": "NG",
            "max_score": 0.66,
            "count": 2,
            "image": "annotated.png",
            "detections": [
                {"label": "划痕", "score": 0.66, "box": [1.2, 2.3, 3.4, 4.5]},
                {"label": "污渍", "score": 0.41, "box": [5, 6, 7, 8]},
            ],
        }

    def test_ng_verdict_and_count(self):
        view = assemble_dino_view(self._dino_result(), "划痕.污渍", 0.3)
        assert view.verdict_str == "✗ NG (不合格 · 2处缺陷)"
        assert view.count_str == "2"
        assert view.score_str == "66.00%"

    def test_ok_verdict(self):
        r = self._dino_result()
        r.update(verdict="OK", count=0, detections=[])
        view = assemble_dino_view(r, "划痕", 0.3)
        assert view.verdict_str == "✓ OK (合格)"

    def test_table_box_format_uses_spaces(self):
        # DINO 表与 Union 表的位置格式不同 (逗号后带空格), 须保持原样
        view = assemble_dino_view(self._dino_result(), "划痕.污渍", 0.3)
        assert view.table[0][3] == "(1, 2, 3, 4)"
        assert view.table[0][2] == "66.00%"

    def test_status_truncates_long_prompt(self):
        long_prompt = "缺" * 100
        view = assemble_dino_view(self._dino_result(), long_prompt, 0.25)
        assert "缺" * 60 in view.status
        assert "缺" * 61 not in view.status
        assert "阈值: 0.25" in view.status

    def test_detections_passthrough_for_persist(self):
        view = assemble_dino_view(self._dino_result(), "划痕", 0.3)
        assert view.detections == self._dino_result()["detections"]


# ─── 3D 融合显示 ─────────────────────────────────────────────

class TestFormatFusionDisplay:
    def test_ok_shows_dash_score(self):
        fused = {"verdict": "OK", "count": 0, "fused_defects": []}
        verdict_str, score_str, count_str, table = format_fusion_display(fused)
        assert verdict_str == "✓ OK (合格)"
        assert score_str == "—"
        assert count_str == "0"
        assert table == []

    def test_ng_picks_max_confidence(self):
        fused = {"verdict": "NG", "count": 2, "fused_defects": [
            {"source": "dino", "type": "划痕", "confidence": 0.55,
             "bbox": [1, 2, 3, 4]},
            {"source": "depth", "type": "深度偏差", "confidence": 0.9,
             "bbox": [5, 6, 7, 8]},
        ]}
        verdict_str, score_str, count_str, table = format_fusion_display(fused)
        assert verdict_str == "✗ NG (不合格 · 2处)"
        assert score_str == "0.90"
        assert count_str == "2"
        assert table[1][1] == "depth · 深度偏差"
        assert table[1][2] == "90%"


# ─── 落库 ────────────────────────────────────────────────────

class TestPersistQcResult:
    def _write_png(self, tmp_path: Path, name: str = "input.png") -> Path:
        p = tmp_path / name
        p.write_bytes(b"\x89PNG-fake-qc-flow-test")
        return p

    def test_success_writes_row_and_copies_image(self, tmp_path):
        data_dir = str(tmp_path / "data")
        img = self._write_png(tmp_path)
        ok = persist_qc_result(str(img), "NG",
                               [{"source": "dino", "label": "划痕",
                                 "score": 0.71}],
                               0.71, "划痕.污渍", data_dir)
        assert ok is True

        conn = get_conn(data_dir)
        rows = conn.execute(
            "SELECT image_path, verdict, anomaly_score, defect_json, "
            "barcode_content FROM qc_results").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0]["verdict"] == "NG"
        assert rows[0]["anomaly_score"] == pytest.approx(0.71)
        assert json.loads(rows[0]["defect_json"])[0]["label"] == "划痕"
        assert rows[0]["barcode_content"] == "划痕.污渍"

        # 图片被复制到 qc_images/, 文件名 = sha1[:16] + 扩展名
        expected = (hashlib.sha1(img.read_bytes()).hexdigest()[:16] + ".png")
        copied = Path(rows[0]["image_path"])
        assert copied.name == expected
        assert copied.parent.name == "qc_images"
        assert copied.is_file()

    def test_failure_returns_false_without_raising(self, tmp_path):
        # data_dir 指向一个已存在的文件 → get_conn 建目录失败
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("x")
        img = self._write_png(tmp_path)
        ok = persist_qc_result(str(img), "OK", [], 0.0, "p",
                               str(blocker), warn=False)
        assert ok is False

    def test_missing_image_still_persists_row(self, tmp_path):
        # 源图不存在时 persist_qc_image 原样返回路径, 落库不阻断
        data_dir = str(tmp_path / "data")
        ok = persist_qc_result(str(tmp_path / "gone.png"), "OK", [], 0.0,
                               "p", data_dir)
        assert ok is True
        conn = get_conn(data_dir)
        n = conn.execute("SELECT COUNT(*) FROM qc_results").fetchone()[0]
        conn.close()
        assert n == 1


# ─── 依赖约束 ────────────────────────────────────────────────

def test_qc_flow_imports_without_gradio():
    """子进程导入 ui.qc_flow 后 gradio 不得进入 sys.modules。

    保证该模块在 requirements-test.txt 最小依赖 (无 gradio) 的
    CI 环境下可导入、可测。
    """
    code = ("import sys; sys.path.insert(0, r'{root}'); "
            "import ui.qc_flow; "
            "assert 'gradio' not in sys.modules").format(root=ROOT)
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
