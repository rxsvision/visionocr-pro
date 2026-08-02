"""Phase 3 VLM 智能 ROI 裁切 — 真机 smoke (Ollama qwen3-vl + 真实工业图)

用法:
  python scripts/smoke_vlm_roi.py e2e <kolektor_root>
      全链路: DINOv2 建库 → 缺陷图推理得 anomaly_map → ROI 裁切 → VLM 解释
  python scripts/smoke_vlm_roi.py image <path> [--box x1,y1,x2,y2]
      单图直测: 检测框证据(可选) → ROI → VLM 解释; 无框则整图兜底

前置: Ollama 服务在线且已拉取 qwen3-vl:8b (ollama list 检查)。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.config import load_config  # noqa: E402
from core.vlm_explain import explain_union  # noqa: E402


class _Reg:
    """最小 registry 适配: 只暴露 vlm_explain 需要的接口。"""

    def __init__(self, engines: dict):
        self._e = engines

    def get(self, name):
        return self._e.get(name)

    def ensure_loaded(self, name):
        eng = self._e.get(name)
        if eng is not None and not eng.is_ready():
            eng.load()


def smoke_e2e(kolektor_root: Path, config: dict) -> bool:
    from core.imutils import imread_unicode
    from engines.vision.dinov2_anomaly import DINOv2AnomalyEngine
    import cv2

    # 扫描 kos01~kos05: 正常建库 + 找一张缺陷图
    normals, defect_img = [], None
    for d in sorted(kolektor_root.iterdir())[:5]:
        if not d.is_dir():
            continue
        for img_p in sorted(d.glob("Part*.jpg")):
            lbl = cv2.imdecode(
                np.fromfile(str(img_p.with_name(
                    img_p.stem + "_label.bmp")), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE)
            if lbl is None:
                continue
            if lbl.max() > 0 and defect_img is None:
                defect_img = str(img_p)
            elif lbl.max() == 0:
                normals.append(str(img_p))
    if defect_img is None or len(normals) < 10:
        print("SMOKE FAIL: 数据不足")
        return False
    print(f"[e2e] 建库 {len(normals)} 张, 缺陷图: {Path(defect_img).name}")

    dv = DINOv2AnomalyEngine(config)
    t0 = time.time()
    dv.load()
    dv.train(normals[:30])
    print(f"[e2e] DINOv2 建库 {time.time() - t0:.1f}s, "
          f"tau={dv._calibrated_threshold:.2f}")

    r = dv.infer(defect_img)
    print(f"[e2e] 缺陷图分数 {r['score']:.2f} "
          f"(pred={r['pred_label']})")

    # 组装 Union 风格结果 → 走 vlm_explain 全链路
    union_result = {"verdict": "NG", "ng_sources": ["dinov2"],
                    "anomaly_map": r.get("anomaly_map"),
                    "dino": None, "yolo": None}
    img = imread_unicode(defect_img)
    # anomaly_map 是 37x37 网格 → vlm_explain 内 select_rois 需要原图尺度,
    # 与 run_union_detection 一致: 外部先 resize 到原图
    import cv2 as _cv2
    union_result["anomaly_map"] = _cv2.resize(
        r["anomaly_map"].astype(np.float32),
        (img.shape[1], img.shape[0]))

    from engines.llm.ollama_provider import OllamaEngine
    vlm = OllamaEngine(config)
    vlm.load()
    reg = _Reg({"ollama_vlm": vlm, "dinov2_anomaly": dv})

    t0 = time.time()
    out = explain_union(reg, defect_img, union_result, config)
    dt = time.time() - t0
    if out.get("error"):
        print(f"SMOKE FAIL: {out['error']}")
        return False
    print(f"[e2e] ROI {len(out['rois'])} 个, VLM 耗时 {dt:.1f}s")
    print("─" * 50)
    print(out["summary"])
    print("─" * 50)
    ok = bool(out["texts"]) and all("解释失败" not in t
                                    for t in out["texts"])
    print("SMOKE PASS" if ok else "SMOKE FAIL: VLM 回答异常")
    return ok


def smoke_image(path: Path, box: list | None, config: dict) -> bool:
    from engines.llm.ollama_provider import OllamaEngine
    vlm = OllamaEngine(config)
    vlm.load()
    if not vlm.is_ready():
        print("SMOKE FAIL: VLM 未就绪 (Ollama 服务/模型检查)")
        return False
    reg = _Reg({"ollama_vlm": vlm})
    union_result = {"verdict": "NG", "ng_sources": ["manual"],
                    "anomaly_map": None, "dino": None,
                    "yolo": {"boxes": [box] if box else [],
                             "scores": [0.9] if box else [],
                             "labels": ["manual"] if box else [],
                             "count": 1 if box else 0}}
    t0 = time.time()
    out = explain_union(reg, str(path), union_result, config)
    dt = time.time() - t0
    if out.get("error"):
        print(f"SMOKE FAIL: {out['error']}")
        return False
    print(f"[image] ROI {len(out['rois'])} 个 "
          f"({[r['source'] for r in out['rois']]}), 耗时 {dt:.1f}s")
    print("─" * 50)
    print(out["summary"])
    print("─" * 50)
    ok = bool(out["texts"]) and all("解释失败" not in t
                                    for t in out["texts"])
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return ok


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    mode, target = sys.argv[1], sys.argv[2]
    config = load_config()

    if mode == "e2e":
        ok = smoke_e2e(Path(target), config)
    elif mode == "image":
        box = None
        if "--box" in sys.argv:
            i = sys.argv.index("--box")
            box = [float(v) for v in sys.argv[i + 1:i + 5]]
        ok = smoke_image(Path(target), box, config)
    else:
        print(f"未知模式: {mode}")
        sys.exit(2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
