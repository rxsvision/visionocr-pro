"""DINOv2 异常检测引擎烟雾测试: load → train → infer → save/load bank"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if len(sys.argv) < 2:
    sys.exit("用法: python scripts/smoke_dinov2_anomaly.py <KolektorSDD数据集目录>")
DATA_ROOT = Path(sys.argv[1])


def scan(root: Path):
    normals, defects = [], []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for img_p in sorted(d.glob("Part*.jpg")):
            lbl_p = img_p.with_name(img_p.stem + "_label.bmp")
            if not lbl_p.exists():
                continue
            lbl = cv2.imdecode(np.fromfile(str(lbl_p), dtype=np.uint8),
                               cv2.IMREAD_GRAYSCALE)
            if lbl is None:
                continue
            (defects if lbl.max() > 0 else normals).append(str(img_p))
    return normals, defects


def main():
    from engines.vision.dinov2_anomaly import DINOv2AnomalyEngine
    normals, defects = scan(DATA_ROOT)
    print(f"数据: 正常 {len(normals)}, 缺陷 {len(defects)}")

    eng = DINOv2AnomalyEngine({
        "device": "auto",
        "qc": {"dinov2": {"input_size": 518, "pca_dim": 64,
                          "n_etalons": 8, "np_epsilon": 0.02}},
    })
    t0 = time.time()
    eng.load()
    assert eng.is_ready(), f"加载失败 state={eng.state}"
    print(f"load: {time.time()-t0:.1f}s")

    train_paths = normals[:25]  # 10 张建库 + 校准 (train 内部留 20%)
    t0 = time.time()
    meta = eng.train(train_paths)
    print(f"train: {meta}, {time.time()-t0:.1f}s")
    assert not meta.get("error"), meta
    assert eng.has_bank

    # infer: 3 正常 + 3 缺陷
    print("\n--- infer ---")
    for tag, paths in (("正常", normals[100:103]), ("缺陷", defects[:3])):
        for p in paths:
            r = eng.infer(p)
            assert r.get("pred_label") in ("OK", "NG"), r
            amap = r["anomaly_map"]
            print(f"  [{tag}] score={r['score']:.3f} pred={r['pred_label']} "
                  f"thr={r['threshold_used']:.3f} map={amap.shape} "
                  f"calib={r.get('calibrated_score')} p={r.get('np_p_value')}")
            assert amap.min() >= 0 and amap.max() <= 1.001

    # save / load bank 往返
    with tempfile.TemporaryDirectory() as td:
        bp = Path(td) / "smoke.npz"
        eng.save_bank(bp, product_name="smoke")
        assert bp.exists()
        eng2 = DINOv2AnomalyEngine(eng.config)
        eng2.load()
        assert eng2.load_bank(bp), "load_bank 失败"
        assert eng2.has_bank
        assert eng2._np_calibrator is not None and \
            eng2._np_calibrator.is_fitted, "NP校准器未恢复"
        r1 = eng.infer(defects[0])
        r2 = eng2.infer(defects[0])
        assert abs(r1["score"] - r2["score"]) < 1e-3, \
            f"save/load 后分数不一致: {r1['score']} vs {r2['score']}"
        print(f"\nsave/load 往返一致: score={r1['score']}")

    print("\nSMOKE PASS")


if __name__ == "__main__":
    main()
