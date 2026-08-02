"""P5 三棱镜成对数据诊断: DINOv2 打分分布 + 对照拼图 (验收判定用)。

用法: python scripts/diag_p5.py <ok_dir> <def_dir> <out_prefix>
输出: <out_prefix>_scores.txt, <out_prefix>_ok.jpg, <out_prefix>_def.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.eval_acceptance import CFG, list_images, filter_majority_shape  # noqa: E402
from core.imutils import imread_unicode  # noqa: E402
import cv2  # noqa: E402


def contact_sheet(paths, scores, out, title, cols=5, cell=(300, 220)):
    """按分数升序拼图, 每格标注序号与分数。"""
    order = np.argsort(scores)
    n = len(order)
    rows = (n + cols - 1) // cols
    cw, ch = cell
    sheet = np.full((rows * (ch + 24), cols * cw, 3), 40, dtype=np.uint8)
    for i, idx in enumerate(order):
        r, c = divmod(i, cols)
        img = imread_unicode(paths[idx])
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        thumb = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
        y0 = r * (ch + 24)
        sheet[y0:y0 + ch, c * cw:(c + 1) * cw] = thumb
        cv2.putText(sheet, f"#{idx} {scores[idx]:.1f}",
                    (c * cw + 6, y0 + ch + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(sheet, title, (8, 0), cv2.FONT_HERSHEY_SIMPLEX, 0, (0, 0, 0), 0)
    cv2.imencode(".jpg", sheet)[1].tofile(out)
    print(f"saved {out}")


def main():
    ok_dir, def_dir, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    ok_paths, _ = filter_majority_shape(list_images(Path(ok_dir)))
    def_paths, _ = filter_majority_shape(list_images(Path(def_dir)))
    print(f"OK {len(ok_paths)}, DEF {len(def_paths)}")

    from engines.vision.dinov2_anomaly import DINOv2AnomalyEngine
    dv = DINOv2AnomalyEngine(CFG)
    dv.load()
    dv.train(ok_paths)
    print(f"tau={dv._calibrated_threshold:.2f}")

    def score(paths):
        return np.array([dv.infer(p)["score"] for p in paths], dtype=np.float64)

    s_ok, s_def = score(ok_paths), score(def_paths)
    lines = [f"tau={dv._calibrated_threshold:.3f}",
             f"OK  n={len(s_ok)} min={s_ok.min():.2f} median={np.median(s_ok):.2f} max={s_ok.max():.2f}",
             f"DEF n={len(s_def)} min={s_def.min():.2f} median={np.median(s_def):.2f} max={s_def.max():.2f}",
             "", "OK scores (idx: path -> score):"]
    for i, (p, s) in enumerate(zip(ok_paths, s_ok)):
        lines.append(f"  OK {i:2d}: {Path(p).name} -> {s:.2f}")
    lines.append("DEF scores:")
    for i, (p, s) in enumerate(zip(def_paths, s_def)):
        lines.append(f"  DEF {i:2d}: {Path(p).name} -> {s:.2f}")
    out_txt = out_prefix + "_scores.txt"
    Path(out_txt).write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {out_txt}")

    contact_sheet(ok_paths, s_ok, out_prefix + "_ok.jpg", "P5 OK")
    contact_sheet(def_paths, s_def, out_prefix + "_def.jpg", "P5 DEF")


if __name__ == "__main__":
    main()
