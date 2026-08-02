"""NP校准 KolektorSDD 实测验证脚本

目标 (可验证):
1. NP阈值下, 留出正常样本经验FPR ≤ epsilon + 波动容差
2. 缺陷Recall不低于 legacy P99×1.2 阈值 (零漏检不回退)
3. 输出 image-level AUROC 与两种阈值的完整对比

数据: KolektorSDD (50个器件 × 8面, label.bmp掩码标注缺陷)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if len(sys.argv) < 2:
    sys.exit("用法: python scripts/eval_np_calibration.py <KolektorSDD数据集目录>")
DATA_ROOT = Path(sys.argv[1])


def imread_u(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8),
                        cv2.IMREAD_COLOR)


def scan_dataset(root: Path):
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


def auroc(scores_pos, scores_neg):
    """Rank-based AUROC (缺陷=正类)."""
    pos = np.asarray(scores_pos)
    neg = np.asarray(scores_neg)
    all_s = np.concatenate([pos, neg])
    order = np.argsort(all_s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    # 处理并列: 平均秩
    sorted_s = all_s[order]
    i = 0
    n = len(all_s)
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    rank_sum_pos = ranks[:len(pos)].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main():
    print(f"数据集: {DATA_ROOT}")
    normals, defects = scan_dataset(DATA_ROOT)
    print(f"扫描完成: 正常 {len(normals)} 张, 缺陷 {len(defects)} 张")
    assert len(normals) > 50 and len(defects) > 10, "数据集不完整"

    # 划分: 80% 正常 → 建库+校准, 20% 正常 → 留出测FPR
    rng = np.random.default_rng(2026)
    idx = rng.permutation(len(normals))
    n_train = int(len(normals) * 0.8)
    train_paths = [normals[i] for i in idx[:n_train]]
    holdout_paths = [normals[i] for i in idx[n_train:]]
    print(f"划分: train={len(train_paths)}, holdout={len(holdout_paths)}, "
          f"defect={len(defects)}")

    # 构建引擎 (GPU, epsilon=0.02)
    from engines.vision.anomalib_engine import AnomalibEngine
    config = {
        "device": "auto",
        "qc": {"patchcore": {
            "input_size": 512, "coreset_ratio": 0.1,
            "conservative_mode": True, "np_epsilon": 0.10,
        }},
    }
    eng = AnomalibEngine(config)
    t0 = time.time()
    eng.load()
    assert eng.is_ready(), "引擎加载失败"

    t0 = time.time()
    meta = eng.train(train_paths)
    print(f"建库完成: {meta}, 耗时 {time.time()-t0:.1f}s")

    calib = eng._np_calibrator
    assert calib is not None and calib.is_fitted, "NP校准器未拟合"
    tau_np = calib.threshold
    cal = np.asarray(eng._train_scores)
    tau_legacy = float(np.percentile(cal, 99)) * 1.2
    print(f"阈值对比: NP tau={tau_np:.4f} (eps={calib.epsilon}) | "
          f"legacy P99x1.2={tau_legacy:.4f} | n_cal={len(cal)}")

    # 推理: 留出正常 + 缺陷
    def score_paths(paths, tag):
        scores = []
        t0 = time.time()
        for p in paths:
            r = eng.infer(p)
            if r.get("error"):
                print(f"  [WARN] {tag} 推理失败: {p}: {r['error']}")
                continue
            scores.append(r["score"])
        print(f"{tag}: {len(scores)} 张, 耗时 {time.time()-t0:.1f}s")
        return np.asarray(scores)

    s_hold = score_paths(holdout_paths, "留出正常")
    s_def = score_paths(defects, "缺陷")

    # 指标
    fpr_np = float(np.mean(s_hold > tau_np))
    fpr_legacy = float(np.mean(s_hold > tau_legacy))
    rec_np = float(np.mean(s_def > tau_np))
    rec_legacy = float(np.mean(s_def > tau_legacy))
    auc = auroc(s_def, s_hold)

    print("\n══════ 结果 ══════")
    print(f"Image AUROC: {auc:.4f}")
    print(f"{'阈值方案':<16}{'阈值':>10}{'FPR(留出正常)':>16}{'Recall(缺陷)':>14}")
    print(f"{'NP校准(eps=2%)':<16}{tau_np:>10.4f}{fpr_np:>16.2%}{rec_np:>14.2%}")
    print(f"{'legacy P99x1.2':<16}{tau_legacy:>10.4f}{fpr_legacy:>16.2%}{rec_legacy:>14.2%}")
    print(f"\n判定: FPR约束{'达标' if fpr_np <= calib.epsilon + 0.02 else '超限'} "
          f"(目标≤{calib.epsilon:.0%}, 容差+2%小样本波动)")
    print(f"判定: Recall{'不回退' if rec_np >= rec_legacy else '回退!'} "
          f"(NP {rec_np:.1%} vs legacy {rec_legacy:.1%})")

    # 校准置信度抽样展示
    print("\n校准置信度抽样 (anomaly_confidence):")
    for s in [float(np.min(s_hold)), float(np.median(s_hold)),
              float(np.max(s_hold)), float(np.median(s_def)),
              float(np.max(s_def))]:
        print(f"  score={s:.4f} → conf={calib.anomaly_confidence(s):.3f}, "
              f"p值={calib.survival(s):.4f}, decide={calib.decide(s)}")


if __name__ == "__main__":
    main()
