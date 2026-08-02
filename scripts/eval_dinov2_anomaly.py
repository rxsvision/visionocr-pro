"""DINOv2 异常检测 vs PatchCore — KolektorSDD 对比验证

目标 (可验证):
1. 同划分 (seed 2026, 正常 80/20) 下对比两引擎 AUROC / FPR / Recall
2. 验证双源 Union OR 互补性: 合并后 Recall 提升幅度 vs FPR 代价
3. NP 校准在两引擎上均满足 FPR 统计约束

与 scripts/eval_np_calibration.py 使用完全相同的扫描与划分逻辑。
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
    sys.exit("用法: python scripts/eval_dinov2_anomaly.py <KolektorSDD数据集目录>")
DATA_ROOT = Path(sys.argv[1])


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
    pos, neg = np.asarray(scores_pos), np.asarray(scores_neg)
    all_s = np.concatenate([pos, neg])
    order = np.argsort(all_s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = all_s[order]
    i, n = 0, len(all_s)
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def score_engine(eng, paths, tag):
    scores, t0 = [], time.time()
    for p in paths:
        r = eng.infer(p)
        if r.get("error"):
            print(f"  [WARN] {tag}: {r['error']}")
            continue
        scores.append(r["score"])
    print(f"  {tag}: {len(scores)} 张, {time.time()-t0:.1f}s")
    return np.asarray(scores)


def main():
    print(f"数据集: {DATA_ROOT}")
    normals, defects = scan_dataset(DATA_ROOT)
    print(f"扫描: 正常 {len(normals)}, 缺陷 {len(defects)}")

    rng = np.random.default_rng(2026)  # 与 PatchCore 评估完全一致的划分
    idx = rng.permutation(len(normals))
    n_train = int(len(normals) * 0.8)
    train_paths = [normals[i] for i in idx[:n_train]]
    holdout_paths = [normals[i] for i in idx[n_train:]]
    print(f"划分: train={len(train_paths)}, holdout={len(holdout_paths)}, "
          f"defect={len(defects)}")

    cfg = {"device": "auto", "qc": {
        "patchcore": {"input_size": 512, "coreset_ratio": 0.1,
                      "conservative_mode": True, "np_epsilon": 0.10},
        "dinov2": {"input_size": 518, "pca_dim": 64,
                   "n_etalons": 8, "np_epsilon": 0.10},
    }}

    # ── PatchCore ──
    from engines.vision.anomalib_engine import AnomalibEngine
    print("\n[1/2] PatchCore (WRN50-2)...")
    pc = AnomalibEngine(cfg)
    pc.load()
    assert pc.is_ready()
    t0 = time.time()
    pc.train(train_paths)
    print(f"  建库 {time.time()-t0:.1f}s, tau_np={pc._calibrated_threshold:.4f}")
    pc_hold = score_engine(pc, holdout_paths, "留出正常")
    pc_def = score_engine(pc, defects, "缺陷")
    tau_pc = pc._np_calibrator.threshold

    # ── DINOv2 ──
    from engines.vision.dinov2_anomaly import DINOv2AnomalyEngine
    print("\n[2/2] DINOv2 (ViT-S/14 + GMM)...")
    dv = DINOv2AnomalyEngine(cfg)
    dv.load()
    assert dv.is_ready()
    t0 = time.time()
    dv.train(train_paths)
    print(f"  建库 {time.time()-t0:.1f}s, tau_np={dv._calibrated_threshold:.4f}")
    dv_hold = score_engine(dv, holdout_paths, "留出正常")
    dv_def = score_engine(dv, defects, "缺陷")
    tau_dv = dv._np_calibrator.threshold

    # ── 指标 ──
    def metrics(s_hold, s_def, tau):
        return (float(np.mean(s_hold > tau)), float(np.mean(s_def > tau)))

    fpr_pc, rec_pc = metrics(pc_hold, pc_def, tau_pc)
    fpr_dv, rec_dv = metrics(dv_hold, dv_def, tau_dv)

    # 双源 Union OR (图像级)
    u_hold = np.logical_or(pc_hold > tau_pc, dv_hold > tau_dv)
    u_def = np.logical_or(pc_def > tau_pc, dv_def > tau_dv)
    fpr_u, rec_u = float(u_hold.mean()), float(u_def.mean())

    auc_pc = auroc(pc_def, pc_hold)
    auc_dv = auroc(dv_def, dv_hold)

    # 互补性: 各自独占捕获的缺陷数
    only_pc = int(np.sum((pc_def > tau_pc) & ~(dv_def > tau_dv)))
    only_dv = int(np.sum((dv_def > tau_dv) & ~(pc_def > tau_pc)))
    both = int(np.sum((pc_def > tau_pc) & (dv_def > tau_dv)))
    neither = int(np.sum(~((pc_def > tau_pc) | (dv_def > tau_dv))))

    print("\n══════ 结果 (NP校准 eps=10%, 同划分 seed=2026) ══════")
    print(f"{'引擎':<22}{'AUROC':>8}{'阈值tau':>12}{'FPR':>10}{'Recall':>10}")
    print(f"{'PatchCore(WRN50)':<22}{auc_pc:>8.4f}{tau_pc:>12.4f}"
          f"{fpr_pc:>10.2%}{rec_pc:>10.2%}")
    print(f"{'DINOv2(ViT-S+GMM)':<22}{auc_dv:>8.4f}{tau_dv:>12.4f}"
          f"{fpr_dv:>10.2%}{rec_dv:>10.2%}")
    print(f"{'双源Union OR':<22}{'—':>8}{'—':>12}"
          f"{fpr_u:>10.2%}{rec_u:>10.2%}")
    print(f"\n缺陷捕获分解 (共{len(dv_def)}): 双源均捕获={both}, "
          f"仅PatchCore={only_pc}, 仅DINOv2={only_dv}, 均漏={neither}")
    print(f"\n判定1: DINOv2 FPR {'达标' if fpr_dv <= 0.12 else '超限'} "
          f"(目标≤10%+2%容差)")
    print(f"判定2: Union Recall {rec_u:.1%} vs 单源最优 "
          f"{max(rec_pc, rec_dv):.1%} → "
          f"{'互补有效' if rec_u > max(rec_pc, rec_dv) else '无增益'}")

    pc.unload()
    dv.unload()


if __name__ == "__main__":
    main()
