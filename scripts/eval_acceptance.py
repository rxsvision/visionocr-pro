"""多数据集验收评估 — PatchCore-NP / DINOv2 / 双源Union OR

用法 (路径全部经 argv 传入, 仓库内不硬编码任何数据路径):

  python scripts/eval_acceptance.py kolektor <root> [--out X.json]
      mask 标注表面缺陷 (Part*.jpg + Part*_label.bmp), 80/20 划分 (seed 2026)

  python scripts/eval_acceptance.py subspacead <root> [--out X.json]
      SubspaceAD 快速换线验收: 1/2/4-shot 建库 vs PatchCore 全库基线
      (验收标准: 1-shot Recall@eps=0.10 ≥ 全库的 85%)

  python scripts/eval_acceptance.py pcb <root> [--out X.json]
      root/images/<类别>/*.jpg 为缺陷, root/PCB_USED/* 为 OK 建库样本

  python scripts/eval_acceptance.py paired <ok_dir> <def_dir> [--name N] [--out X.json]
      成对打光: ok_dir=对照(正常)图, def_dir=缺陷图 (按众数尺寸过滤杂项)

  python scripts/eval_acceptance.py bootstrap <dir> [--name N] [--bank-frac 0.75] [--out X.json]
      无标注目录: 以 DINOv2 特征质心自举选"近正常"子集建库,
      全量打分输出离群排名 (供人工目检核验), 附健壮性/耗时统计

指标口径:
- AUROC: 图像级 rank-based (含并列处理)
- FPR/Recall: 各引擎 NP 阈值 (或 legacy fallback) 下的图像级判定
- Union OR: 任一引擎判 NG 即 NG (零漏检架构)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CFG = {"device": "auto", "qc": {
    "patchcore": {"input_size": 512, "coreset_ratio": 0.1,
                  "conservative_mode": True, "np_epsilon": 0.10},
    "dinov2": {"input_size": 518, "pca_dim": 64,
               "n_etalons": 8, "np_epsilon": 0.10},
    "subspacead": {"input_size": 448, "layers": [-4, -5],
                   "pca_ev": 0.99, "aug_count": 30,
                   "fast_max_images": 4, "np_epsilon": 0.10},
}}


# ─── 通用工具 ───────────────────────────────────────────────
def imdecode_gray(path: str) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8),
                        cv2.IMREAD_GRAYSCALE)


def list_images(d: Path, recursive: bool = False) -> list[str]:
    it = d.rglob("*") if recursive else d.iterdir()
    return sorted(str(p) for p in it
                  if p.is_file() and p.suffix.lower() in IMG_EXT)


def filter_majority_shape(paths: list[str]) -> tuple[list[str], list[str]]:
    """按众数 (h,w) 过滤杂项图 (说明图/实物图等混入文件)。"""
    shapes: dict[tuple[int, int], list[str]] = {}
    bad: list[str] = []
    for p in paths:
        img = imdecode_gray(p)
        if img is None:
            bad.append(p)
            continue
        shapes.setdefault(img.shape[:2], []).append(p)
    if not shapes:
        return [], bad
    best = max(shapes.values(), key=len)
    excluded = bad + [p for k, v in shapes.items() for p in v
                      if len(v) < len(best)]
    return best, excluded


def auroc(scores_pos, scores_neg) -> float:
    pos, neg = np.asarray(scores_pos), np.asarray(scores_neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
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


def make_engines():
    from engines.vision.anomalib_engine import AnomalibEngine
    from engines.vision.dinov2_anomaly import DINOv2AnomalyEngine
    pc, dv = AnomalibEngine(CFG), DINOv2AnomalyEngine(CFG)
    pc.load()
    dv.load()
    assert pc.is_ready() and dv.is_ready()
    return pc, dv


def threshold_mode(eng) -> str:
    return "NP" if getattr(eng, "_np_calibrator", None) is not None \
        else "legacy-P99x1.2"


def score_paths(eng, paths: list[str], tag: str) -> tuple[np.ndarray, list]:
    scores, errors = [], []
    for p in paths:
        r = eng.infer(p)
        if r.get("error"):
            errors.append((p, r["error"]))
            continue
        scores.append(r["score"])
    print(f"  {tag}: {len(scores)} 张打分"
          + (f", {len(errors)} 张失败" if errors else ""))
    return np.asarray(scores, dtype=np.float64), errors


def eval_block(name: str, pc, dv, ok_paths: list[str],
               def_paths: list[str], root: Path | None = None) -> dict:
    """双引擎建库 + 打分 + 指标。ok_paths 需已 shuffle (定序)。"""
    print(f"\n── [{name}] 建库: OK {len(ok_paths)} 张 ──")
    t0 = time.time()
    pc.train(ok_paths)
    t_pc_train = time.time() - t0
    t0 = time.time()
    dv.train(ok_paths)
    t_dv_train = time.time() - t0
    tau_pc = pc._calibrated_threshold
    tau_dv = dv._calibrated_threshold
    print(f"  PatchCore: {t_pc_train:.1f}s, tau={tau_pc:.4f} "
          f"({threshold_mode(pc)})")
    print(f"  DINOv2:    {t_dv_train:.1f}s, tau={tau_dv:.4f} "
          f"({threshold_mode(dv)})")

    print(f"── [{name}] 打分: 正常 {len(ok_paths)}, 缺陷 {len(def_paths)} ──")
    t0 = time.time()
    pc_ok, e1 = score_paths(pc, ok_paths, "PC正常")
    pc_def, e2 = score_paths(pc, def_paths, "PC缺陷")
    t_pc_infer = (time.time() - t0) / max(1, len(ok_paths) + len(def_paths))
    t0 = time.time()
    dv_ok, e3 = score_paths(dv, ok_paths, "DV正常")
    dv_def, e4 = score_paths(dv, def_paths, "DV缺陷")
    t_dv_infer = (time.time() - t0) / max(1, len(ok_paths) + len(def_paths))

    def m(s_ok, s_def, tau):
        return (float(np.mean(s_ok > tau)) if len(s_ok) else float("nan"),
                float(np.mean(s_def > tau)) if len(s_def) else float("nan"))

    fpr_pc, rec_pc = m(pc_ok, pc_def, tau_pc)
    fpr_dv, rec_dv = m(dv_ok, dv_def, tau_dv)
    # Union OR (对齐长度: 仅统计双引擎均成功的样本)
    n_ok = min(len(pc_ok), len(dv_ok))
    n_def = min(len(pc_def), len(dv_def))
    u_ok = np.logical_or(pc_ok[:n_ok] > tau_pc, dv_ok[:n_ok] > tau_dv)
    u_def = np.logical_or(pc_def[:n_def] > tau_pc, dv_def[:n_def] > tau_dv)
    fpr_u, rec_u = float(u_ok.mean()), float(u_def.mean())
    only_pc = int(np.sum((pc_def[:n_def] > tau_pc)
                         & ~(dv_def[:n_def] > tau_dv)))
    only_dv = int(np.sum((dv_def[:n_def] > tau_dv)
                         & ~(pc_def[:n_def] > tau_pc)))
    both = int(np.sum((pc_def[:n_def] > tau_pc)
                      & (dv_def[:n_def] > tau_dv)))
    missed = n_def - both - only_pc - only_dv

    block = {
        "n_ok": len(ok_paths), "n_def": len(def_paths),
        "pc": {"auroc": auroc(pc_def, pc_ok), "tau": float(tau_pc),
               "fpr": fpr_pc, "recall": rec_pc,
               "mode": threshold_mode(pc),
               "train_s": round(t_pc_train, 1),
               "infer_ms": round(t_pc_infer * 1000, 1)},
        "dv": {"auroc": auroc(dv_def, dv_ok), "tau": float(tau_dv),
               "fpr": fpr_dv, "recall": rec_dv,
               "mode": threshold_mode(dv),
               "train_s": round(t_dv_train, 1),
               "infer_ms": round(t_dv_infer * 1000, 1)},
        "union": {"fpr": fpr_u, "recall": rec_u,
                  "both": both, "only_pc": only_pc,
                  "only_dv": only_dv, "missed": missed},
        "errors": len(e1) + len(e2) + len(e3) + len(e4),
    }

    # ── 多 eps 运行点 (零漏检政策选型依据; 用引擎 cal 分数重算分位数) ──
    import math
    ops = {}
    for eps in (0.02, 0.10, 0.15):
        row = {}
        taus = {}
        for key, eng, s_ok, s_def in (
                ("pc", pc, pc_ok, pc_def), ("dv", dv, dv_ok, dv_def)):
            cal = np.sort(np.asarray(eng._train_scores, dtype=np.float64))
            ncal = len(cal)
            rank = min(max(int(math.ceil((1 - eps) * (ncal + 1))), 1), ncal)
            tau_e = float(cal[rank - 1])
            taus[key] = tau_e
            row[key] = {
                "tau": round(tau_e, 4),
                "fpr": round(float(np.mean(s_ok > tau_e)), 4)
                if len(s_ok) else None,
                "recall": round(float(np.mean(s_def > tau_e)), 4)
                if len(s_def) else None,
            }
        u_ok_e = np.logical_or(pc_ok[:n_ok] > taus["pc"],
                               dv_ok[:n_ok] > taus["dv"])
        u_def_e = np.logical_or(pc_def[:n_def] > taus["pc"],
                                dv_def[:n_def] > taus["dv"])
        row["union"] = {"fpr": round(float(u_ok_e.mean()), 4),
                        "recall": round(float(u_def_e.mean()), 4)}
        ops[f"{eps:.2f}"] = row
    block["ops"] = ops
    print(f"  {'引擎':<14}{'AUROC':>8}{'tau':>12}{'FPR':>9}{'Recall':>9}")
    for k, label in (("pc", "PatchCore"), ("dv", "DINOv2")):
        b = block[k]
        print(f"  {label:<14}{b['auroc']:>8.4f}{b['tau']:>12.4f}"
              f"{b['fpr']:>9.2%}{b['recall']:>9.2%}")
    print(f"  {'Union OR':<14}{'—':>8}{'—':>12}"
          f"{fpr_u:>9.2%}{rec_u:>9.2%}")
    print(f"  捕获分解: 双捕获={both} 仅PC={only_pc} "
          f"仅DV={only_dv} 均漏={missed}")
    return block


# ─── 各模式 ─────────────────────────────────────────────────
def scan_kolektor(root: Path):
    normals, defects = [], []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for img_p in sorted(d.glob("Part*.jpg")):
            lbl_p = img_p.with_name(img_p.stem + "_label.bmp")
            if not lbl_p.exists():
                continue
            lbl = imdecode_gray(str(lbl_p))
            if lbl is None:
                continue
            (defects if lbl.max() > 0 else normals).append(str(img_p))
    return normals, defects


def _sweep_epsilon(eng, cal_scores, hold_scores, def_scores):
    """split-conformal 分位数随 eps 的 FPR/Recall 曲线 (同一 bank)。"""
    import math
    cal = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = len(cal)
    rows = {}
    for eps in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        rank = min(max(int(math.ceil((1 - eps) * (n + 1))), 1), n)
        tau = float(cal[rank - 1])
        rows[f"{eps:.2f}"] = {
            "tau": round(tau, 4),
            "fpr_hold": round(float(np.mean(hold_scores > tau)), 4),
            "recall": round(float(np.mean(def_scores > tau)), 4),
        }
    return rows


def _recall_at_matched_fpr(hold_scores, def_scores, eps):
    """匹配误报率口径的 Recall (防退化, 跨方法公平)。

    用真实未见 holdout OK 分数的 (1-eps) 次序统计量作阈值, 再看缺陷
    Recall。阈值锚定在两个方法共同的诚实参照 (holdout 正常图) 上,
    与各自自校准质量无关——避免某方法自校准偏松 (阈值塌陷→全判NG)
    而虚高 Recall。缺陷图从不参与选阈值, 故 Recall 诚实。

    Returns: (tau, fpr_hold, recall)
    """
    import math
    hold = np.sort(np.asarray(hold_scores, dtype=np.float64))
    n = len(hold)
    if n == 0 or len(def_scores) == 0:
        return float("nan"), float("nan"), float("nan")
    rank = min(max(int(math.ceil((1 - eps) * (n + 1))), 1), n)
    tau = float(hold[rank - 1])
    fpr = float(np.mean(np.asarray(hold_scores) > tau))
    recall = float(np.mean(np.asarray(def_scores) > tau))
    return tau, fpr, recall


def mode_kolektor(args) -> dict:
    root = Path(args.root)
    normals, defects = scan_kolektor(root)
    rng = np.random.default_rng(2026)  # 与既有评估完全一致的划分
    idx = rng.permutation(len(normals))
    n_train = int(len(normals) * 0.8)
    train = [normals[i] for i in idx[:n_train]]
    holdout = [normals[i] for i in idx[n_train:]]
    print(f"[kolektor] 正常 {len(normals)} (train {len(train)} / "
          f"holdout {len(holdout)}), 缺陷 {len(defects)}")
    pc, dv = make_engines()
    # 注: holdout 正常图参与 FPR 统计 (未见过), 缺陷图全部参与 Recall
    blk = eval_block("kolektor", pc, dv, train, defects)

    pc_hold, _ = score_paths(pc, holdout, "PC留出")
    dv_hold, _ = score_paths(dv, holdout, "DV留出")
    pc_def, _ = score_paths(pc, defects, "PC缺陷(复算)")
    dv_def, _ = score_paths(dv, defects, "DV缺陷(复算)")
    blk["holdout_fpr"] = {
        "pc": float(np.mean(pc_hold > blk["pc"]["tau"])),
        "dv": float(np.mean(dv_hold > blk["dv"]["tau"])),
    }
    # 诚实口径: 以未见 holdout 正常图为负样本的 AUROC
    blk["pc"]["auroc_holdout"] = auroc(pc_def, pc_hold)
    blk["dv"]["auroc_holdout"] = auroc(dv_def, dv_hold)
    print(f"  留出正常 FPR: PC={blk['holdout_fpr']['pc']:.2%}, "
          f"DV={blk['holdout_fpr']['dv']:.2%} (真·未见样本)")
    print(f"  留出 AUROC: PC={blk['pc']['auroc_holdout']:.4f}, "
          f"DV={blk['dv']['auroc_holdout']:.4f}")

    # eps 扫描: 零漏检政策下的运行点选择依据
    blk["eps_sweep"] = {
        "pc": _sweep_epsilon(pc, pc._train_scores, pc_hold, pc_def),
        "dv": _sweep_epsilon(dv, dv._train_scores, dv_hold, dv_def),
    }
    print("  eps 扫描 (holdout FPR / Recall):")
    for eng_key in ("pc", "dv"):
        line = "    " + eng_key.upper() + ": "
        for eps, r in blk["eps_sweep"][eng_key].items():
            line += f"[{eps}] {r['fpr_hold']:.1%}/{r['recall']:.0%} "
        print(line)
    pc.unload()
    dv.unload()
    return {"dataset": "kolektor", **blk}


def mode_pcb(args) -> dict:
    root = Path(args.root)
    normals = list_images(root / "PCB_USED")
    defects_by_cls: dict[str, list[str]] = {}
    for cls in sorted((root / "images").iterdir()):
        if cls.is_dir():
            defects_by_cls[cls.name] = list_images(cls)
    defects = [p for v in defects_by_cls.values() for p in v]
    print(f"[pcb] OK {len(normals)}, 缺陷 {len(defects)} "
          f"({len(defects_by_cls)} 类)")
    rng = np.random.default_rng(2026)
    ok_shuffled = [normals[i] for i in rng.permutation(len(normals))]
    pc, dv = make_engines()
    blk = eval_block("pcb", pc, dv, ok_shuffled, defects)
    # 分类别 Recall (缺陷类型难度画像)
    per_cls = {}
    for cls, paths in defects_by_cls.items():
        n = len(paths)
        s_pc = np.asarray([pc.infer(p)["score"] for p in paths],
                          dtype=np.float64)
        s_dv = np.asarray([dv.infer(p)["score"] for p in paths],
                          dtype=np.float64)
        r_pc = float(np.mean(s_pc > blk["pc"]["tau"]))
        r_dv = float(np.mean(s_dv > blk["dv"]["tau"]))
        per_cls[cls] = {"n": n, "recall_pc": r_pc, "recall_dv": r_dv}
        print(f"  类别 {cls:<18} n={n:>3}  "
              f"PC Recall={r_pc:.1%}  DV Recall={r_dv:.1%}")
    blk["per_class"] = per_cls
    blk["note"] = "FPR 以建库 OK 图测量 (无独立正常测试集, 偏乐观)"
    pc.unload()
    dv.unload()
    return {"dataset": "pcb", **blk}


def mode_yolo(args) -> dict:
    """YOLO 结构缺陷源验收: root=PCB_DATASET根(取PCB_USED正常),
    def_dir=留出 val 图目录 (训练时未见, 诚实口径)。"""
    from engines.vision.yolo_defect import YOLODefectEngine
    root, val_dir = Path(args.root), Path(args.def_dir)
    defects = list_images(val_dir)
    normals = list_images(root / "PCB_USED")
    print(f"[yolo] val缺陷 {len(defects)}, 正常 {len(normals)}")

    eng = YOLODefectEngine(CFG)
    eng.load()
    if not eng.is_ready():
        return {"dataset": "yolo", "error": "权重加载失败"}

    t0 = time.time()

    def flag(paths):
        out = []
        for p in paths:
            r = eng.infer(p)
            out.append((r.get("count", 0) > 0, r.get("count", 0),
                        float(r.get("max_score", 0) or 0)))
        return out

    d_flags = flag(defects)
    n_flags = flag(normals)
    dt = (time.time() - t0) / max(1, len(defects) + len(normals))

    recall = float(np.mean([f for f, _, _ in d_flags]))
    fpr = float(np.mean([f for f, _, _ in n_flags]))
    # 分类别 Recall: 文件名 NN_<class>_MM.jpg
    per_cls: dict[str, list[bool]] = {}
    for p, (fl, _, _) in zip(defects, d_flags):
        parts = Path(p).stem.split("_")
        cls = "_".join(parts[1:-1]) if len(parts) >= 3 else "?"
        per_cls.setdefault(cls, []).append(fl)
    per_cls_stats = {k: {"n": len(v), "recall": round(float(np.mean(v)), 4)}
                     for k, v in sorted(per_cls.items())}
    for k, v in per_cls_stats.items():
        print(f"  类别 {k:<18} n={v['n']:>3}  Recall={v['recall']:.1%}")

    result = {"dataset": "yolo",
              "n_def": len(defects), "n_ok": len(normals),
              "recall": recall, "fpr": fpr,
              "infer_ms": round(dt * 1000, 1),
              "per_class": per_cls_stats,
              "note": "val 划分为训练时留出, 指标为诚实口径"}
    print(f"  YOLO: Recall={recall:.2%}, FPR={fpr:.2%}, "
          f"{dt * 1000:.1f} ms/张")
    return result


def mode_paired(args) -> dict:
    ok_dir, def_dir = Path(args.root), Path(args.def_dir)
    ok_all, ex1 = filter_majority_shape(list_images(ok_dir))
    def_all, ex2 = filter_majority_shape(list_images(def_dir))
    print(f"[paired:{args.name}] OK {len(ok_all)} (排除 {len(ex1)}), "
          f"缺陷 {len(def_all)} (排除 {len(ex2)})")
    rng = np.random.default_rng(2026)
    ok_shuffled = [ok_all[i] for i in rng.permutation(len(ok_all))]
    pc, dv = make_engines()
    blk = eval_block(f"paired:{args.name}", pc, dv, ok_shuffled, def_all)
    blk["name"] = args.name
    blk["excluded"] = [Path(p).name for p in ex1 + ex2]
    pc.unload()
    dv.unload()
    return {"dataset": "paired", **blk}


def mode_bootstrap(args) -> dict:
    """无标注自举: 特征质心 → 近心子集建库 → 全量排名。"""
    d = Path(args.dir)
    paths, excluded = filter_majority_shape(list_images(d))
    print(f"[bootstrap:{args.name}] 图像 {len(paths)} "
          f"(排除杂项 {len(excluded)}), bank_frac={args.bank_frac}")
    out = {"dataset": "bootstrap", "name": args.name,
           "n_images": len(paths), "excluded": [Path(p).name
                                                for p in excluded]}
    if len(paths) < 3:
        out["verdict"] = "样本不足(<3), 仅健壮性核查"
        print("  样本不足, 跳过建库")
        return out

    pc, dv = make_engines()
    # DINOv2 图像级特征 (patch 均值) → 质心距离 → 选近心子集建库
    vecs, valid = [], []
    for p in paths:
        feat = dv._extract_features(p)
        if feat is None:
            continue
        vecs.append(feat.mean(axis=0))
        valid.append(p)
    V = np.stack(vecs)
    centroid = V.mean(axis=0)
    dist = np.linalg.norm(V - centroid, axis=1)
    order = np.argsort(dist, kind="mergesort")
    n_bank = max(3, int(len(valid) * args.bank_frac))
    bank_paths = [valid[i] for i in order[:n_bank]]
    out["reliable"] = len(valid) >= 10
    print(f"  有效 {len(valid)}, 建库子集 {n_bank} (距质心最近)")

    blk = eval_block(f"bootstrap:{args.name}", pc, dv,
                     bank_paths,
                     [valid[i] for i in order[n_bank:]])
    out.update(blk)

    # 全量离群排名 (供人工目检): 双引擎分数合并
    rows = []
    for p in valid:
        r_pc = pc.infer(p)
        r_dv = dv.infer(p)
        rows.append({
            "file": Path(p).name,
            "pc_score": round(float(r_pc.get("score", 0)), 4),
            "pc_ng": bool(r_pc.get("pred_label") == "NG"),
            "dv_score": round(float(r_dv.get("score", 0)), 4),
            "dv_ng": bool(r_dv.get("pred_label") == "NG"),
        })
    # 各自 min-max 归一后相加排序 (量纲不同)
    def norm(vals):
        arr = np.asarray(vals, dtype=np.float64)
        rng_ = arr.max() - arr.min()
        return (arr - arr.min()) / rng_ if rng_ > 1e-12 else arr * 0
    mix = norm([r["pc_score"] for r in rows]) + \
        norm([r["dv_score"] for r in rows])
    for r, m in zip(rows, mix):
        r["mix"] = round(float(m), 4)
    rows.sort(key=lambda r: -r["mix"])
    out["top_outliers"] = rows[:12]
    out["n_flagged_union"] = sum(1 for r in rows if r["pc_ng"] or r["dv_ng"])
    print(f"  Union 标记 NG: {out['n_flagged_union']}/{len(rows)}")
    print("  Top 离群 (mix=归一分数和):")
    for r in rows[:8]:
        print(f"    {r['mix']:.3f}  PC={'NG' if r['pc_ng'] else 'ok'} "
              f"DV={'NG' if r['dv_ng'] else 'ok'}  {r['file']}")
    out["all_rows"] = rows
    pc.unload()
    dv.unload()
    return out


def mode_subspacead(args) -> dict:
    """SubspaceAD 快速换线验收: 1/2/4-shot vs PatchCore 全库基线。

    验收标准 (方案 §5.2): 1-shot 模式 Recall@eps=0.10 ≥ PatchCore
    全库模式同口径的 85%。同次运行计算基线, 保证口径一致。
    """
    from engines.vision.subspace_ad import SubspaceADEngine
    from engines.vision.anomalib_engine import AnomalibEngine

    root = Path(args.root)
    normals, defects = scan_kolektor(root)
    rng = np.random.default_rng(2026)  # 与 mode_kolektor 完全相同的划分
    idx = rng.permutation(len(normals))
    n_train = int(len(normals) * 0.8)
    train = [normals[i] for i in idx[:n_train]]
    holdout = [normals[i] for i in idx[n_train:]]
    print(f"[subspacead] 正常 {len(normals)} (train {len(train)} / "
          f"holdout {len(holdout)}), 缺陷 {len(defects)}")

    out = {"dataset": "kolektor-subspacead",
           "n_ok_train": len(train), "n_ok_hold": len(holdout),
           "n_def": len(defects), "shots": {}}

    for k in (1, 2, 4):
        support = train[:k]  # train 已按 seed=2026 定序洗牌, 取前 k
        print(f"\n── [{k}-shot] 建库: "
              f"{[Path(p).name for p in support]} ──")
        sa = SubspaceADEngine(CFG)
        sa.load()
        assert sa.is_ready(), "SubspaceAD 加载失败"
        t0 = time.time()
        meta = sa.train(support)
        t_train = time.time() - t0
        assert not meta.get("error"), meta
        print(f"  建库 {t_train:.1f}s, pca_k={meta['pca_k']}, "
              f"增广入池={meta['n_augmented']}, "
              f"tau={sa._calibrated_threshold:.4f}")
        s_hold, _ = score_paths(sa, holdout, f"SA{k}留出")
        s_def, _ = score_paths(sa, defects, f"SA{k}缺陷")
        tau = sa._calibrated_threshold
        # 匹配FPR口径 (验收依据): 阈值锚定 holdout 正常图, 防自校准退化
        mf = {f"{e:.2f}": _recall_at_matched_fpr(s_hold, s_def, e)
              for e in (0.05, 0.10, 0.20)}
        blk = {
            "train_s": round(t_train, 1),
            "pca_k": meta["pca_k"],
            "auroc": auroc(s_def, s_hold),
            "matched_fpr": {e: {"tau": round(t, 4),
                                "fpr_hold": round(f, 4),
                                "recall": round(r, 4)}
                            for e, (t, f, r) in mf.items()},
            # 自校准运行点 (仅透明展示; 快速模式增广自评偏乐观,
            # 不作为验收依据, 见 matched_fpr)
            "selfcal": {
                "tau": float(tau),
                "fpr_hold": float(np.mean(s_hold > tau)),
                "recall": float(np.mean(s_def > tau)),
            },
        }
        r10 = blk["matched_fpr"]["0.10"]["recall"]
        print(f"  AUROC={blk['auroc']:.4f}  "
              f"匹配FPR@0.10: Recall={r10:.2%} "
              f"(tau={blk['matched_fpr']['0.10']['tau']:.4f})")
        sc = blk["selfcal"]
        print(f"  自校准(部署口径, 偏乐观): tau={sc['tau']:.4f} "
              f"FPR-hold={sc['fpr_hold']:.2%} Recall={sc['recall']:.2%}")
        out["shots"][str(k)] = blk
        sa.unload()

    # 基线: PatchCore 全库 (同划分; 验收同样用匹配FPR口径)
    print("\n── 基线: PatchCore 全库 ──")
    pc = AnomalibEngine(CFG)
    pc.load()
    assert pc.is_ready(), "PatchCore 加载失败"
    t0 = time.time()
    pc.train(train)
    t_pc = time.time() - t0
    pc_hold, _ = score_paths(pc, holdout, "PC留出")
    pc_def, _ = score_paths(pc, defects, "PC缺陷")
    pc_mf = {f"{e:.2f}": _recall_at_matched_fpr(pc_hold, pc_def, e)
             for e in (0.05, 0.10, 0.20)}
    out["patchcore_full"] = {
        "train_s": round(t_pc, 1),
        "auroc": auroc(pc_def, pc_hold),
        "matched_fpr": {e: {"tau": round(t, 4),
                            "fpr_hold": round(f, 4),
                            "recall": round(r, 4)}
                        for e, (t, f, r) in pc_mf.items()},
        "selfcal": {
            "tau": float(pc._calibrated_threshold),
            "fpr_hold": float(np.mean(
                pc_hold > pc._calibrated_threshold)),
            "recall": float(np.mean(
                pc_def > pc._calibrated_threshold)),
        },
    }
    print(f"  建库 {t_pc:.1f}s, AUROC={out['patchcore_full']['auroc']:.4f}"
          f"  匹配FPR@0.10: "
          f"Recall={out['patchcore_full']['matched_fpr']['0.10']['recall']:.2%}")

    # 验收判定: 匹配FPR@0.10 口径 (阈值锚定共同 holdout, 防退化)
    sa_r10 = out["shots"]["1"]["matched_fpr"]["0.10"]["recall"]
    pc_r10 = out["patchcore_full"]["matched_fpr"]["0.10"]["recall"]
    ratio = sa_r10 / pc_r10 if pc_r10 > 0 else float("nan")
    auroc_ratio = (out["shots"]["1"]["auroc"]
                   / out["patchcore_full"]["auroc"])
    out["acceptance"] = {
        "criterion": ("1-shot Recall@匹配FPR=0.10 >= 0.85 x PatchCore 全库 "
                      "(阈值锚定共同holdout正常图, 防自校准退化)"),
        "sa_recall_1shot": sa_r10,
        "pc_recall_full": pc_r10,
        "ratio": round(ratio, 4),
        "auroc_1shot": out["shots"]["1"]["auroc"],
        "auroc_full": out["patchcore_full"]["auroc"],
        "auroc_ratio": round(auroc_ratio, 4),
        "pass": bool(ratio >= 0.85),
    }
    print(f"\n验收: 1-shot Recall={sa_r10:.1%} vs 全库 {pc_r10:.1%} "
          f"(比值 {ratio:.1%}, 门槛 85%) | AUROC {out['shots']['1']['auroc']:.4f}"
          f" vs {out['patchcore_full']['auroc']:.4f} → "
          f"{'PASS' if out['acceptance']['pass'] else 'FAIL'}")
    pc.unload()
    return out


# ─── main ──────────────────────────────────────────────────
def _gdino_flags(reg, cfg, paths: list[str], tag: str):
    """GDINO 逐图 NG 标志 (runtime 配置: DEFAULT_PROMPT + conf + size filter)。"""
    from core.defect_detector import run_detection, DEFAULT_PROMPT
    thr = cfg["qc"].get("confidence_threshold", 0.3)
    size_cfg = cfg["qc"].get("defect_size")
    flags, errs = [], 0
    for i, p in enumerate(paths):
        r = run_detection(reg, p, prompt=DEFAULT_PROMPT,
                          threshold=thr, size_cfg=size_cfg)
        if r.get("verdict") == "ERROR":
            errs += 1
            flags.append(False)
        else:
            flags.append(r.get("verdict") == "NG")
        if (i + 1) % 25 == 0:
            print(f"    [{tag}] {i + 1}/{len(paths)}", flush=True)
    print(f"    [{tag}] 完成 {len(paths)} 张, NG={sum(flags)}, 错误 {errs}")
    return np.asarray(flags, dtype=bool), errs


def mode_fusion55(args) -> dict:
    """§5.5 分阶段融合验收: KolektorSDD Union OR 基线 vs 分阶段融合。

    验收标准 (方案 §5.5): Union Recall 不降前提下, FPR-hold 较 v1.3.0
    (Union OR 任一NG即NG) 下降 ≥30%。
    口径 (零漏检): "Recall 不降"按 有效召回 = 被判 NG 或 REVIEW 的缺陷占比
    衡量 — 分阶段融合只把"单源孤证 NG"降级为 REVIEW 黄牌(人工复核),
    从不把 Union OR 捕获的缺陷静默放行 OK, 故有效召回恒等于基线召回。
    FPR-hold 按 自主 NG (不含 REVIEW) 在留出正常图上的比例衡量 —
    这是产线真实拦截误报, REVIEW 转人工不计数。
    """
    from core.fusion import staged_fusion, calibrated_n_samples

    root = Path(args.root)
    normals, defects = scan_kolektor(root)
    rng = np.random.default_rng(2026)  # 与 mode_kolektor 完全一致的划分
    idx = rng.permutation(len(normals))
    n_train = int(len(normals) * 0.8)
    train = [normals[i] for i in idx[:n_train]]
    holdout = [normals[i] for i in idx[n_train:]]
    print(f"[fusion55] 正常 {len(normals)} (train {len(train)} / "
          f"holdout {len(holdout)}), 缺陷 {len(defects)}")

    pc, dv = make_engines()
    print("── 建库 (train) ──")
    pc.train(train)
    dv.train(train)
    tau_pc, tau_dv = pc._calibrated_threshold, dv._calibrated_threshold
    n_cal_pc = calibrated_n_samples(pc)
    n_cal_dv = calibrated_n_samples(dv)
    print(f"  PC tau={tau_pc:.4f} n_cal={n_cal_pc} | "
          f"DV tau={tau_dv:.4f} n_cal={n_cal_dv}")

    print("── 打分: 留出正常 + 缺陷 ──")
    pc_hold, _ = score_paths(pc, holdout, "PC留出")
    dv_hold, _ = score_paths(dv, holdout, "DV留出")
    pc_def, _ = score_paths(pc, defects, "PC缺陷")
    dv_def, _ = score_paths(dv, defects, "DV缺陷")
    pc_h, dv_h = pc_hold > tau_pc, dv_hold > tau_dv
    pc_d, dv_d = pc_def > tau_pc, dv_def > tau_dv

    # ── GDINO (可选, 默认开): runtime registry + config ──
    g_h = np.zeros(len(holdout), dtype=bool)
    g_d = np.zeros(len(defects), dtype=bool)
    gdino_on = bool(getattr(args, "gdino", True))
    gdino_err = 0
    if gdino_on:
        try:
            from core.config import load_config
            from engines.registry import EngineRegistry
            cfg = load_config()
            reg = EngineRegistry(cfg)
            reg.register_all()
            print("── GDINO (runtime conf) ──")
            g_h, e1 = _gdino_flags(reg, cfg, holdout, "GDINO留出")
            g_d, e2 = _gdino_flags(reg, cfg, defects, "GDINO缺陷")
            gdino_err = e1 + e2
        except Exception as e:  # noqa: BLE001
            print(f"  GDINO 不可用, 跳过 (错误: {e})")
            gdino_on = False

    # ── 逐图 ng_sources ──
    hold_src = [["patchcore"] * int(a) + ["dinov2"] * int(b)
                + ["dino"] * int(c)
                for a, b, c in zip(pc_h, dv_h, g_h)]
    def_src = [["patchcore"] * int(a) + ["dinov2"] * int(b)
               + ["dino"] * int(c)
               for a, b, c in zip(pc_d, dv_d, g_d)]

    # ── 基线: Union OR (v1.3.0, 任一 NG 即 NG) ──
    base_h = np.asarray([len(s) > 0 for s in hold_src])
    base_d = np.asarray([len(s) > 0 for s in def_src])
    base_fpr = float(base_h.mean())
    base_recall = float(base_d.mean())

    # ── 分阶段融合 ──
    from core.config import load_config as _lc
    fusion_cfg = _lc()["qc"].get("union", {}).get("fusion", {})
    n_cal_by_source = {"patchcore": n_cal_pc, "dinov2": n_cal_dv}

    def apply_fusion(srcs):
        out = []
        for s in srcs:
            r = staged_fusion(s, n_cal_by_source, fusion_cfg)
            out.append(r)
        return out

    fus_h = apply_fusion(hold_src)
    fus_d = apply_fusion(def_src)
    stage = fus_d[0]["stage"] if fus_d else None

    v_h = np.asarray([r["verdict"] for r in fus_h])
    v_d = np.asarray([r["verdict"] for r in fus_d])
    st_fpr_ng = float(np.mean(v_h == "NG"))          # 自主 NG 误报
    st_fpr_review = float(np.mean(v_h == "REVIEW"))  # 转人工 (不计数)
    st_recall_ng = float(np.mean(v_d == "NG"))       # 自主召回 (严格)
    st_recall_eff = float(np.mean(
        (v_d == "NG") | (v_d == "REVIEW")))          # 有效召回 (零漏检口径)

    # 不变量自检: 分阶段融合不得把基线捕获的缺陷静默放行 OK
    leaked = int(np.sum(base_d & (v_d == "OK")))

    # 验收: 有效召回不降 + 自主NG FPR-hold 下降 ≥30%
    fpr_drop = (base_fpr - st_fpr_ng) / base_fpr if base_fpr > 0 else 0.0
    acc_pass = bool(st_recall_eff >= base_recall - 1e-9
                    and fpr_drop >= 0.30 and leaked == 0)

    out = {
        "dataset": "kolektor-fusion55",
        "n": {"train": len(train), "holdout": len(holdout),
              "defects": len(defects)},
        "n_cal": {"patchcore": n_cal_pc, "dinov2": n_cal_dv},
        "fusion_stage": stage, "fusion_cfg": fusion_cfg,
        "gdino": {"enabled": gdino_on, "errors": gdino_err,
                  "fpr_hold": float(g_h.mean()),
                  "recall": float(g_d.mean())},
        "baseline_union_or": {"fpr_hold": base_fpr,
                              "recall": base_recall},
        "staged_fusion": {
            "fpr_hold_autong": st_fpr_ng,
            "fpr_hold_review": st_fpr_review,
            "recall_autong": st_recall_ng,
            "recall_effective": st_recall_eff,
            "leaked_defects": leaked,
        },
        "acceptance": {
            "criterion": ("有效召回(NG+REVIEW)不降 且 自主NG FPR-hold 较 "
                          "Union OR 基线下降≥30% 且 无缺陷被静默放行OK"),
            "base_fpr": base_fpr, "staged_fpr_autong": st_fpr_ng,
            "fpr_drop": round(fpr_drop, 4),
            "base_recall": base_recall,
            "staged_recall_eff": st_recall_eff,
            "leaked": leaked, "pass": acc_pass,
        },
    }
    print("\n══ Union OR 基线 ══")
    print(f"  FPR-hold={base_fpr:.2%}  Recall={base_recall:.2%}")
    print(f"══ 分阶段融合 (stage={stage}, n_cal="
          f"{min(n_cal_pc or 999, n_cal_dv or 999)}) ══")
    print(f"  自主NG FPR-hold={st_fpr_ng:.2%}  "
          f"REVIEW率={st_fpr_review:.2%}")
    print(f"  自主Recall={st_recall_ng:.2%}  "
          f"有效Recall(NG+REVIEW)={st_recall_eff:.2%}")
    print(f"  泄漏缺陷(被放行OK)={leaked}")
    print(f"\n验收: FPR-hold {base_fpr:.2%}→{st_fpr_ng:.2%} "
          f"(降 {fpr_drop:.0%}, 门槛≥30%) | 有效Recall "
          f"{base_recall:.2%}→{st_recall_eff:.2%} | 泄漏 {leaked} → "
          f"{'PASS' if acc_pass else 'FAIL'}")
    pc.unload()
    dv.unload()
    return out


def mode_calibration(args) -> dict:
    """§6.2 校准协议验收: 小批建库 (n_cal=3) → 补独立校准图 → NP 重标定。

    模拟真实 onboarding 工作流:
      建库 (10 张, n_cal=3, 融合 Stage 1) → 测 FPR/Recall
      → 校准协议 (独立 60 张校准集, recalibrate_engine) → 重测
    验收标准:
      1. n_cal 3 → ≥30, 融合阶段 Stage 1 → ≥2
      2. 协议后 FPR-hold ≤ 协议前 (重标定不放宽误报)
      3. Union Recall 协议后 ≥ 协议前 − 10pts (FPR-Recall 交换容忍,
         实测如实报告; 零漏检口径下漏检由人工复核兜底)
      4. bank npz 持久化回读: 重标定后 np_calib_json n_samples = 校准图数
    """
    from core.fusion import calibrated_n_samples, fusion_stage
    from core.np_calibration import recalibrate_engine

    root = Path(args.root)
    normals, defects = scan_kolektor(root)
    rng = np.random.default_rng(2026)
    idx = rng.permutation(len(normals))
    bank_paths = [normals[i] for i in idx[:10]]          # 小批建库
    cal_paths = [normals[i] for i in idx[10:70]]          # 独立校准集 (60)
    holdout = [normals[i] for i in idx[70:]]              # 真留出测FPR
    print(f"[calibration] 正常 {len(normals)}: 建库 {len(bank_paths)} / "
          f"校准 {len(cal_paths)} / 留出 {len(holdout)}, "
          f"缺陷 {len(defects)}")

    pc, dv = make_engines()

    # ── 前置态: 小批建库 (n_cal=3, Stage 1) ──
    print("── 建库 (10 张) ──")
    pc.train(bank_paths)
    dv.train(bank_paths)
    n_before = {"patchcore": calibrated_n_samples(pc),
                "dinov2": calibrated_n_samples(dv)}
    stage_before = fusion_stage(min(n_before.values()))
    print(f"  n_cal={n_before}, stage={stage_before}")

    print("── 协议前打分: 留出 + 缺陷 ──")
    pc_h0, _ = score_paths(pc, holdout, "PC留出")
    dv_h0, _ = score_paths(dv, holdout, "DV留出")
    pc_d0, _ = score_paths(pc, defects, "PC缺陷")
    dv_d0, _ = score_paths(dv, defects, "DV缺陷")
    fpr_before = float(np.mean((pc_h0 > pc._calibrated_threshold)
                                | (dv_h0 > dv._calibrated_threshold)))
    rec_before = float(np.mean((pc_d0 > pc._calibrated_threshold)
                               | (dv_d0 > dv._calibrated_threshold)))
    tau_before = {"patchcore": float(pc._calibrated_threshold),
                  "dinov2": float(dv._calibrated_threshold)}

    # ── 校准协议: 独立校准集打分 → 重标定 ──
    print("── 校准协议: 60 张独立校准图 ──")
    pc_cal, e1 = score_paths(pc, cal_paths, "PC校准")
    dv_cal, e2 = score_paths(dv, cal_paths, "DV校准")
    r_pc = recalibrate_engine(pc, list(pc_cal))
    r_dv = recalibrate_engine(dv, list(dv_cal))
    assert r_pc["ok"] and r_dv["ok"], f"重标定失败: {r_pc} / {r_dv}"
    n_after = {"patchcore": calibrated_n_samples(pc),
               "dinov2": calibrated_n_samples(dv)}
    stage_after = fusion_stage(min(n_after.values()))
    print(f"  n_cal={n_after}, stage={stage_after}")

    # ── 持久化回读校验: save_bank → load_bank → n_samples 一致 ──
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_pc = Path(td) / "pc.npz"
        tmp_dv = Path(td) / "dv.npz"
        pc.save_bank(tmp_pc, product_name="caltest")
        dv.save_bank(tmp_dv, product_name="caltest")
        pc2, dv2 = make_engines()
        assert pc2.load_bank(tmp_pc) and dv2.load_bank(tmp_dv)
        n_reload = {"patchcore": calibrated_n_samples(pc2),
                    "dinov2": calibrated_n_samples(dv2)}
        assert n_reload == n_after, f"回读不一致: {n_reload} != {n_after}"
        pc2.unload()
        dv2.unload()
    print(f"  npz 持久化回读 OK: {n_reload}")

    # ── 协议后打分 ──
    print("── 协议后打分: 留出 + 缺陷 ──")
    pc_h1, _ = score_paths(pc, holdout, "PC留出")
    dv_h1, _ = score_paths(dv, holdout, "DV留出")
    pc_d1, _ = score_paths(pc, defects, "PC缺陷")
    dv_d1, _ = score_paths(dv, defects, "DV缺陷")
    fpr_after = float(np.mean((pc_h1 > pc._calibrated_threshold)
                              | (dv_h1 > dv._calibrated_threshold)))
    rec_after = float(np.mean((pc_d1 > pc._calibrated_threshold)
                              | (dv_d1 > dv._calibrated_threshold)))
    tau_after = {"patchcore": float(pc._calibrated_threshold),
                 "dinov2": float(dv._calibrated_threshold)}

    acc_pass = bool(
        min(n_after.values()) >= 30 and stage_after >= 2
        and fpr_after <= fpr_before + 1e-9
        and rec_after >= rec_before - 0.10)

    out = {
        "dataset": "kolektor-calibration",
        "n": {"bank": len(bank_paths), "cal": len(cal_paths),
              "holdout": len(holdout), "defects": len(defects),
              "score_errors": len(e1) + len(e2)},
        "before": {"n_cal": n_before, "stage": stage_before,
                   "tau": tau_before, "fpr_hold": fpr_before,
                   "union_recall": rec_before},
        "after": {"n_cal": n_after, "stage": stage_after,
                  "tau": tau_after, "fpr_hold": fpr_after,
                  "union_recall": rec_after,
                  "npz_reload_n": n_reload},
        "acceptance": {
            "criterion": ("n_cal≥30 且 stage≥2 且 FPR-hold 不升 且 "
                          "Union Recall 降幅 ≤10pts"),
            "n_cal_ok": min(n_after.values()) >= 30,
            "stage_ok": stage_after >= 2,
            "fpr_ok": fpr_after <= fpr_before + 1e-9,
            "recall_ok": rec_after >= rec_before - 0.10,
            "pass": acc_pass,
        },
    }
    print("\n══ 协议前 (n_cal=3, Stage 1) ══")
    print(f"  FPR-hold={fpr_before:.2%}  Union Recall={rec_before:.2%}")
    print(f"══ 协议后 (n_cal={min(n_after.values())}, "
          f"Stage {stage_after}) ══")
    print(f"  FPR-hold={fpr_after:.2%}  Union Recall={rec_after:.2%}")
    print(f"  tau: PC {tau_before['patchcore']:.3f}→"
          f"{tau_after['patchcore']:.3f} | DV {tau_before['dinov2']:.3f}→"
          f"{tau_after['dinov2']:.3f}")
    print(f"\n验收: {'PASS' if acc_pass else 'FAIL'}")
    pc.unload()
    dv.unload()
    return out


def mode_dvab(args) -> dict:
    """§5.1 DINOv2 PixOOD 借鉴 A/B: 基线 vs 死etalon重初始化 (P1)
    vs per-etalon 局部NP归一化 (P4) vs 两者组合。

    特征缓存: _extract_features 包一层缓存 — 变体差异仅在 GMM 拟合
    与打分归一化, 原始 ViT 特征与变体无关, 避免 4 倍骨干推理。
    判定口径: 同一 eps=0.10 NP 校准下, Recall 不降 (容忍 2pts 波动)
    前提下比 FPR-hold 与 AUROC; 基线最优则如实记录不升级。
    """
    root = Path(args.root)
    normals, defects = scan_kolektor(root)
    rng = np.random.default_rng(2026)   # 与 mode_kolektor/fusion55 同划分
    idx = rng.permutation(len(normals))
    n_train = int(len(normals) * 0.8)
    train = [normals[i] for i in idx[:n_train]]
    holdout = [normals[i] for i in idx[n_train:]]
    print(f"[dvab] 正常 {len(normals)} (train {len(train)} / "
          f"holdout {len(holdout)}), 缺陷 {len(defects)}")

    from engines.vision.dinov2_anomaly import DINOv2AnomalyEngine
    eng = DINOv2AnomalyEngine(CFG)
    eng.load()
    assert eng.is_ready()

    cache: dict = {}
    orig_extract = eng._extract_features

    def cached_extract(image):
        key = image if isinstance(image, str) else None
        if key is not None and key in cache:
            return cache[key]
        f = orig_extract(image)
        if key is not None:
            cache[key] = f
        return f

    eng._extract_features = cached_extract

    variants = [
        ("baseline", False, False),
        ("reinit", True, False),
        ("localnp", False, True),
        ("reinit+localnp", True, True),
    ]
    rows = {}
    for name, do_reinit, do_localnp in variants:
        eng._reinit_dead = do_reinit
        eng._per_etalon_np = do_localnp
        eng._pca = eng._gmm = None
        eng._etalon_np_mu = eng._etalon_np_sigma = None
        eng._np_calibrator = None
        eng._calibrated_threshold = None
        print(f"── 变体: {name} ──")
        t0 = time.time()
        meta = eng.train(train)
        if meta.get("error"):
            rows[name] = {"error": meta["error"]}
            continue
        hold_s, e1 = score_paths(eng, holdout, f"{name}留出")
        def_s, e2 = score_paths(eng, defects, f"{name}缺陷")
        tau = float(eng._calibrated_threshold)
        rows[name] = {
            "train_sec": round(time.time() - t0, 1),
            "n_cal": int(eng._np_calibrator.n_samples),
            "tau": round(tau, 4),
            "auroc": round(auroc(def_s, hold_s), 4),
            "fpr_hold": round(float(np.mean(hold_s > tau)), 4),
            "recall": round(float(np.mean(def_s > tau)), 4),
            "errors": len(e1) + len(e2),
        }
        print(f"  [{name}] AUROC={rows[name]['auroc']} "
              f"FPR-hold={rows[name]['fpr_hold']} "
              f"Recall={rows[name]['recall']} "
              f"({rows[name]['train_sec']}s)")

    # ── 诚实推荐: Recall 不降 (容忍 2pts) 前提下比 FPR, 再比 AUROC ──
    base = rows.get("baseline", {})
    rec_base = base.get("recall", 0.0)
    fpr_base = base.get("fpr_hold", 1.0)
    auroc_base = base.get("auroc", 0.0)
    candidates = []
    for name, r in rows.items():
        if name == "baseline" or r.get("error"):
            continue
        if r["recall"] < rec_base - 0.02:
            continue  # Recall 降幅超容忍 → 不考虑
        candidates.append(name)
    if candidates:
        best = min(candidates, key=lambda n: (rows[n]["fpr_hold"],
                                              -rows[n]["auroc"]))
        improve = (rows[best]["fpr_hold"] < fpr_base - 1e-9
                   or rows[best]["auroc"] > auroc_base + 1e-9)
    else:
        best, improve = "baseline", False
    recommendation = best if improve else "baseline"

    out = {"dataset": "kolektor-dvab",
           "n": {"train": len(train), "holdout": len(holdout),
                 "defects": len(defects)},
           "variants": rows,
           "decision": {
               "criterion": ("Recall 降幅≤2pts 前提下 FPR-hold/AUROC 更优 "
                             "才升级; 否则保持基线"),
               "recall_tolerance_pts": 2,
               "best": best, "improves_over_baseline": improve,
               "recommendation": recommendation,
           }}
    print(f"\n══ A/B 结论: 推荐 {recommendation} "
          f"{'(升级)' if improve else '(基线已最优, 不升级)'} ══")
    eng.unload()
    return out


def mode_prompts(args) -> dict:
    """§5.3 GDDM 提示词挖掘 A/B 门控: 基线 DEFAULT_PROMPT vs 挖掘词集。

    门控口径 (不达标即砍): 挖掘词集存在某 conf 工作点满足
    Recall≥20% ∧ FPR-hold≤10% ∧ 平均框数≤3, 且优于基线最佳工作点,
    才建议采纳; 否则 GDINO 对该数据集仍无可用工作点, §5.3 如实砍掉。
    效率: 每 (图, 词集) 仅在 --confs 最低 conf 推理一次, 事后按各 conf
    过滤计数 (框分数单调, 结果等价)。
    """
    from core.config import load_config
    from core.defect_detector import DEFAULT_PROMPT, translate_prompt
    from engines.registry import EngineRegistry

    mined_path = Path(args.mined or "")
    if not mined_path.is_file():
        raise FileNotFoundError(
            "缺少挖掘结果, 先运行: python scripts/mine_prompts.py "
            "--root <数据集> --out mined_prompts.json")
    mined = json.loads(mined_path.read_text(encoding="utf-8"))
    mined_prompt = mined.get("prompt_candidate", "")
    if not mined_prompt.strip():
        raise ValueError("挖掘结果为空 (prompt_candidate 无词)")

    root = Path(args.root)
    normals, defects = scan_kolektor(root)
    rng = np.random.default_rng(2026)   # 与 fusion55 同划分 (holdout 口径一致)
    idx = rng.permutation(len(normals))
    n_train = int(len(normals) * 0.8)
    holdout = [normals[i] for i in idx[n_train:]]
    print(f"[prompts] 正常 {len(normals)} (holdout {len(holdout)}), "
          f"缺陷 {len(defects)}")

    confs = sorted({round(float(c), 3) for c in args.confs.split(",")
                    if c.strip()})
    if not confs:
        raise ValueError("--confs 解析为空")
    conf_min = confs[0]

    cfg = load_config()
    reg = EngineRegistry(cfg)
    reg.register_all()
    engine = reg.get("grounding_dino")
    if engine is None:
        raise RuntimeError("grounding_dino 引擎未注册")
    if not engine.is_ready():
        reg.ensure_loaded("grounding_dino")
    if not engine.is_ready():
        raise RuntimeError("GDINO 模型加载失败")

    sets = {"baseline": translate_prompt(DEFAULT_PROMPT),
            "mined": mined_prompt}
    print(f"  基线提示词: {sets['baseline']}")
    print(f"  挖掘提示词: {sets['mined']}")
    print(f"  conf 网格: {confs} (单次推理@{conf_min}, 事后过滤)")

    results: dict = {}
    for name, prompt in sets.items():
        det_scores: dict[str, list[float]] = {}
        errors = 0
        pool = [(p, "holdout") for p in holdout] + \
               [(p, "defect") for p in defects]
        for i, (p, pool_tag) in enumerate(pool):
            r = engine.infer(p, prompt=prompt, threshold=conf_min)
            if r.get("error"):
                errors += 1
                det_scores[p] = []
            else:
                det_scores[p] = [float(s) for s in r["scores"]]
            if (i + 1) % 25 == 0:
                print(f"    [{name}] {i + 1}/{len(pool)}", flush=True)
        per_conf = {}
        for c in confs:
            n_box_h = [sum(1 for s in det_scores[p] if s >= c)
                       for p in holdout]
            n_box_d = [sum(1 for s in det_scores[p] if s >= c)
                       for p in defects]
            per_conf[str(c)] = {
                "recall": round(float(np.mean(np.asarray(n_box_d) > 0)), 4),
                "fpr_hold": round(float(np.mean(np.asarray(n_box_h) > 0)), 4),
                "avg_boxes_hold": round(float(np.mean(n_box_h)), 3),
                "avg_boxes_defect": round(float(np.mean(n_box_d)), 3),
                "max_boxes": int(max(n_box_h + n_box_d)),
            }
        results[name] = {"errors": errors, "by_conf": per_conf}
        print(f"    [{name}] 完成, 错误 {errors}")

    def best_working_point(by_conf: dict) -> tuple | None:
        """门控工作点: Recall≥20% ∧ FPR≤10% ∧ 平均框≤3, 取 Recall 最高。"""
        cand = [(c, m) for c, m in by_conf.items()
                if m["recall"] >= 0.20 and m["fpr_hold"] <= 0.10
                and m["avg_boxes_hold"] <= 3.0]
        if not cand:
            return None
        return max(cand, key=lambda t: t[1]["recall"])

    wp_base = best_working_point(results["baseline"]["by_conf"])
    wp_mined = best_working_point(results["mined"]["by_conf"])
    if wp_mined is not None and (
            wp_base is None
            or wp_mined[1]["recall"] >= wp_base[1]["recall"] + 0.05):
        decision, reason = "adopt", \
            f"挖掘词集存在可用工作点 conf={wp_mined[0]} 且优于基线"
    elif wp_base is not None:
        decision, reason = "retain_baseline", \
            "基线已有可用工作点, 挖掘词集未带来显著提升"
    else:
        decision, reason = "cut", \
            "两组词集均无可用工作点 — GDINO 对该数据集不适用, §5.3 砍掉"

    out = {"dataset": "kolektor-prompts",
           "n": {"holdout": len(holdout), "defects": len(defects)},
           "confs": confs,
           "prompts": sets,
           "mined_source": str(mined_path),
           "results": results,
           "gate": {
               "criterion": ("存在 conf 工作点: Recall≥20% ∧ FPR-hold≤10% "
                             "∧ 平均框数≤3; 采纳需优于基线最佳工作点"),
               "baseline_working_point": (
                   {"conf": wp_base[0], **wp_base[1]} if wp_base else None),
               "mined_working_point": (
                   {"conf": wp_mined[0], **wp_mined[1]} if wp_mined else None),
               "decision": decision,
               "reason": reason,
           }}
    print(f"\n══ §5.3 门控判决: {decision} ══")
    print(f"  {reason}")
    for name in ("baseline", "mined"):
        for c, m in results[name]["by_conf"].items():
            print(f"  {name:<9} conf={c:<5} Recall={m['recall']:.1%} "
                  f"FPR-hold={m['fpr_hold']:.1%} "
                  f"平均框(正常)={m['avg_boxes_hold']}")
    return out


def main():
    ap = argparse.ArgumentParser(description="多数据集验收评估")
    ap.add_argument("mode",
                    choices=["kolektor", "pcb", "yolo", "paired",
                             "bootstrap", "subspacead", "fusion55",
                             "calibration", "dvab", "prompts"])
    ap.add_argument("root", nargs="?", default="")
    ap.add_argument("def_dir", nargs="?", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--bank-frac", type=float, default=0.75)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-gdino", dest="gdino", action="store_false",
                    help="fusion55: 跳过 GDINO (仅表面双源)")
    ap.add_argument("--mined", default="",
                    help="prompts: mine_prompts.py 输出的挖掘结果 JSON")
    ap.add_argument("--confs", default="0.2,0.3,0.5",
                    help="prompts: GDINO 置信度网格 (逗号分隔)")
    ap.set_defaults(gdino=True)
    args = ap.parse_args()

    handlers = {"kolektor": mode_kolektor, "pcb": mode_pcb,
                "yolo": mode_yolo, "paired": mode_paired,
                "bootstrap": mode_bootstrap,
                "subspacead": mode_subspacead,
                "fusion55": mode_fusion55,
                "calibration": mode_calibration,
                "dvab": mode_dvab,
                "prompts": mode_prompts}
    # paired: root=ok_dir, def_dir=def_dir; bootstrap: root=dir
    if args.mode == "bootstrap":
        args.dir = args.root
    result = handlers[args.mode](args)

    out_path = Path(args.out) if args.out else Path(
        f"acceptance_{args.mode}_{args.name or 'x'}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\n结果已写: {out_path}")


if __name__ == "__main__":
    main()
