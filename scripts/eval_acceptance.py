"""多数据集验收评估 — PatchCore-NP / DINOv2 / 双源Union OR

用法 (路径全部经 argv 传入, 仓库内不硬编码任何数据路径):

  python scripts/eval_acceptance.py kolektor <root> [--out X.json]
      mask 标注表面缺陷 (Part*.jpg + Part*_label.bmp), 80/20 划分 (seed 2026)

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


# ─── main ──────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="多数据集验收评估")
    ap.add_argument("mode",
                    choices=["kolektor", "pcb", "yolo", "paired",
                             "bootstrap"])
    ap.add_argument("root", nargs="?", default="")
    ap.add_argument("def_dir", nargs="?", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--bank-frac", type=float, default=0.75)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    handlers = {"kolektor": mode_kolektor, "pcb": mode_pcb,
                "yolo": mode_yolo, "paired": mode_paired,
                "bootstrap": mode_bootstrap}
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
