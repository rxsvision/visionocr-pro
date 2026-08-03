"""校准协议 (方案 §6.2) — n_cal 扩充 + 验收报告

背景:
    建库时引擎仅用上传列表尾部 20% holdout 做 NP 校准。UI 建议 10~30 张
    → n_cal 只有 3~6, 远低于分阶段融合的 Stage 2/3 门槛 (10/50), 产品
    实际永远停留在 Stage 1 (纯 OR, 误报偏高)。这就是"n_cal=3 问题"。

协议流程:
    1. 建库 (OK ≥8, 现有 register_ok_samples, 不动)
    2. 补采独立校准图 ≥30 张 — 不得是建库图; 建议变换光照/角度拍 3 组,
       覆盖产线真实正常件分布
    3. 校准图对已建库逐源打分 → NPCalibrator 重拟合 → tau 更新
       (统计保证不变: P(正常 > tau) ≤ epsilon; 样本增多 → 阈值粒度收紧)
    4. 持久化: 校准集落盘 data/calibration/{product}/{时间戳}/ (随银行版本化,
       时间戳子目录保留历史); 重标定写回 bank npz (np_calib_json)
    5. 验收报告: n_cal / epsilon / tau(旧→新) / 融合阶段变化;
       若提供 NG 缺陷样本, 附加 Recall 实测回归

设计原则:
    - 校准图绝不进入 bank (self-match 偏差压低 tau → FPR 膨胀)。
    - 重标定"替换"而非"合并"建库期 holdout 分数: 建库与补采可能隔了
      光照/季节漂移, 最新校准集代表当前产线分布, 语义清晰。旧 tau 记入
      manifest 供审计。
    - SubspaceAD 不参与: 辅助提示通道, 快速模式增广自校准已知偏乐观,
      其 tau 不参与自主判定, 重标定无收益。
    - Recall 回归仅当用户提供 NG 样本时才有; 只有 OK 图时报告只承诺
      FPR 上界 (epsilon), 不虚构 Recall。
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from core import anomaly_bank
from core.np_calibration import recalibrate_engine

logger = logging.getLogger("visionocr.calib_protocol")

_CAL_ROOT = Path("data/calibration")

# 协议建议值 (方案 §6.2: 校准样本 ≥30, 光照/角度 3 组)
MIN_CAL_RECOMMENDED = 30

# 参与校准协议的表面源: (fusion源名, registry引擎名,
#                         anomaly_bank 加载函数名, bank路径函数名)
# 按名称在调用时解析 (而非 import 时捕获), 便于测试 monkeypatch。
_SOURCES = [
    ("patchcore", "anomalib", "load_product_bank", "bank_path"),
    ("dinov2", "dinov2_anomaly", "load_product_bank_dinov2",
     "bank_path_dinov2"),
]


def _safe_name(product_name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", product_name.strip())


def calibration_dir(product_name: str) -> Path:
    """产品校准集根目录 data/calibration/{product}/。"""
    return _CAL_ROOT / _safe_name(product_name)


def _score_images(engine, image_paths: list[str]) -> tuple[list[float], int]:
    """逐图推理收集图像级分数。返回 (有效分数, 失败张数)。"""
    scores: list[float] = []
    n_fail = 0
    for p in image_paths:
        try:
            r = engine.infer(p)
        except Exception as e:  # noqa: BLE001 — 单图失败不中断协议
            logger.warning("校准图推理异常 %s: %s", p, e)
            n_fail += 1
            continue
        if not isinstance(r, dict) or r.get("error") or r.get("pred_label") == "ERROR":
            n_fail += 1
            continue
        s = r.get("score")
        if s is None:
            n_fail += 1
            continue
        scores.append(float(s))
    return scores, n_fail


def _persist_session(product: str, cal_paths: list[str],
                     manifest: dict) -> Path:
    """校准图 + manifest 落盘到时间戳子目录 (保留历史版本)。"""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    session = calibration_dir(product) / ts
    session.mkdir(parents=True, exist_ok=True)
    copied = []
    for i, p in enumerate(cal_paths, 1):
        src = Path(p)
        dst = session / f"cal_{i:04d}{src.suffix.lower() or '.png'}"
        try:
            shutil.copy2(src, dst)
            copied.append(dst.name)
        except OSError as e:
            logger.warning("校准图落盘失败 %s: %s", src, e)
    manifest["copied_files"] = copied
    (session / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return session


def recalibrate_product(registry, product_name: str,
                        cal_image_paths: list[str],
                        ng_image_paths: Optional[list[str]] = None,
                        fusion_cfg: Optional[dict] = None) -> dict:
    """对已建库产品执行校准协议 (§6.2)。

    Args:
        registry: EngineRegistry。
        product_name: 已建库产品名。
        cal_image_paths: 独立校准 OK 图 (不得入 bank; 建议 ≥30, 光照/角度 3 组)。
        ng_image_paths: 可选 NG 缺陷样本 → Recall 回归实测。
        fusion_cfg: qc.union.fusion 配置段 (用于阶段门槛, 可空 → 默认)。

    Returns:
        {"ok": bool, "product": str, "error": str|None,
         "sources": {源名: {"status": str, "n_cal": int, "epsilon": float,
                            "tau_before": float|None, "tau": float|None,
                            "score_mean": float, "score_max": float,
                            "n_scored": int, "n_failed": int,
                            "ng_recall": float|None, ...}},
         "n_cal_min": int|None, "stage_before": int, "stage_after": int,
         "session_dir": str|None, "ng_regression": {...}|None}
    """
    from core.fusion import calibrated_n_samples, fusion_stage

    out: dict = {"ok": False, "product": product_name, "error": None,
                 "sources": {}, "n_cal_min": None,
                 "stage_before": 1, "stage_after": 1,
                 "session_dir": None, "ng_regression": None}

    cal_paths = [p for p in (cal_image_paths or []) if Path(p).is_file()]
    if len(cal_paths) < 3:
        out["error"] = (f"有效校准图仅 {len(cal_paths)} 张, 至少 3 张 "
                        f"(建议 ≥{MIN_CAL_RECOMMENDED} 张)")
        return out

    # ── 阶段基线 (重标定前) ──
    n_before: dict[str, Optional[int]] = {}
    engines: dict[str, object] = {}
    for src, eng_name, _loader_n, _path_n in _SOURCES:
        eng = registry.get(eng_name) if registry is not None else None
        engines[src] = eng
        n_before[src] = None

    def _min_n(nd: dict) -> Optional[int]:
        vals = [v for v in nd.values() if v]
        return min(vals) if vals else None

    # 先尝试加载 bank 再读 stage_before (bank 未载入内存时 n 为 None)
    bank_ok = {}
    for src, eng_name, loader_n, path_n in _SOURCES:
        if engines[src] is None:
            bank_ok[src] = False
            continue
        try:
            loader = getattr(anomaly_bank, loader_n)
            bank_ok[src] = bool(loader(registry, product_name))
        except Exception as e:  # noqa: BLE001
            logger.warning("加载 %s bank 失败: %s", src, e)
            bank_ok[src] = False
        if bank_ok[src]:
            n_before[src] = calibrated_n_samples(engines[src])
    out["stage_before"] = fusion_stage(_min_n(n_before), fusion_cfg)

    if not any(bank_ok.values()):
        out["error"] = (f"产品「{product_name}」无可用特征库 "
                        f"(PatchCore/DINOv2 均未建库), 请先注册建库")
        return out

    # ── 逐源打分 + 重标定 ──
    ng_paths = [p for p in (ng_image_paths or []) if Path(p).is_file()]
    ng_scores: dict[str, list[float]] = {}

    for src, eng_name, loader_n, path_n in _SOURCES:
        rec: dict = {"status": "skipped", "n_cal": 0, "epsilon": None,
                     "tau_before": None, "tau": None, "score_mean": None,
                     "score_max": None, "n_scored": 0, "n_failed": 0,
                     "ng_recall": None}
        out["sources"][src] = rec
        eng = engines[src]
        if eng is None:
            rec["status"] = "引擎未注册"
            continue
        if not bank_ok[src]:
            rec["status"] = "无特征库 (未建库或加载失败)"
            continue

        scores, n_fail = _score_images(eng, cal_paths)
        rec["n_scored"], rec["n_failed"] = len(scores), n_fail
        if scores:
            rec["score_mean"] = round(float(np.mean(scores)), 4)
            rec["score_max"] = round(float(np.max(scores)), 4)

        res = recalibrate_engine(eng, scores)
        rec["tau_before"] = res["tau_before"]
        rec["epsilon"] = res["epsilon"]
        if not res["ok"]:
            rec["status"] = f"重标定失败: {res['error']}"
            continue
        rec["n_cal"] = res["n"]
        rec["tau"] = res["tau"]
        rec["status"] = "ok"

        # 重标定写回 bank npz (np_calib_json 更新, bank 本体不变)
        try:
            bank_p = getattr(anomaly_bank, path_n)(product_name)
            eng.save_bank(bank_p, product_name=product_name)
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"ok (重标定成功, 落盘失败: {e})"
            logger.warning("%s bank 重存失败: %s", src, e)

        # NG 回归 (可选)
        if ng_paths:
            ng_s, _ = _score_images(eng, ng_paths)
            ng_scores[src] = ng_s
            if ng_s and rec["tau"] is not None:
                rec["ng_recall"] = round(
                    sum(1 for s in ng_s if s > rec["tau"]) / len(ng_s), 4)

    # ── 融合阶段 (重标定后) ──
    n_after = {src: (out["sources"][src]["n_cal"] or None)
               for src in out["sources"]}
    out["n_cal_min"] = _min_n(n_after)
    out["stage_after"] = fusion_stage(out["n_cal_min"], fusion_cfg)

    # ── NG 回归汇总 (Union OR 口径) ──
    if ng_paths and ng_scores:
        n_ng = len(ng_paths)
        union_hit = 0
        for i in range(n_ng):
            if any(i < len(ss) and ss[i] > (out["sources"][s]["tau"] or 0.0)
                   for s, ss in ng_scores.items()
                   if out["sources"][s]["tau"] is not None):
                union_hit += 1
        out["ng_regression"] = {
            "n_defects": n_ng,
            "union_recall": round(union_hit / n_ng, 4) if n_ng else None,
        }

    # ── 校准集落盘 (含审计 manifest) ──
    manifest = {
        "product": product_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_cal_images": len(cal_paths),
        "source_files": [Path(p).name for p in cal_paths],
        "sources": out["sources"],
        "n_cal_min": out["n_cal_min"],
        "stage_before": out["stage_before"],
        "stage_after": out["stage_after"],
        "ng_regression": out["ng_regression"],
        "note": "校准图独立于 bank (不入库); 重标定替换建库期 holdout 校准",
    }
    try:
        session = _persist_session(product_name, cal_paths, manifest)
        out["session_dir"] = str(session)
    except OSError as e:
        logger.warning("校准集落盘失败 (不影响重标定生效): %s", e)

    out["ok"] = any(r["status"] == "ok"
                    for r in out["sources"].values())
    if not out["ok"]:
        out["error"] = "所有源重标定均失败"
    return out


def format_report_md(result: dict) -> str:
    """验收报告 Markdown (UI 展示 + 人工留档)。"""
    if not result.get("ok") and result.get("error"):
        return f"⚠ 校准协议未执行: {result['error']}"

    lines = [f"### 📐 校准协议验收报告 — {result['product']}", ""]
    srcs = result["sources"]

    lines.append("| 源 | 状态 | n_cal | ε | τ (旧→新) | 校准分数 均值/最大 |")
    lines.append("|---|---|---|---|---|---|")
    for src, r in srcs.items():
        name = {"patchcore": "PatchCore", "dinov2": "DINOv2"}.get(src, src)
        if r["status"] != "ok":
            lines.append(f"| {name} | {r['status']} | — | — | — | — |")
            continue
        tb = f"{r['tau_before']:.4f}" if r["tau_before"] is not None else "无"
        sm = (f"{r['score_mean']:.3f} / {r['score_max']:.3f}"
              if r["score_mean"] is not None else "—")
        lines.append(f"| {name} | ✓ | {r['n_cal']} | {r['epsilon']:.2f} "
                     f"| {tb} → {r['tau']:.4f} | {sm} |")
    lines.append("")

    eps_txt = next((f"{r['epsilon']:.0%}" for r in srcs.values()
                    if r["epsilon"]), "—")
    lines.append(
        f"- NP 保证: 正常件误报率 ≤ ε = {eps_txt} "
        f"(有限样本 split-conformal, 校准图不入 bank)")
    lines.append(
        f"- 融合阶段: Stage {result['stage_before']} → "
        f"**Stage {result['stage_after']}** "
        f"(n_cal 最小值 = {result['n_cal_min']}; 门槛: Stage2 ≥10, Stage3 ≥50)")

    ng = result.get("ng_regression")
    if ng:
        lines.append(
            f"- NG 回归 (用户提供 {ng['n_defects']} 张缺陷图): "
            f"Union Recall = {ng['union_recall']:.1%}")
        for src, r in srcs.items():
            if r.get("ng_recall") is not None:
                lines.append(f"  - {src}: Recall = {r['ng_recall']:.1%}")
        lines.append("  - ⚠ 小样本回归置信区间宽, 仅作方向性参考")

    if result.get("session_dir"):
        lines.append(f"- 校准集已存档: `{result['session_dir']}`")

    # ── 指引 ──
    hints = []
    n_min = result.get("n_cal_min") or 0
    if n_min < MIN_CAL_RECOMMENDED:
        hints.append(f"校准图 < {MIN_CAL_RECOMMENDED} 张, 建议补足后重跑 (拍 3 组不同光照/角度)")
    if result["stage_after"] == 1:
        hints.append("n_cal < 10: 融合仍为 Stage 1 (纯 OR)。补 ≥10 张校准图可升级双源互证 (降误报)")
    elif result["stage_after"] == 2:
        hints.append("已达 Stage 2 双源互证; 补到 ≥50 张可进 Stage 3 (漂移监控)")
    jumped = [src for src, r in srcs.items()
              if r["tau"] is not None and r["tau_before"]
              and r["tau"] > 2.0 * r["tau_before"]]
    if jumped and not ng:
        hints.append("τ 显著上移 (校准分布比建库期宽): 建议补传 NG 样本跑 Recall 回归确认不漏检")
    if hints:
        lines.append("- 指引: " + "; ".join(hints))
    return "\n".join(lines)
