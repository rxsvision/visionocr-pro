"""分阶段融合决策 (v1.4.0 §5.5)

背景:
    v1.3.0 Union = 四源纯 OR (任一 NG → 终判 NG)。零漏检有余, 误报偏高:
    未校准源 (GroundingDINO 固定置信度阈值) 与校准源 (NP 阈值) 一视同仁,
    KolektorSDD 实测 holdout FPR 主要来自单源孤证。

本模块按"校准样本量 n_cal"分阶段收紧判决, 目标: Recall 不降 (有效召回 =
自主 NG + REVIEW 黄牌, 任何可疑图都不会被静默放行) 前提下大幅压 FPR。

阶段定义 (n_cal = 参与校准表面源中最小的 NP 校准样本数):
    Stage 1 (n_cal < stage2_min_cal, 默认 10):
        保守 OR — 阈值粒度 ~1/n 过粗, 融合无统计收益, 沿用 v1.3.0 行为。
    Stage 2 (10 ≤ n_cal < stage3_min_cal, 默认 50):
        校准一致投票 —
          · 校准源 ≥2 个 NG                 → NG
          · 校准源 1 个 NG + 未校准源 ≥1 NG → NG (跨模态互证)
          · 其余"任一源 NG"                 → REVIEW (黄牌, 强制人工复核)
    Stage 3 (n_cal ≥ 50):
        Stage 2 规则 + DriftMonitor 漂移监控 (仅 log 预警 + 结果字段,
        不自动改判决; 自适应重训属 v1.5 范围)。

设计边界 (诚实声明):
    1. 原方案"权重=各源校准集 Youden's J"需要缺陷标签, 而产品建库登记
       仅 OK 图 → 无法计算。校准源权重一律取 1; 在"≥2 源才判 NG"语义下
       与加权投票等价, 此处如实记录偏差。
    2. REVIEW 不是漏检: 零漏检政策的硬约束是"不允许缺陷被静默判 OK",
       REVIEW 图进入人工复核队列。验收口径为
       有效 Recall(NG+REVIEW) 不降 + 自主 NG 的 FPR 下降。
    3. 校准源不足 2 个参与时, 一致投票不可行 → 回退参与源的 OR
       (宁可误报不漏检), 并在结果中注明 fallback。
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Optional

logger = logging.getLogger("visionocr.fusion")

# 源分类: 校准源 = 有 NP 校准阈值的表面异常引擎;
# 未校准源 = 固定置信度阈值的结构检测引擎
CALIBRATED_SOURCES = ("patchcore", "dinov2")
UNCALIBRATED_SOURCES = ("dino", "yolo")

_DEFAULTS = {
    "mode": "staged",        # staged | or (or=v1.3.0 行为, 兼容回退)
    "stage2_min_cal": 10,
    "stage3_min_cal": 50,
    "drift_window": 200,
    "drift_min_samples": 30,
    "drift_rate_factor": 3.0,   # 近期 NG 率 > max(3*eps, eps+0.05) → 预警
    "drift_eps_floor": 0.05,
}


def cfg_get(fusion_cfg: dict | None, key: str):
    """带默认值的配置读取 (容忍 None/缺键)。"""
    v = (fusion_cfg or {}).get(key, _DEFAULTS[key])
    return v


def calibrated_n_samples(engine) -> Optional[int]:
    """引擎 NP 校准样本数 (未拟合/无校准器 → None)。"""
    cal = getattr(engine, "_np_calibrator", None)
    if cal is None or not getattr(cal, "is_fitted", False):
        return None
    n = int(getattr(cal, "n_samples", 0) or 0)
    return n if n > 0 else None


def fusion_stage(n_cal: Optional[int], fusion_cfg: dict | None = None) -> int:
    """按校准样本量返回阶段号 (1/2/3); n_cal=None → Stage 1。"""
    if n_cal is None:
        return 1
    if n_cal >= int(cfg_get(fusion_cfg, "stage3_min_cal")):
        return 3
    if n_cal >= int(cfg_get(fusion_cfg, "stage2_min_cal")):
        return 2
    return 1


def staged_fusion(ng_sources: list[str],
                  n_cal_by_source: dict[str, Optional[int]],
                  fusion_cfg: dict | None = None) -> dict:
    """分阶段融合判决 (纯函数, 无副作用, 便于单测)。

    Args:
        ng_sources: 本次检测判 NG 的源列表 (顺序无关)。
        n_cal_by_source: 校准源 → NP 校准样本数 (未参与/未拟合 → None)。
                         仅用于定阶段与判断"校准源是否参与"。
        fusion_cfg: qc.union.fusion 配置段 (可空 → 全默认)。

    Returns:
        {"verdict": "OK"/"NG"/"REVIEW", "stage": 1/2/3,
         "n_cal": int|None, "mode": str,
         "review_required": bool, "review_reasons": [str],
         "fallback_or": bool}
    """
    mode = str(cfg_get(fusion_cfg, "mode")).lower()
    ng = [s for s in ng_sources
          if s in CALIBRATED_SOURCES or s in UNCALIBRATED_SOURCES]
    cal_present = [s for s in CALIBRATED_SOURCES
                   if (n_cal_by_source or {}).get(s)]
    n_cal = min((n_cal_by_source[s] for s in cal_present), default=None)

    # 阶段由校准状态决定 (与单图是否 NG 无关) — 漂移监控依赖它对
    # 每张图 (含正常 OK 图) 都报告正确阶段
    stage = fusion_stage(n_cal, fusion_cfg)
    out = {"verdict": "OK", "stage": stage, "n_cal": n_cal, "mode": mode,
           "review_required": False, "review_reasons": [],
           "fallback_or": False}

    if not ng:
        return out

    # ── 兼容回退: mode=or → v1.3.0 纯 OR ──
    if mode == "or":
        out["verdict"] = "NG"
        return out

    # ── Stage 1: 保守 OR (校准不足, 零漏检优先) ──
    if stage == 1:
        out["verdict"] = "NG"
        return out

    # ── Stage 2/3: 校准一致投票 ──
    ng_cal = [s for s in ng if s in CALIBRATED_SOURCES]
    ng_uncal = [s for s in ng if s in UNCALIBRATED_SOURCES]

    # 校准源参与数 <2 → 一致投票不可行, 回退 OR (记录 fallback)
    if len(cal_present) < 2:
        out["verdict"] = "NG"
        out["fallback_or"] = True
        out["review_reasons"].append(
            f"校准源仅 {len(cal_present)} 个参与, 无法一致投票, 回退 OR")
        return out

    if len(ng_cal) >= 2:
        out["verdict"] = "NG"          # 双校准源互证
    elif len(ng_cal) == 1 and ng_uncal:
        out["verdict"] = "NG"          # 校准+未校准 跨模态互证
    else:
        out["verdict"] = "REVIEW"      # 孤证 → 黄牌人工复核 (零漏检兜底)
        out["review_required"] = True
        if ng_cal:
            out["review_reasons"].append(
                f"仅单一校准源判NG ({ng_cal[0]}), 无第二源互证")
        if ng_uncal and not ng_cal:
            out["review_reasons"].append(
                f"仅未校准源判NG ({'+'.join(ng_uncal)}), 无校准源支持")
    return out


# ─── 漂移监控 (Stage 3): 校准分数漂移预警 ──────────────────
class DriftMonitor:
    """滑动窗口监控各源"超 NP 阈值比例"漂移。

    判据: 近期窗口内 score>tau 的比例显著高于校准目标 eps
    (> max(factor*eps, eps+floor)) → 返回预警字符串 (调用方记 log)。
    只预警不改判决 — 自适应重训阈值属 v1.5 范围。
    """

    def __init__(self, window: int = 200, min_samples: int = 30,
                 rate_factor: float = 3.0, eps_floor: float = 0.05):
        self.window = max(10, int(window))
        self.min_samples = max(5, int(min_samples))
        self.rate_factor = float(rate_factor)
        self.eps_floor = float(eps_floor)
        self._buf: dict[str, deque] = {}
        self._lock = threading.Lock()

    def observe(self, key: str, score: float, tau: float,
                eps: float) -> Optional[str]:
        """记录一次观测; 触发漂移时返回预警文本, 否则 None。"""
        with self._lock:
            buf = self._buf.setdefault(
                key, deque(maxlen=self.window))
            buf.append((float(score), float(tau), float(eps)))
            if len(buf) < self.min_samples:
                return None
            exceed = sum(1 for s, t, _ in buf if s > t)
            rate = exceed / len(buf)
            eps_ref = max(e for _, _, e in buf)
            limit = max(self.rate_factor * eps_ref,
                        eps_ref + self.eps_floor)
            if rate > limit:
                return (f"[{key}] 分数漂移预警: 近期 {len(buf)} 张 "
                        f"超阈值比例 {rate:.1%} > 上限 {limit:.1%} "
                        f"(目标 eps={eps_ref:.0%}), 建议复检打光/来料 "
                        f"或补登记 OK 样本重校准")
            return None

    def stats(self, key: str) -> Optional[dict]:
        with self._lock:
            buf = self._buf.get(key)
            if not buf:
                return None
            exceed = sum(1 for s, t, _ in buf if s > t)
            return {"n": len(buf), "exceed_rate": exceed / len(buf)}


# 进程级单例 (Union 热路径共享; 测试可直接实例化)
_DRIFT = DriftMonitor()


def get_drift_monitor() -> DriftMonitor:
    return _DRIFT
