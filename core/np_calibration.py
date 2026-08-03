"""Neyman-Pearson 异常分数校准层 (Phase: NP校准)

职责:
- 将任意异常检测器的原始分数 (如 PatchCore 最近邻距离) 校准为
  统计可控的决策阈值与 (0,1) 异常置信度。
- 核心保证: 给定目标误报率 epsilon, 阈值 tau 满足
  P(正常样本分数 > tau) <= epsilon (交换性假设下的有限样本保证,
  即 split-conformal 分位数校准)。

设计取舍 (Simple is best):
- 纯 numpy 实现, 无 scipy/sklearn 新依赖。
- 阈值: 次序统计量分位数 (分布无关, 保证成立)。
- 校准概率: log1p 域高斯拟合 (n>=10 且非退化), 否则经验生存函数兜底。
- 不做 KDE/混合高斯等重武器; 分数分布右偏用 log1p 域高斯已足够。

典型用法:
    calib = NPCalibrator(epsilon=0.02)
    if calib.fit(normal_scores):
        pred = calib.decide(score)          # NP 判定
        conf = calib.anomaly_confidence(s)  # (0,1) 异常置信度
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger("visionocr.np_calib")

_VERSION = 1
_MIN_SAMPLES = 3  # 少于此数无法给出有意义的保证
_SMALL_SAMPLE_N = 100  # 少于此数阈值粒度 (~1/n) 过粗, 需提醒补样本


def _normal_cdf(z: float) -> float:
    """标准正态 CDF (不依赖 scipy)。"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class NPCalibrator:
    """Neyman-Pearson 校准器: 原始异常分数 → 可控阈值 + 校准置信度。"""

    def __init__(self, epsilon: float = 0.02):
        """
        Args:
            epsilon: 目标正常样本误报率上界 (0 < epsilon < 1)。
                     工业含义: 最多允许 epsilon 比例的正常件被判 NG。
        """
        if not (0.0 < epsilon < 1.0):
            raise ValueError(f"epsilon 必须在 (0,1), 得到 {epsilon}")
        self.epsilon = float(epsilon)
        self.threshold: Optional[float] = None
        self.n_samples: int = 0
        # log1p 域高斯参数 (参数化校准概率用)
        self._mu: Optional[float] = None
        self._sigma: Optional[float] = None
        # 经验兜底: 排序后的校准分数
        self._sorted_scores: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self.threshold is not None

    def fit(self, normal_scores: Sequence[float]) -> bool:
        """用正常样本分数拟合 null 分布并计算 NP 阈值。

        Args:
            normal_scores: 正常样本的图像级异常分数 (越大越异常)。

        Returns:
            bool: 拟合是否成功 (样本不足或全为非法值时返回 False)。
        """
        arr = np.asarray(list(normal_scores), dtype=np.float64)
        # 过滤非法值: 非有限 + 负分数 (异常分数语义为非负距离)
        arr = arr[np.isfinite(arr) & (arr >= 0.0)]
        n = arr.size
        if n < _MIN_SAMPLES:
            logger.warning("NP校准失败: 正常样本分数仅 %d 个 (<%d)",
                           n, _MIN_SAMPLES)
            return False

        arr_sorted = np.sort(arr)
        self._sorted_scores = arr_sorted
        self.n_samples = int(n)

        # ── NP 阈值: split-conformal 分位数 (有限样本 FPR 保证) ──
        # 取次序统计量 rank = ceil((1-eps)*(n+1)), 保证
        # P(新正常样本分数 > tau) <= eps (交换性下)。
        rank = int(math.ceil((1.0 - self.epsilon) * (n + 1)))
        rank = min(max(rank, 1), n)
        self.threshold = float(arr_sorted[rank - 1])

        # ── 小样本守卫: 保证仍成立, 但阈值粒度 ~1/n 过粗 ──
        # n 小时阈值被单个次序统计量钉死, eps 的调节实际上以 1/n 为
        # 步长跳变 (如 n=48 时只能取 ~2% 的整数倍), 单库实测误报率
        # 可能明显偏离 eps。不阻断拟合, 仅提醒补充 OK 样本。
        if n < _SMALL_SAMPLE_N:
            logger.warning(
                "NP校准样本偏少: n=%d (<%d), 阈值粒度约 %.1f%% "
                "(一个次序统计量步长≈1/n), 单库实际误报率可能偏离 "
                "eps=%.0f%%; 建议为该产品登记更多 OK 样本。",
                n, _SMALL_SAMPLE_N, 100.0 / n, self.epsilon * 100.0)

        # ── 校准概率: log1p 域高斯拟合 ──
        log_s = np.log1p(arr)
        mu = float(log_s.mean())
        sigma = float(log_s.std(ddof=1)) if n >= 2 else 0.0
        if n >= 10 and sigma > 1e-8:
            self._mu, self._sigma = mu, sigma
        else:
            # 退化/小样本: 仅用经验生存函数
            self._mu, self._sigma = None, None

        logger.info("NP校准完成: n=%d, eps=%.3f, tau=%.4f, "
                    "parametric=%s",
                    n, self.epsilon, self.threshold,
                    self._sigma is not None)
        return True

    def decide(self, score: float) -> bool:
        """NP 决策: 分数超过阈值 → 判为异常 (True)。"""
        if self.threshold is None:
            raise RuntimeError("校准器未拟合, 请先调用 fit()")
        return bool(score > self.threshold)

    def survival(self, score: float) -> float:
        """p 值: P_null(正常样本分数 >= score)。越小越异常。"""
        if not self.is_fitted:
            raise RuntimeError("校准器未拟合, 请先调用 fit()")
        score = max(float(score), 0.0)

        if self._sigma is not None and self._mu is not None:
            z = (math.log1p(score) - self._mu) / self._sigma
            return float(np.clip(1.0 - _normal_cdf(z), 0.0, 1.0))

        # 经验生存函数兜底 (+1 平滑避免端点 0/1)
        s = self._sorted_scores
        n_ge = int(np.sum(s >= score))
        return (n_ge + 1.0) / (s.size + 1.0)

    def anomaly_confidence(self, score: float) -> float:
        """(0,1) 异常置信度 = 1 - p值。越高越异常, 供 UI 展示。"""
        return float(np.clip(1.0 - self.survival(score), 0.0, 1.0))

    # ─── 持久化 ──────────────────────────────────────────────
    def to_dict(self) -> dict:
        if not self.is_fitted:
            return {}
        return {
            "version": _VERSION,
            "epsilon": self.epsilon,
            "threshold": self.threshold,
            "n_samples": self.n_samples,
            "mu": self._mu,
            "sigma": self._sigma,
            "scores_sorted": self._sorted_scores.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Optional["NPCalibrator"]:
        """从字典恢复; 数据非法/版本不符时返回 None (不抛异常)。"""
        if not d or d.get("version") != _VERSION:
            return None
        try:
            cal = cls(epsilon=float(d["epsilon"]))
            cal.threshold = float(d["threshold"])
            cal.n_samples = int(d["n_samples"])
            cal._mu = d.get("mu")
            cal._sigma = d.get("sigma")
            if cal._mu is not None:
                cal._mu = float(cal._mu)
            if cal._sigma is not None:
                cal._sigma = float(cal._sigma)
            cal._sorted_scores = np.asarray(d["scores_sorted"],
                                            dtype=np.float64)
            if cal.n_samples < _MIN_SAMPLES or cal._sorted_scores.size == 0:
                return None
            return cal
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("NP校准器恢复失败: %s", e)
            return None


def recalibrate_engine(engine, normal_scores: Sequence[float],
                       epsilon: Optional[float] = None) -> dict:
    """用新校准分数重拟合引擎的 NP 校准器并更新判定阈值。

    供校准协议 (§6.2 n_cal 扩充) 使用: 引擎建库时仅用尾部 20% holdout
    校准 (n_cal 常为个位数); 补采独立校准图后用本函数重拟合, 阈值统计
    保证不变 (P(正常>tau) ≤ eps), 但粒度随 n 增大而收紧。

    约定: 引擎需具备 _np_epsilon / _np_calibrator / _calibrated_threshold /
    _train_scores 属性 (PatchCore / DINOv2 / SubspaceAD 三引擎一致)。
    本函数不做落盘, 调用方负责 engine.save_bank(...) 持久化。

    Args:
        engine: 已建库的异常检测引擎。
        normal_scores: 新校准集 (独立于 bank 的正常图) 的图像级分数。
        epsilon: 目标误报率上界; None → 沿用引擎当前 _np_epsilon。

    Returns:
        {"ok": bool, "n": int, "epsilon": float, "tau": float|None,
         "tau_before": float|None, "error": str|None}
        拟合失败时不改动引擎现有校准状态。
    """
    eps = float(epsilon) if epsilon is not None else float(
        getattr(engine, "_np_epsilon", 0.02))
    tau_before = getattr(engine, "_calibrated_threshold", None)
    calib = NPCalibrator(epsilon=eps)
    if not calib.fit(normal_scores):
        return {"ok": False, "n": int(np.size(normal_scores)),
                "epsilon": eps, "tau": None, "tau_before": tau_before,
                "error": f"校准样本不足 (需 ≥{_MIN_SAMPLES} 个有效分数)"}
    engine._np_calibrator = calib
    engine._calibrated_threshold = calib.threshold
    engine._train_scores = [float(s) for s in normal_scores]
    logger.info("NP重标定: n_cal=%d, eps=%.3f, tau %.4f → %.4f",
                calib.n_samples, eps,
                tau_before if tau_before is not None else float("nan"),
                calib.threshold)
    return {"ok": True, "n": calib.n_samples, "epsilon": eps,
            "tau": calib.threshold, "tau_before": tau_before,
            "error": None}
