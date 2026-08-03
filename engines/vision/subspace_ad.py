"""SubspaceAD 快速换线异常检测引擎 (v1.4.0)

算法参考: SubspaceAD (CVPR 2026, arXiv 2602.23013), 官方实现
https://github.com/CLendering/SubspaceAD (Apache-2.0, 已核验)。
本文件为基于其公开算法的工程适配独立实现 (ViT-S/448px 消融配置),
不 vendor 原始代码; 依 Apache-2.0 保留署名。

核心管线:
  letterbox(448px) → DINOv2-S/14 patch tokens, 多层(-4,-5)均值聚合
  → PCA 子空间 (成分数按累计解释方差 τ=0.99 自动选择)
  → 逐patch异常分数 = 重构残差平方和 ||x - x_recon||²
  → 图像级分数 = 分数图放大+高斯模糊(σ=4)后 top-1% 像素均值 (mtop1p)

快速换线模式 (1-4 张 OK 图即可上线):
  每张支持图做 30 个随机旋转增广 (seed 可复现) 构成正常特征池,
  旋转空角用图像边缘均值色填充 (KolektorSDD A/B 实测: 黑角填充
  污染子空间, 1-shot AUROC 0.68 vs 边缘填充 0.85);
  其中 ~10% 增广视图留出不入子空间, 作为 NP 校准样本
  (k=1 时 n_cal=3, 保证校准器可拟合)。

边界 (诚实声明):
- 对逻辑/结构异常弱 (论文作者自述), 仅测过 MVTec/VisA,
  真实产线迁移性未经验证 → 定位为快速换线/降级通道,
  不替代 PatchCore 主判, 不参与 Union OR (避免抬升联合误报率)。
- 分数跨产品不可比 (逐产品建子空间), 判定依赖 NP 校准阈值。
- 快速模式自校准系统性偏乐观: 校准样本是支持图自身旋转视图,
  残差低于真实新图 (KolektorSDD 实测 tau≈0.14 vs 正常件均值≈0.53),
  部署阈值会过度报警 → 快速模式判定必须人工复核,
  累积 ≥10 张真实 OK 图后应切换标准建库模式重校准。

API 与 AnomalibEngine/DINOv2AnomalyEngine 对齐
(train/infer/save_bank/load_bank/has_bank)。显存 ~1GB (ViT-S FP32)。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from engines.base import BaseEngine, EngineMeta, EngineState

logger = logging.getLogger("visionocr.subspace_ad")

_DEFAULT_MODEL_ID = "facebook/dinov2-small"
_DEFAULT_INPUT_SIZE = 448   # 官方 ViT-S 消融配置 (32x32 patch grid)
_DEFAULT_LAYERS = (-4, -5)  # 官方 ViT-S 层选择 (深度 55%~70% 处中间层)
_DEFAULT_PCA_EV = 0.99      # 累计解释方差截断 (官方 τ)
_DEFAULT_AUG_COUNT = 30     # 快速模式每张支持图旋转增广数 (官方 aug_count)
_DEFAULT_FAST_MAX = 4       # ≤ 此张数触发快速换线模式
_DEFAULT_CAL_FRAC = 0.10    # 增广视图留出作 NP 校准的比例
_DEFAULT_BLUR_SIGMA = 4.0   # 分数图高斯模糊 σ (官方)
_DEFAULT_TOP_FRAC = 0.01    # mtop1p 尾部聚合比例 (官方 ρ=1%)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SubspaceADEngine(BaseEngine):
    """DINOv2 特征 + PCA 子空间重构残差的免训练少样本异常检测。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self._model = None
        self._sub_mu: Optional[np.ndarray] = None       # (d,) 特征均值
        self._sub_components: Optional[np.ndarray] = None  # (k, d) 主成分
        self._sub_eigvals: Optional[np.ndarray] = None
        self._bank_meta: dict = {}
        self._np_calibrator = None
        self._train_scores: list[float] = []
        self._calibrated_threshold: Optional[float] = None
        self._last_valid_region: Optional[dict] = None

        qc_cfg = config.get("qc", {}) or {}
        sa_cfg = qc_cfg.get("subspacead", {}) or {}
        self._model_id: str = sa_cfg.get("model_id", _DEFAULT_MODEL_ID)
        self._img_size: int = sa_cfg.get("input_size", _DEFAULT_INPUT_SIZE)
        if self._img_size % 14 != 0:
            self._img_size = (self._img_size // 14) * 14 or 14
        self._layers: list[int] = list(sa_cfg.get("layers", _DEFAULT_LAYERS))
        self._pca_ev: float = float(sa_cfg.get("pca_ev", _DEFAULT_PCA_EV))
        self._aug_count: int = int(sa_cfg.get("aug_count",
                                              _DEFAULT_AUG_COUNT))
        self._fast_max: int = int(sa_cfg.get("fast_max_images",
                                             _DEFAULT_FAST_MAX))
        self._cal_frac: float = float(sa_cfg.get("cal_frac",
                                                 _DEFAULT_CAL_FRAC))
        self._blur_sigma: float = float(sa_cfg.get("blur_sigma",
                                                   _DEFAULT_BLUR_SIGMA))
        self._top_frac: float = float(sa_cfg.get("top_frac",
                                                 _DEFAULT_TOP_FRAC))
        self._np_epsilon: float = float(sa_cfg.get("np_epsilon", 0.02))
        self._seed: int = int(sa_cfg.get("seed", 42))

    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="subspace_ad",
            display_name="SubspaceAD 快速换线 (1-4图建库)",
            category="vision",
            vram_gb=1.0,
            license="Apache-2.0",
            description="DINOv2-S 特征 + PCA 子空间重构残差 (免训练, "
                        "1-4 张 OK 图极速换线; 降级通道, 不替代主判)",
            tags=["缺陷检测", "异常检测", "少样本", "快速换线", "SubspaceAD"],
            resident=False,
        )

    def load(self) -> None:
        """加载 DINOv2-S/14 (HF 缓存优先离线, FP32, 输出全部隐层)。"""
        try:
            import torch
            from transformers import AutoModel
        except ImportError as e:
            logger.error("依赖缺失: %s (需要 torch + transformers)", e)
            self.state = EngineState.ERROR
            return

        try:
            device = self._device()
            # 两段式加载 (P0-5 同款): 缓存命中全离线, 缺失回退联网
            try:
                model = AutoModel.from_pretrained(
                    self._model_id, output_hidden_states=True,
                    local_files_only=True)
                source = "本地缓存(离线)"
            except Exception as cache_err:
                logger.info("SubspaceAD 骨干本地缓存缺失 (%s), "
                            "回退联网下载...", cache_err)
                model = AutoModel.from_pretrained(
                    self._model_id, output_hidden_states=True)
                source = "网络下载"
            model.eval()
            model.to(device)
            self._model = model
            self._n_skip = 1 + int(getattr(
                model.config, "num_register_tokens", 0) or 0)
            self.state = EngineState.READY
            logger.info("SubspaceAD 就绪 (%s, %s, %dpx, layers=%s, "
                        "pca_ev=%.2f, 来源: %s)",
                        device, self._model_id, self._img_size,
                        self._layers, self._pca_ev, source)
        except Exception as e:
            logger.error("SubspaceAD 骨干加载失败: %s", e)
            self.state = EngineState.ERROR

    # ─── 训练 ───────────────────────────────────────────────
    def train(self, image_paths: list[str], **kwargs) -> dict:
        """从 OK 样本拟合 PCA 子空间 + NP 校准。

        快速换线模式 (n ≤ fast_max_images): 每张图旋转增广
        aug_count 次, ~cal_frac 增广视图留出作校准。
        常规模式 (n ≥ 5): 不增广; n ≥ 10 时留 20% 校准。

        Returns:
            {"n_images", "pca_k", ...} 或 {"error"}
        """
        if not self.is_ready():
            return {"error": "模型未加载"}
        if not image_paths:
            return {"error": "无有效图像"}

        aug_count = int(kwargs.get("aug_count", self._aug_count))
        n_total = len(image_paths)
        fast_mode = n_total <= self._fast_max
        rng = np.random.default_rng(self._seed)

        bank_feats: list[np.ndarray] = []
        cal_paths: list[Any] = []   # 路径 (常规) 或 PIL 图 (快速留出)
        n_ok = 0
        n_aug = 0

        if fast_mode:
            cal_per_img = max(1, int(aug_count * self._cal_frac)) \
                if aug_count > 0 else 0
            for path in image_paths:
                pil = self._open_pil(path)
                if pil is None:
                    continue
                n_ok += 1
                feat = self._extract_features(pil)
                if feat is not None:
                    bank_feats.append(feat)
                if aug_count > 0:
                    views = self._rotated_views(pil, aug_count, rng)
                    # 尾部 cal_per_img 个视图留出校准, 不入子空间
                    for v in views[:aug_count - cal_per_img]:
                        f = self._extract_features(v)
                        if f is not None:
                            bank_feats.append(f)
                            n_aug += 1
                    cal_paths.extend(views[aug_count - cal_per_img:])
            logger.info("SubspaceAD 快速换线模式: %d 张支持图 + %d 增广视图, "
                        "%d 增广视图留作校准", n_ok, n_aug, len(cal_paths))
        else:
            if n_total >= 10:
                n_cal = max(3, n_total // 5)
                bank_paths = image_paths[:-n_cal]
                cal_paths = list(image_paths[-n_cal:])
            else:
                bank_paths = cal_paths = list(image_paths)
            for path in bank_paths:
                feat = self._extract_features(path)
                if feat is not None:
                    bank_feats.append(feat)
                    n_ok += 1

        if not bank_feats:
            return {"error": "无有效图像"}
        X = np.concatenate(bank_feats, axis=0)
        logger.info("SubspaceAD 特征池: %d 张图, %d patches",
                    n_ok, X.shape[0])

        # ── PCA 子空间: SVD, 累计解释方差 τ 截断 ──
        mu = X.mean(axis=0)
        Xc = X - mu
        # eigh(协方差) 对 d=384 足够快且数值稳定
        cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = eigvecs[:, order]
        total = eigvals.sum()
        if total <= 1e-12:
            return {"error": "特征退化 (方差为零), 无法建子空间"}
        cum = np.cumsum(eigvals) / total
        k = int(np.searchsorted(cum, self._pca_ev) + 1)
        k = max(1, min(k, eigvals.shape[0]))
        self._sub_mu = mu.astype(np.float32)
        self._sub_components = eigvecs[:, :k].T.astype(np.float32)
        self._sub_eigvals = eigvals[:k].astype(np.float32)
        logger.info("PCA 子空间: k=%d 维 (累计解释方差 %.4f ≥ τ=%.2f)",
                    k, float(cum[k - 1]), self._pca_ev)

        self._bank_meta = {
            "n_images": n_ok,
            "n_augmented": int(n_aug),
            "n_patches": int(X.shape[0]),
            "pca_k": int(k),
            "pca_ev_achieved": round(float(cum[k - 1]), 4),
            "img_size": self._img_size,
            "layers": ",".join(str(l) for l in self._layers),
            "mode": "fast" if fast_mode else "standard",
        }

        # ── NP 校准: 校准样本图像级分数 ──
        cal_scores = []
        for item in cal_paths:
            s = self._image_score(item)
            if s is not None:
                cal_scores.append(s)
        self._fit_np_calibrator(cal_scores, fast_mode)

        self._bank_meta["threshold"] = float(
            self._calibrated_threshold or 0.0)
        return dict(self._bank_meta)

    def _fit_np_calibrator(self, cal_scores: list[float],
                           fast_mode: bool) -> None:
        """NP 校准, 失败降级 P99×1.2 启发式 (与兄弟引擎一致)。"""
        self._np_calibrator = None
        self._train_scores = cal_scores
        if not cal_scores:
            self._calibrated_threshold = None
            if fast_mode:
                logger.warning("快速换线无校准样本 (aug_count=0 且支持图<3): "
                               "阈值不可用, 仅输出分数, 判定恒 OK")
            return
        from core.np_calibration import NPCalibrator
        calib = NPCalibrator(epsilon=self._np_epsilon)
        if calib.fit(cal_scores):
            self._np_calibrator = calib
            self._calibrated_threshold = calib.threshold
            logger.info("SubspaceAD NP校准: eps=%.3f, tau=%.4f, n_cal=%d",
                        self._np_epsilon, self._calibrated_threshold,
                        len(cal_scores))
        else:
            p99 = float(np.percentile(np.asarray(cal_scores), 99))
            self._calibrated_threshold = p99 * 1.2
            logger.info("SubspaceAD 阈值校准(legacy fallback): P99=%.4f, "
                        "tau=%.4f, n_cal=%d", p99,
                        self._calibrated_threshold, len(cal_scores))
        if fast_mode:
            logger.warning("快速换线阈值基于增广视图自评, 偏乐观; "
                           "建议 ε≥0.10 并保留人工复核, 尽快补足 OK 样本 "
                           "切换全量建库")

    # ─── 推理 ───────────────────────────────────────────────
    def infer(self, image: Any, **kwargs) -> dict:
        """检测单张图像, 返回残差分数 + 热力图 (与兄弟引擎契约对齐)。"""
        if not self.is_ready():
            return {"score": 0, "anomaly_map": None, "pred_label": "ERROR",
                    "error": "模型未加载"}
        if self._sub_components is None or self._sub_mu is None:
            return {"score": 0, "anomaly_map": None, "pred_label": "ERROR",
                    "error": "特征库为空, 请先注册 OK 样本"}

        res = self._patch_residuals(image)
        if res is None:
            return {"score": 0, "anomaly_map": None, "pred_label": "ERROR",
                    "error": "无法读取图像"}

        anomaly_map, grid_size = self._residuals_to_map(res)
        m0, m1 = anomaly_map.min(), anomaly_map.max()
        if m1 > m0:
            vis_map = (anomaly_map - m0) / (m1 - m0)
        else:
            vis_map = anomaly_map - m0

        score = self._image_score_from_residuals(res)

        threshold = kwargs.get("threshold", None)
        if threshold is None:
            if self._calibrated_threshold is not None:
                threshold = self._calibrated_threshold
            else:
                threshold = float("inf")  # 无校准: 不误判, 仅输出分数

        # 快速换线模式: 自校准基于支持图自身旋转视图, 系统性偏乐观
        # (KolektorSDD 实测 tau≈0.14 vs 真实正常件分数均值≈0.53),
        # 自主判定会退化为全 NG → 不给判定, 仅输出分数+热力图供人工复核
        fast_mode = self._bank_meta.get("mode") == "fast"
        if fast_mode:
            pred = "REVIEW"
        else:
            pred = "NG" if score > threshold else "OK"
        result = {
            "score": round(score, 4),
            "anomaly_map": vis_map,
            "pred_label": pred,
            "grid_size": grid_size,
            "threshold_used": round(float(threshold), 4)
            if np.isfinite(threshold) else None,
        }
        if fast_mode:
            result["review_required"] = True
            result["calibration_mode"] = "fast_selfcal"
        if self._np_calibrator is not None and self._np_calibrator.is_fitted:
            result["calibrated_score"] = round(
                self._np_calibrator.anomaly_confidence(score), 4)
            result["np_p_value"] = round(
                self._np_calibrator.survival(score), 6)
        return result

    def _image_score(self, image: Any) -> Optional[float]:
        """图像级分数 (训练校准用, mtop1p)。"""
        res = self._patch_residuals(image)
        if res is None:
            return None
        return self._image_score_from_residuals(res)

    def _image_score_from_residuals(self, res: np.ndarray) -> float:
        """patch 残差 → 放大到输入分辨率 → 高斯模糊 → top-1% 均值。"""
        import cv2
        grid = self._img_size // 14
        gh, gw = self._residual_grid_shape(res)
        patch_map = res.reshape(gh, gw).astype(np.float32)
        full = cv2.resize(patch_map, (self._img_size, self._img_size),
                          interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(full, (0, 0), self._blur_sigma)
        flat = blurred.ravel()
        n_top = max(1, int(np.ceil(self._top_frac * flat.size)))
        return float(np.sort(flat)[-n_top:].mean())

    def _residual_grid_shape(self, res: np.ndarray) -> tuple[int, int]:
        grid = self._img_size // 14
        vr = self._last_valid_region
        if vr and (vr["pad_top"] > 0 or vr["pad_left"] > 0):
            gw = max(1, int(vr["new_w"] * grid / self._img_size))
            gh = max(1, int(vr["new_h"] * grid / self._img_size))
            if gh * gw == res.shape[0]:
                return gh, gw
        if res.shape[0] == grid * grid:
            return grid, grid
        gs = int(np.sqrt(res.shape[0]))
        if gs * gs == res.shape[0]:
            return gs, gs
        return 1, res.shape[0]

    def _residuals_to_map(self, res: np.ndarray) -> tuple[np.ndarray, int]:
        gh, gw = self._residual_grid_shape(res)
        return res.reshape(gh, gw), self._img_size // 14

    def _patch_residuals(self, image: Any) -> Optional[np.ndarray]:
        """逐 patch 重构残差平方和: ||x-μ||² - ||投影坐标||²。"""
        feat = self._extract_features(image)
        if feat is None:
            return None
        X0 = feat.astype(np.float32) - self._sub_mu
        Z = X0 @ self._sub_components.T
        res = (X0 * X0).sum(axis=1) - (Z * Z).sum(axis=1)
        return np.clip(res, 0.0, None)

    # ─── 增广与图像读取 ────────────────────────────────────
    def _open_pil(self, image: Any):
        from PIL import Image
        if isinstance(image, (str, Path)):
            try:
                return Image.open(str(image)).convert("RGB")
            except Exception:
                return None
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]  # BGR→RGB
            return Image.fromarray(image)
        if hasattr(image, "convert"):
            return image.convert("RGB")
        return None

    @staticmethod
    def _edge_mean_fillcolor(pil):
        """图像边缘均值填充色 (替代 fillcolor=0)。

        诊断结论 (KolektorSDD A/B): 黑色旋转角会污染 PCA 子空间并
        抬高 mtop1p 背景分, 1-shot AUROC 0.68 vs 边缘填充 0.85。
        """
        arr = np.asarray(pil)
        if arr.ndim == 2:
            border = np.concatenate([arr[0, :], arr[-1, :],
                                     arr[:, 0], arr[:, -1]])
            return int(round(float(border.mean())))
        border = np.concatenate([arr[0, :, :].reshape(-1, arr.shape[2]),
                                 arr[-1, :, :].reshape(-1, arr.shape[2]),
                                 arr[:, 0, :].reshape(-1, arr.shape[2]),
                                 arr[:, -1, :].reshape(-1, arr.shape[2])],
                                axis=0)
        return tuple(int(round(float(c))) for c in border.mean(axis=0))

    def _rotated_views(self, pil, n: int, rng) -> list:
        """随机旋转增广 (官方 1-shot 关键机制; 边缘均值填充防黑角污染)。"""
        from PIL import Image
        fill = self._edge_mean_fillcolor(pil)
        views = []
        for _ in range(n):
            angle = float(rng.uniform(0.0, 345.0))
            views.append(pil.rotate(angle, resample=Image.BILINEAR,
                                    fillcolor=fill))
        return views

    # ─── 特征提取 ───────────────────────────────────────────
    def _extract_features(self, image: Any) -> Optional[np.ndarray]:
        """letterbox 预处理 → 多层 patch tokens 均值 (有效区域)。

        Returns: (n_valid_patches, 384) ndarray
        """
        import torch

        pil = image if hasattr(image, "rotate") else self._open_pil(image)
        if pil is None:
            return None

        w, h = pil.size
        size = self._img_size
        scale = size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        new_w = max(1, (new_w // 14) * 14 or 14)
        new_h = max(1, (new_h // 14) * 14 or 14)
        resized = pil.resize((new_w, new_h))

        # 边缘复制填充 (与兄弟引擎一致, 避免灰色填充假异常)
        arr = np.array(resized)
        pad_top = (size - new_h) // 2
        pad_left = (size - new_w) // 2
        full = np.zeros((size, size, 3), dtype=np.uint8)
        full[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = arr
        if pad_top > 0:
            full[:pad_top, pad_left:pad_left + new_w] = \
                arr[0:1, :, :].repeat(pad_top, axis=0)
            full[pad_top + new_h:, pad_left:pad_left + new_w] = \
                arr[-1:, :, :].repeat(size - pad_top - new_h, axis=0)
        if pad_left > 0:
            full[:, :pad_left] = \
                full[:, pad_left:pad_left + 1].repeat(pad_left, axis=1)
            full[:, pad_left + new_w:] = \
                full[:, pad_left + new_w - 1:pad_left + new_w].repeat(
                    size - pad_left - new_w, axis=1)

        self._last_valid_region = {
            "pad_top": pad_top, "pad_left": pad_left,
            "new_h": new_h, "new_w": new_w,
            "orig_h": h, "orig_w": w,
        }

        x = (torch.from_numpy(full).permute(2, 0, 1).float() / 255.0
             - torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)) \
            / torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        x = x.unsqueeze(0).to(self._device())

        try:
            with torch.no_grad():
                out = self._model(pixel_values=x)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and x.device.type == "cuda":
                logger.warning("GPU 显存不足, 降级 CPU 推理 (本次)")
                import torch as _t
                _t.cuda.empty_cache()
                self._model.to("cpu")
                with torch.no_grad():
                    out = self._model(pixel_values=x.to("cpu"))
                self._model.to(self._device())
            else:
                raise

        # 多层均值聚合 (官方 agg_method=mean)
        hidden = out.hidden_states
        layers = [hidden[l] for l in self._layers]
        tokens = torch.stack(layers, dim=0).mean(dim=0)
        tokens = tokens[0, self._n_skip:].cpu().numpy()
        return self._crop_to_valid(tokens)

    def _crop_to_valid(self, tokens: np.ndarray) -> np.ndarray:
        """裁切到有效区域 (排除 letterbox 填充 patch)。"""
        vr = self._last_valid_region
        grid = self._img_size // 14
        if tokens.shape[0] != grid * grid or vr is None:
            return tokens
        if vr["pad_top"] == 0 and vr["pad_left"] == 0 \
                and vr["new_h"] == self._img_size \
                and vr["new_w"] == self._img_size:
            return tokens
        gt = int(vr["pad_top"] * grid / self._img_size)
        gl = int(vr["pad_left"] * grid / self._img_size)
        gh = max(1, int(vr["new_h"] * grid / self._img_size))
        gw = max(1, int(vr["new_w"] * grid / self._img_size))
        gt, gl = min(gt, grid - 1), min(gl, grid - 1)
        gh, gw = min(gh, grid - gt), min(gw, grid - gl)
        m = tokens.reshape(grid, grid, -1)
        return m[gt:gt + gh, gl:gl + gw, :].reshape(-1, tokens.shape[1])

    # ─── 持久化 ─────────────────────────────────────────────
    def save_bank(self, path: str | Path, product_name: str = "") -> None:
        """保存子空间 + NP 校准到 .npz。"""
        if self._sub_components is None or self._sub_mu is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "product_name": product_name,
            "sub_mu": self._sub_mu,
            "sub_components": self._sub_components,
            "sub_eigvals": self._sub_eigvals,
            **{f"meta_{k}": v for k, v in self._bank_meta.items()},
        }
        if self._calibrated_threshold is not None:
            save_kwargs["calibrated_threshold"] = self._calibrated_threshold
        if self._np_calibrator is not None and self._np_calibrator.is_fitted:
            import json
            save_kwargs["np_calib_json"] = json.dumps(
                self._np_calibrator.to_dict(), ensure_ascii=False)
        np.savez_compressed(str(path), **save_kwargs)
        logger.info("SubspaceAD 特征库已保存: %s (k=%d, threshold=%s)",
                    path, self._sub_components.shape[0],
                    f"{self._calibrated_threshold:.4f}"
                    if self._calibrated_threshold else "N/A")

    def load_bank(self, path: str | Path) -> bool:
        """从 .npz 恢复子空间 + NP 校准。"""
        path = Path(path)
        if not path.exists():
            logger.warning("SubspaceAD 特征库不存在: %s", path)
            return False
        try:
            data = np.load(str(path), allow_pickle=True)
            self._sub_mu = data["sub_mu"].astype(np.float32)
            self._sub_components = data["sub_components"].astype(np.float32)
            self._sub_eigvals = data["sub_eigvals"].astype(np.float32) \
                if "sub_eigvals" in data.files else None
            self._bank_meta = {
                k.replace("meta_", ""): data[k].item()
                for k in data.files if k.startswith("meta_")
            }
            self._calibrated_threshold = (
                float(data["calibrated_threshold"])
                if "calibrated_threshold" in data.files else None)
            self._np_calibrator = None
            if "np_calib_json" in data.files:
                import json
                from core.np_calibration import NPCalibrator
                try:
                    self._np_calibrator = NPCalibrator.from_dict(
                        json.loads(str(data["np_calib_json"])))
                except Exception as e:
                    logger.warning("NP校准器解析失败, 忽略: %s", e)
            logger.info("SubspaceAD 特征库已加载: %s (k=%d, threshold=%s)",
                        path.name, self._sub_components.shape[0],
                        f"{self._calibrated_threshold:.4f}"
                        if self._calibrated_threshold else "N/A")
            return True
        except Exception as e:
            logger.error("加载 SubspaceAD 特征库失败: %s", e)
            return False

    @property
    def has_bank(self) -> bool:
        return self._sub_components is not None \
            and self._sub_mu is not None

    def unload(self) -> None:
        self._model = None
        self._sub_mu = None
        self._sub_components = None
        self._sub_eigvals = None
        self._bank_meta = {}
        self._np_calibrator = None
        self._train_scores = []
        self._calibrated_threshold = None
        self.state = EngineState.UNLOADED
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ─── 内部 ───────────────────────────────────────────────
    def _device(self) -> str:
        import torch
        cfg = self.config.get("device", "auto")
        if cfg == "cuda" or (cfg == "auto" and torch.cuda.is_available()):
            return "cuda"
        return "cpu"
