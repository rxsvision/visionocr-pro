"""DINOv2 少样本异常检测引擎 (技术吸收 Phase 2)

借鉴 PixOOD 的分布建模思想 (许可证: CC-BY-NC-SA 4.0 + Toyota 专利,
仅借鉴算法思想, 本实现为完全自研, 无代码/权重继承):
- 多 etalon 分布建模: 用 GMM (对角协方差) 拟合正常样本特征分布,
  每个高斯分量即一个 "etalon" (正常模式原型)
- 异常分数 = 负对数似然 (NLL): 偏离正常分布越远分数越高
- NP 校准: 复用 core.np_calibration, 误报率有有限样本统计保证

骨干: DINOv2-S/14 (facebook/dinov2-small, 代码+权重均 Apache-2.0,
已联网核实)。自监督 ViT 特征对表面缺陷的判别力显著强于 ImageNet
监督特征 (AnomalyDINO, CVPR'24 结论), 与 PatchCore(WRN50) 特征空间
互补, 作为 Union 零漏检架构的第 4 检测源。

管线: letterbox(518px, 14 倍数) → ViT patch tokens(37x37, 384d)
      → 有效区域裁切 → PCA 白化降维(64d) → GMM(8 etalons)
      → 逐patch NLL → top-k 均值 → NP 阈值判定

API 与 AnomalibEngine 对齐 (train/infer/save_bank/load_bank/has_bank),
便于 bank 管理与 UI 复用。

显存: ~1GB (ViT-S FP32 推理)。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from engines.base import BaseEngine, EngineMeta, EngineState

logger = logging.getLogger("visionocr.dinov2")

_DEFAULT_MODEL_ID = "facebook/dinov2-small"
_DEFAULT_INPUT_SIZE = 518   # 14 的倍数, DINOv2 标准分辨率
_DEFAULT_PCA_DIM = 64       # 白化降维目标维度 (GMM 对角协方差稳定性)
_DEFAULT_N_ETALONS = 8      # GMM 分量数 (正常模式原型数)
_MAX_GMM_SAMPLES = 50000    # GMM 拟合样本上限 (控耗时)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2AnomalyEngine(BaseEngine):
    """DINOv2 特征 + GMM 分布建模的少样本异常检测。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self._model = None
        self._pca = None          # sklearn PCA (whiten=True)
        self._gmm = None          # sklearn GaussianMixture (diag)
        self._bank_meta: dict = {}
        self._np_calibrator = None
        self._train_scores: list[float] = []
        self._calibrated_threshold: Optional[float] = None
        # §5.1 per-etalon 局部 NP 归一化统计 (建库时拟合, npz 持久化)
        self._etalon_np_mu: Optional[np.ndarray] = None     # (K,)
        self._etalon_np_sigma: Optional[np.ndarray] = None  # (K,)

        qc_cfg = config.get("qc", {}) or {}
        dv_cfg = qc_cfg.get("dinov2", {}) or {}
        self._model_id: str = dv_cfg.get("model_id", _DEFAULT_MODEL_ID)
        self._img_size: int = dv_cfg.get("input_size", _DEFAULT_INPUT_SIZE)
        if self._img_size % 14 != 0:
            self._img_size = (self._img_size // 14) * 14 or 14
        self._pca_dim: int = dv_cfg.get("pca_dim", _DEFAULT_PCA_DIM)
        self._n_etalons: int = dv_cfg.get("n_etalons", _DEFAULT_N_ETALONS)
        self._np_epsilon: float = float(dv_cfg.get("np_epsilon", 0.02))
        # §5.1 PixOOD 思想借鉴 (自研实现, A/B 验收后决定默认值)
        self._reinit_dead: bool = bool(dv_cfg.get("reinit_dead_etalons", False))
        self._reinit_rounds: int = int(dv_cfg.get("reinit_rounds", 2))
        self._dead_weight_frac: float = float(
            dv_cfg.get("dead_weight_frac", 0.5))
        self._per_etalon_np: bool = bool(dv_cfg.get("per_etalon_np", False))

    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="dinov2_anomaly",
            display_name="DINOv2 异常检测 (少样本)",
            category="vision",
            vram_gb=1.0,
            license="Apache-2.0",
            description="DINOv2-S/14 特征 + GMM 多原型分布建模 + NP 校准 "
                        "(与 PatchCore 互补的 Union 第4源)",
            tags=["缺陷检测", "异常检测", "少样本", "DINOv2", "零漏检"],
            resident=True,
        )

    def load(self) -> None:
        """加载 DINOv2-S/14 (HF 缓存优先, FP32)。"""
        try:
            import torch
            from transformers import AutoModel
        except ImportError as e:
            logger.error("依赖缺失: %s (需要 torch + transformers)", e)
            self.state = EngineState.ERROR
            return

        try:
            device = self._device()
            # FP32 加载 (transformers 5.x FP16 路径已知问题, 且 ViT-S 显存充裕)
            model = AutoModel.from_pretrained(self._model_id)
            model.eval()
            model.to(device)
            self._model = model
            self._n_skip = 1 + int(getattr(
                model.config, "num_register_tokens", 0) or 0)
            self.state = EngineState.READY
            logger.info("DINOv2 异常检测就绪 (%s, %s, %dpx, pca=%d, "
                        "etalons=%d)", device, self._model_id,
                        self._img_size, self._pca_dim, self._n_etalons)
        except Exception as e:
            logger.error("DINOv2 加载失败: %s", e)
            self.state = EngineState.ERROR

    # ─── 训练 ───────────────────────────────────────────────
    def train(self, image_paths: list[str], **kwargs) -> dict:
        """从 OK 样本拟合 PCA + GMM + NP 校准。

        Args:
            image_paths: OK 样本路径 (建议 10~30 张)

        Returns:
            {"n_images", "n_components", "pca_dim", "n_etalons"} 或 {"error"}
        """
        if not self.is_ready():
            return {"error": "模型未加载"}
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture

        # 留出 20% 校准集 (与 PatchCore 引擎一致的策略)
        n_total = len(image_paths)
        if n_total >= 10:
            n_cal = max(3, n_total // 5)
            bank_paths = image_paths[:-n_cal]
            cal_paths = image_paths[-n_cal:]
        else:
            bank_paths = cal_paths = list(image_paths)

        # 逐图提取有效区域 patch 特征
        bank_feats = []
        n_ok = 0
        for path in bank_paths:
            feat = self._extract_features(path)
            if feat is not None:
                bank_feats.append(feat)
                n_ok += 1
        if not bank_feats:
            return {"error": "无有效图像"}
        X = np.concatenate(bank_feats, axis=0).astype(np.float32)
        logger.info("DINOv2 特征: %d 张图, %d patches", n_ok, X.shape[0])

        # PCA 白化降维
        pca_dim = min(self._pca_dim, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=pca_dim, whiten=True)
        Xw = pca.fit_transform(X)

        # GMM 多 etalon 拟合 (样本上限控制)
        if Xw.shape[0] > _MAX_GMM_SAMPLES:
            rng = np.random.default_rng(2026)
            idx = rng.choice(Xw.shape[0], _MAX_GMM_SAMPLES, replace=False)
            Xw_fit = Xw[idx]
        else:
            Xw_fit = Xw
        n_etalons = max(1, min(self._n_etalons, Xw_fit.shape[0] // 10))
        gmm = GaussianMixture(n_components=n_etalons,
                              covariance_type="diag",
                              reg_covar=1e-4, random_state=2026,
                              n_init=1, max_iter=200)
        gmm.fit(Xw_fit)
        logger.info("GMM 拟合完成: %d etalons, %d 样本, %d 轮",
                    n_etalons, Xw_fit.shape[0], gmm.n_iter_)

        # §5.1 P1 借鉴: 死 etalon 重初始化 (欠表达正常模式补采)
        if self._reinit_dead and n_etalons > 1:
            gmm = self._reinit_dead_etalons(Xw_fit, gmm)

        self._pca = pca
        self._gmm = gmm
        # §5.1 P4 借鉴: per-etalon 局部 NP 归一化统计
        self._etalon_np_mu = self._etalon_np_sigma = None
        if self._per_etalon_np:
            self._fit_etalon_np_stats(Xw_fit)
        self._bank_meta = {
            "n_images": n_ok,
            "n_patches": int(X.shape[0]),
            "pca_dim": int(pca_dim),
            "n_etalons": int(n_etalons),
            "img_size": self._img_size,
            "reinit_dead": bool(self._reinit_dead),
            "per_etalon_np": bool(self._per_etalon_np
                                  and self._etalon_np_mu is not None),
        }

        # NP 校准: 留出正常样本的图像级分数
        cal_scores = []
        for path in cal_paths:
            s = self._image_score(path)
            if s is not None:
                cal_scores.append(s)
        self._fit_np_calibrator(cal_scores)

        self._bank_meta["threshold"] = float(
            self._calibrated_threshold or 0.0)
        return dict(self._bank_meta)

    def _fit_np_calibrator(self, cal_scores: list[float]) -> None:
        """NP 校准拟合, 失败时降级 P99×margin 启发式 (与 PatchCore 对齐)。"""
        self._np_calibrator = None
        self._train_scores = cal_scores
        if not cal_scores:
            self._calibrated_threshold = None
            return
        from core.np_calibration import NPCalibrator
        calib = NPCalibrator(epsilon=self._np_epsilon)
        if calib.fit(cal_scores):
            self._np_calibrator = calib
            self._calibrated_threshold = calib.threshold
            logger.info("NP阈值校准: eps=%.3f, tau=%.4f, n_cal=%d",
                        self._np_epsilon, self._calibrated_threshold,
                        len(cal_scores))
        else:
            p99 = float(np.percentile(np.asarray(cal_scores), 99))
            self._calibrated_threshold = p99 * 1.2
            logger.info("阈值校准(legacy fallback): P99=%.4f, tau=%.4f, "
                        "n_cal=%d", p99, self._calibrated_threshold,
                        len(cal_scores))

    # ─── 推理 ───────────────────────────────────────────────
    def infer(self, image: Any, **kwargs) -> dict:
        """检测单张图像, 返回异常分数 + 热力图 (与 AnomalibEngine 对齐)。"""
        if not self.is_ready():
            return {"score": 0, "anomaly_map": None, "pred_label": "ERROR",
                    "error": "模型未加载"}
        if self._gmm is None or self._pca is None:
            return {"score": 0, "anomaly_map": None, "pred_label": "ERROR",
                    "error": "特征库为空, 请先注册 OK 样本"}

        nll = self._patch_nll(image)
        if nll is None:
            return {"score": 0, "anomaly_map": None, "pred_label": "ERROR",
                    "error": "无法读取图像"}

        # 空间重排 → anomaly_map (有效区域 grid)
        anomaly_map, grid_size = self._nll_to_map(nll)

        # 归一化 0~1 (可视化)
        m0, m1 = anomaly_map.min(), anomaly_map.max()
        if m1 > m0:
            anomaly_map = (anomaly_map - m0) / (m1 - m0)

        # 图像级分数: top-k patch NLL 均值 (与 PatchCore 一致的聚合)
        k = max(1, len(nll) // 20)
        score = float(np.sort(nll)[-k:].mean())

        threshold = kwargs.get("threshold", None)
        if threshold is None:
            if self._calibrated_threshold is not None:
                threshold = self._calibrated_threshold
            else:
                threshold = 0.0  # 无校准: 不误判, 仅输出分数

        pred = "NG" if score > threshold else "OK"
        result = {
            "score": round(score, 4),
            "anomaly_map": anomaly_map,
            "pred_label": pred,
            "grid_size": grid_size,
            "threshold_used": round(float(threshold), 4),
        }
        if self._np_calibrator is not None and self._np_calibrator.is_fitted:
            result["calibrated_score"] = round(
                self._np_calibrator.anomaly_confidence(score), 4)
            result["np_p_value"] = round(
                self._np_calibrator.survival(score), 6)
        return result

    def _image_score(self, image: Any) -> Optional[float]:
        """图像级分数 (训练校准用, 无判定)。"""
        nll = self._patch_nll(image)
        if nll is None:
            return None
        k = max(1, len(nll) // 20)
        return float(np.sort(nll)[-k:].mean())

    def _patch_nll(self, image: Any) -> Optional[np.ndarray]:
        """逐 patch 负对数似然。"""
        feat = self._extract_features(image)
        if feat is None:
            return None
        Xw = self._pca.transform(feat.astype(np.float32))
        loglik = self._gmm.score_samples(Xw)
        return -loglik

    def _nll_to_map(self, nll: np.ndarray) -> tuple[np.ndarray, int]:
        grid_total = self._img_size // 14
        vr = getattr(self, "_last_valid_region", None)
        if vr and (vr["pad_top"] > 0 or vr["pad_left"] > 0):
            gw = max(1, int(vr["new_w"] * grid_total / self._img_size))
            gh = max(1, int(vr["new_h"] * grid_total / self._img_size))
        else:
            gw = gh = grid_total
        if gw * gh == nll.shape[0]:
            return nll.reshape(gh, gw), grid_total
        gs = int(np.sqrt(nll.shape[0]))
        if gs * gs == nll.shape[0]:
            return nll.reshape(gs, gs), gs
        return nll.reshape(1, -1), nll.shape[0]

    # ─── 特征提取 ───────────────────────────────────────────
    def _extract_features(self, image: Any) -> Optional[np.ndarray]:
        """letterbox 预处理 → ViT patch tokens (有效区域)。

        Returns: (n_valid_patches, 384) ndarray
        """
        import torch
        from PIL import Image

        if isinstance(image, (str, Path)):
            try:
                pil = Image.open(str(image)).convert("RGB")
            except Exception:
                return None
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]  # BGR→RGB
            pil = Image.fromarray(image)
        elif hasattr(image, "convert"):
            pil = image.convert("RGB")
        else:
            return None

        w, h = pil.size
        size = self._img_size
        scale = size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        new_w = max(1, (new_w // 14) * 14 or 14)
        new_h = max(1, (new_h // 14) * 14 or 14)
        resized = pil.resize((new_w, new_h), Image.LANCZOS)

        # 边缘复制填充 (与 PatchCore 引擎一致, 避免灰色填充假异常)
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
                torch.cuda.empty_cache()
                self._model.to("cpu")
                with torch.no_grad():
                    out = self._model(pixel_values=x.to("cpu"))
                self._model.to(self._device())
            else:
                raise

        tokens = out.last_hidden_state[0, self._n_skip:].cpu().numpy()
        return self._crop_to_valid(tokens)

    def _crop_to_valid(self, tokens: np.ndarray) -> np.ndarray:
        """裁切到有效区域 (排除 letterbox 填充 patch)。"""
        vr = getattr(self, "_last_valid_region", None)
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
        """保存 PCA + GMM + NP 校准到 .npz。"""
        if self._pca is None or self._gmm is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "product_name": product_name,
            "pca_components": self._pca.components_,
            "pca_mean": self._pca.mean_,
            "pca_var": self._pca.explained_variance_,
            "gmm_weights": self._gmm.weights_,
            "gmm_means": self._gmm.means_,
            "gmm_covars": self._gmm.covariances_,
            **{f"meta_{k}": v for k, v in self._bank_meta.items()},
        }
        if self._calibrated_threshold is not None:
            save_kwargs["calibrated_threshold"] = self._calibrated_threshold
        if self._np_calibrator is not None and self._np_calibrator.is_fitted:
            import json
            save_kwargs["np_calib_json"] = json.dumps(
                self._np_calibrator.to_dict(), ensure_ascii=False)
        np.savez_compressed(str(path), **save_kwargs)
        logger.info("DINOv2 特征库已保存: %s (%d etalons, threshold=%s)",
                    path, len(self._gmm.weights_),
                    f"{self._calibrated_threshold:.4f}"
                    if self._calibrated_threshold else "N/A")

    def load_bank(self, path: str | Path) -> bool:
        """从 .npz 恢复 PCA + GMM + NP 校准。"""
        path = Path(path)
        if not path.exists():
            logger.warning("DINOv2 特征库不存在: %s", path)
            return False
        try:
            from sklearn.decomposition import PCA
            from sklearn.mixture import GaussianMixture
            data = np.load(str(path), allow_pickle=True)

            pca = PCA(n_components=int(data["pca_components"].shape[0]),
                      whiten=True)  # whiten 必须与训练时一致, 否则分数漂移
            pca.components_ = data["pca_components"]
            pca.mean_ = data["pca_mean"]
            pca.explained_variance_ = data["pca_var"]
            pca.n_components_ = int(pca.components_.shape[0])
            pca.noise_variance_ = 0.0

            m = int(data["gmm_weights"].shape[0])
            gmm = GaussianMixture(n_components=m, covariance_type="diag")
            gmm.weights_ = data["gmm_weights"]
            gmm.means_ = data["gmm_means"]
            gmm.covariances_ = data["gmm_covars"]
            gmm.precisions_ = 1.0 / gmm.covariances_
            gmm.precisions_cholesky_ = np.sqrt(gmm.precisions_)
            gmm.converged_ = True
            gmm.n_iter_ = 1

            self._pca = pca
            self._gmm = gmm
            self._bank_meta = {
                k.replace("meta_", ""): data[k].item()
                for k in data.files if k.startswith("meta_")
            }
            if "calibrated_threshold" in data.files:
                self._calibrated_threshold = float(
                    data["calibrated_threshold"])
            else:
                self._calibrated_threshold = None
            self._np_calibrator = None
            if "np_calib_json" in data.files:
                import json
                from core.np_calibration import NPCalibrator
                try:
                    self._np_calibrator = NPCalibrator.from_dict(
                        json.loads(str(data["np_calib_json"])))
                except Exception as e:
                    logger.warning("NP校准器解析失败, 忽略: %s", e)
            logger.info("DINOv2 特征库已加载: %s (%d etalons, threshold=%s)",
                        path.name, m,
                        f"{self._calibrated_threshold:.4f}"
                        if self._calibrated_threshold else "N/A")
            return True
        except Exception as e:
            logger.error("加载 DINOv2 特征库失败: %s", e)
            return False

    @property
    def has_bank(self) -> bool:
        return self._gmm is not None and self._pca is not None

    def unload(self) -> None:
        self._model = None
        self._pca = None
        self._gmm = None
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
