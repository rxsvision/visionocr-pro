"""PatchCore 少样本异常检测引擎 (Phase 4B)

自包含实现, 仅依赖 torch + torchvision (无需 anomalib 包)。
算法: 用预训练 WideResNet50 提取 OK 样本的 patch 级特征 → 构建记忆库 →
      推理时计算新图像特征到记忆库最近邻距离 → 异常分数 + 热力图。

典型用法:
- 注册: 10~30 张 OK 样本 → train() → 保存 .pt 特征库
- 检测: 加载特征库 → infer() → anomaly_score + anomaly_map

显存: ~1.5GB (WideResNet50 推理), 记忆库在 CPU 内存 (~50~200MB/产品)。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from engines.base import BaseEngine, EngineMeta, EngineState

logger = logging.getLogger("visionocr.patchcore")

_IMG_SIZE = 256  # PatchCore 标准输入尺寸
_FEATURE_LAYERS = ("layer2", "layer3")  # 多层特征融合


class AnomalibEngine(BaseEngine):
    """PatchCore 异常检测 (少样本, 无需训练标注)。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self._extractor = None  # 特征提取器
        self._memory_bank: Optional[np.ndarray] = None  # (N, D) 特征库
        self._bank_meta: dict = {}  # 元信息 (产品名, 样本数等)

    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="anomalib",
            display_name="PatchCore (少样本异常检测)",
            category="vision",
            vram_gb=1.5,
            license="Apache-2.0",
            description="10~30张OK样本建库, 无需标注即可检测未知缺陷",
            tags=["缺陷检测", "异常检测", "少样本", "PatchCore"],
        )

    def load(self) -> None:
        """加载 WideResNet50 特征提取器。"""
        try:
            import torch
            from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
        except ImportError as e:
            logger.error("依赖缺失: %s (需要 torch + torchvision)", e)
            self.state = EngineState.ERROR
            return

        try:
            device = self._device()
            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            model = wide_resnet50_2(weights=weights)
            model.eval()
            model.to(device)

            # 提取中间层特征的 hook
            self._features: dict[str, torch.Tensor] = {}
            for name in _FEATURE_LAYERS:
                layer = getattr(model, name)
                layer.register_forward_hook(self._make_hook(name))

            self._extractor = model
            self.state = EngineState.READY
            logger.info("PatchCore 特征提取器就绪 (%s)", device)
        except Exception as e:
            logger.error("PatchCore 加载失败: %s", e)
            self.state = EngineState.ERROR

    def train(self, image_paths: list[str],
              coreset_ratio: float = 0.1) -> dict:
        """从 OK 样本构建记忆库。

        Args:
            image_paths: OK 样本图像路径列表 (建议 10~30 张)
            coreset_ratio: 核心集采样比例 (越小越快, 精度略降)

        Returns:
            {"bank_size": int, "n_images": int, "feature_dim": int}
        """
        if not self.is_ready():
            return {"error": "模型未加载"}

        all_features = []
        for path in image_paths:
            feat = self._extract_features(path)
            if feat is not None:
                all_features.append(feat)

        if not all_features:
            return {"error": "无有效图像", "bank_size": 0}

        bank = np.concatenate(all_features, axis=0)  # (N_total, D)
        logger.info("原始特征: %d patches", bank.shape[0])

        # Coreset 子采样 (贪心最远点)
        if coreset_ratio < 1.0 and bank.shape[0] > 100:
            n_select = max(100, int(bank.shape[0] * coreset_ratio))
            bank = self._coreset_subsample(bank, n_select)
            logger.info("Coreset 采样后: %d patches", bank.shape[0])

        self._memory_bank = bank
        self._bank_meta = {
            "n_images": len(all_features),
            "bank_size": bank.shape[0],
            "feature_dim": bank.shape[1],
        }
        return self._bank_meta

    def infer(self, image: Any, **kwargs) -> dict:
        """检测单张图像, 返回异常分数和热力图。

        Returns:
            {"score": float, "anomaly_map": np.ndarray (H,W) 0~1,
             "pred_label": "OK"/"NG"}
        """
        if not self.is_ready():
            return {"score": 0, "anomaly_map": None, "pred_label": "ERROR",
                    "error": "模型未加载"}
        if self._memory_bank is None:
            return {"score": 0, "anomaly_map": None, "pred_label": "ERROR",
                    "error": "记忆库为空, 请先注册 OK 样本"}

        feat = self._extract_features(image)
        if feat is None:
            return {"score": 0, "anomaly_map": None, "pred_label": "ERROR",
                    "error": "无法读取图像"}

        # 计算每个 patch 到记忆库的最近邻距离
        distances = self._nearest_distances(feat)  # (n_patches,)

        # 异常图: reshape 回空间维度
        grid_size = int(np.sqrt(feat.shape[0]))
        anomaly_map = distances.reshape(grid_size, grid_size)

        # 归一化到 0~1
        map_min, map_max = anomaly_map.min(), anomaly_map.max()
        if map_max > map_min:
            anomaly_map = (anomaly_map - map_min) / (map_max - map_min)

        # 图像级分数: 取 top-k patch 均值 (比 max 更鲁棒)
        k = max(1, len(distances) // 20)
        score = float(np.sort(distances)[-k:].mean())

        # 阈值判定 (可配置)
        threshold = kwargs.get("threshold",
                               self.config.get("qc", {}).get(
                                   "confidence_threshold", 0.5))
        pred = "NG" if score > threshold else "OK"

        return {
            "score": round(score, 4),
            "anomaly_map": anomaly_map,
            "pred_label": pred,
        }

    def save_bank(self, path: str | Path, product_name: str = "") -> None:
        """保存记忆库到 .npz 文件。"""
        if self._memory_bank is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            bank=self._memory_bank,
            product_name=product_name,
            **{f"meta_{k}": v for k, v in self._bank_meta.items()},
        )
        logger.info("记忆库已保存: %s (%d patches)", path, self._memory_bank.shape[0])

    def load_bank(self, path: str | Path) -> bool:
        """从 .npz 文件加载记忆库。"""
        path = Path(path)
        if not path.exists():
            logger.warning("记忆库文件不存在: %s", path)
            return False
        try:
            data = np.load(str(path), allow_pickle=True)
            self._memory_bank = data["bank"]
            self._bank_meta = {
                k.replace("meta_", ""): data[k].item()
                for k in data.files if k.startswith("meta_")
            }
            logger.info("记忆库已加载: %s (%d patches)", path.name,
                        self._memory_bank.shape[0])
            return True
        except Exception as e:
            logger.error("加载记忆库失败: %s", e)
            return False

    @property
    def has_bank(self) -> bool:
        return self._memory_bank is not None

    def unload(self) -> None:
        self._extractor = None
        self._memory_bank = None
        self._bank_meta = {}
        self.state = EngineState.UNLOADED
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ─── 内部实现 ────────────────────────────────────────────
    def _device(self) -> str:
        import torch
        cfg = self.config.get("device", "auto")
        if cfg == "cuda" or (cfg == "auto" and torch.cuda.is_available()):
            return "cuda"
        return "cpu"

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def _extract_features(self, image: Any) -> Optional[np.ndarray]:
        """提取图像的 patch 级特征向量。

        Returns: (n_patches, feature_dim) ndarray
        """
        import torch
        import torch.nn.functional as F
        from torchvision import transforms
        from PIL import Image

        # 统一输入
        if isinstance(image, (str, Path)):
            try:
                pil = Image.open(str(image)).convert("RGB")
            except Exception:
                return None
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]  # BGR→RGB
            pil = Image.fromarray(image)
        elif hasattr(image, "convert"):  # PIL
            pil = image.convert("RGB")
        else:
            return None

        transform = transforms.Compose([
            transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(pil).unsqueeze(0).to(self._device())

        with torch.no_grad():
            self._extractor(tensor)

        # 融合多层特征
        feat2 = self._features["layer2"]  # (1, 512, 32, 32)
        feat3 = self._features["layer3"]  # (1, 1024, 16, 16)

        # 上采样 layer3 到 layer2 的空间尺寸
        feat3_up = F.interpolate(feat3, size=feat2.shape[2:],
                                 mode="bilinear", align_corners=False)
        # 拼接: (1, 1536, 32, 32)
        fused = torch.cat([feat2, feat3_up], dim=1)

        # 转为 patch 向量: (n_patches, 1536)
        b, c, h, w = fused.shape
        patches = fused.permute(0, 2, 3, 1).reshape(-1, c)  # (1024, 1536)
        return patches.cpu().numpy()

    def _nearest_distances(self, query: np.ndarray) -> np.ndarray:
        """计算 query patches 到 memory bank 的最近邻 L2 距离。

        Args:
            query: (n_query, D)
        Returns:
            (n_query,) 最近邻距离
        """
        # 分块计算避免内存爆炸
        bank = self._memory_bank
        chunk_size = 512
        min_dists = np.full(query.shape[0], np.inf, dtype=np.float32)

        for i in range(0, query.shape[0], chunk_size):
            q_chunk = query[i:i + chunk_size]  # (chunk, D)
            # L2 距离: ||q - b||^2 = ||q||^2 + ||b||^2 - 2*q·b
            q_norm = np.sum(q_chunk ** 2, axis=1, keepdims=True)  # (chunk, 1)
            b_norm = np.sum(bank ** 2, axis=1, keepdims=True).T  # (1, N)
            dists = q_norm + b_norm - 2 * (q_chunk @ bank.T)  # (chunk, N)
            dists = np.maximum(dists, 0)  # 数值稳定
            min_dists[i:i + chunk_size] = dists.min(axis=1)

        return np.sqrt(min_dists)

    @staticmethod
    def _coreset_subsample(features: np.ndarray, n_select: int) -> np.ndarray:
        """贪心最远点采样 (Coreset)。"""
        n = features.shape[0]
        if n_select >= n:
            return features

        selected = [np.random.randint(n)]
        min_dists = np.full(n, np.inf)

        for _ in range(n_select - 1):
            last = features[selected[-1]]
            dists = np.sum((features - last) ** 2, axis=1)
            min_dists = np.minimum(min_dists, dists)
            selected.append(int(np.argmax(min_dists)))

        return features[selected]
