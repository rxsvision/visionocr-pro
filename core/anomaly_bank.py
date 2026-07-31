"""PatchCore 特征库管理 (Phase 4B)

按产品隔离记忆库, 支持:
- 注册 OK 样本 → 构建特征库 → 保存
- 加载已有特征库 → 直接检测
- 列出/删除产品特征库

存储: data/banks/{product_name}.npz
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("visionocr.anomaly_bank")

_BANKS_DIR = Path("data/banks")


def list_banks() -> list[str]:
    """列出所有已建库的产品名。"""
    if not _BANKS_DIR.exists():
        return []
    return sorted(p.stem for p in _BANKS_DIR.glob("*.npz"))


def bank_path(product_name: str) -> Path:
    """获取产品特征库文件路径。"""
    return _BANKS_DIR / f"{product_name}.npz"


def bank_exists(product_name: str) -> bool:
    """检查产品特征库是否已存在。"""
    return bank_path(product_name).exists()


def delete_bank(product_name: str) -> bool:
    """删除产品特征库。"""
    p = bank_path(product_name)
    if p.exists():
        p.unlink()
        logger.info("已删除特征库: %s", product_name)
        return True
    return False


def register_ok_samples(registry, product_name: str,
                        image_paths: list[str],
                        coreset_ratio: float = 0.1) -> dict:
    """注册 OK 样本并构建+保存特征库。

    Args:
        registry: EngineRegistry
        product_name: 产品名 (用于隔离存储)
        image_paths: OK 样本图像路径列表
        coreset_ratio: 核心集采样比例

    Returns:
        {"bank_size": int, "n_images": int, ...} 或 {"error": str}
    """
    engine = registry.get("anomalib")
    if engine is None:
        return {"error": "PatchCore 引擎未注册"}

    if not engine.is_ready():
        registry.ensure_loaded("anomalib")
    if not engine.is_ready():
        return {"error": "PatchCore 模型加载失败"}

    # 构建记忆库
    result = engine.train(image_paths, coreset_ratio=coreset_ratio)
    if result.get("error"):
        return result

    # 保存
    _BANKS_DIR.mkdir(parents=True, exist_ok=True)
    engine.save_bank(bank_path(product_name), product_name=product_name)
    result["product_name"] = product_name
    result["saved_to"] = str(bank_path(product_name))
    return result


def load_product_bank(registry, product_name: str) -> bool:
    """加载指定产品的特征库到引擎。"""
    engine = registry.get("anomalib")
    if engine is None:
        return False
    if not engine.is_ready():
        registry.ensure_loaded("anomalib")
    if not engine.is_ready():
        return False
    return engine.load_bank(bank_path(product_name))


def run_anomaly_detection(registry, image_path: str,
                          product_name: str = "",
                          threshold: float = 0.5) -> dict:
    """执行 PatchCore 异常检测。

    如果指定了 product_name 且当前引擎无记忆库, 自动加载对应产品库。

    Returns:
        {"score": float, "anomaly_map": ndarray, "pred_label": "OK"/"NG",
         "heatmap_overlay": ndarray (BGR)}
    """
    import cv2
    import numpy as np

    engine = registry.get("anomalib")
    if engine is None:
        return {"pred_label": "ERROR", "error": "PatchCore 引擎未注册"}

    if not engine.is_ready():
        registry.ensure_loaded("anomalib")
    if not engine.is_ready():
        return {"pred_label": "ERROR", "error": "模型加载失败"}

    # 自动加载产品库
    if product_name and not engine.has_bank:
        if not load_product_bank(registry, product_name):
            return {"pred_label": "ERROR",
                    "error": f"产品「{product_name}」无特征库, 请先注册 OK 样本"}

    from core.infer_stats import Timer
    with Timer("anomalib"):
        result = engine.infer(image_path, threshold=threshold)
    if result.get("error"):
        return result

    # 生成热力图叠加
    anomaly_map = result.get("anomaly_map")
    if anomaly_map is not None:
        from core.imutils import imread_unicode, imwrite_unicode
        img = imread_unicode(image_path)
        if img is not None:
            h, w = img.shape[:2]
            heatmap = cv2.resize(anomaly_map, (w, h))
            heatmap_u8 = (heatmap * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)

            # 标注判定
            label = result["pred_label"]
            color = (0, 0, 255) if label == "NG" else (0, 200, 0)
            cv2.putText(overlay, f"{label} ({result['score']:.3f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            result["heatmap_overlay"] = overlay

            # 审计保存: 热力图PNG持久化 (产线追溯)
            try:
                import time as _time
                results_dir = Path(__file__).parent.parent / "results" / "heatmaps"
                results_dir.mkdir(parents=True, exist_ok=True)
                ts = _time.strftime("%Y%m%d_%H%M%S")
                stem = Path(image_path).stem
                save_path = results_dir / f"{stem}_{ts}_{label}.png"
                imwrite_unicode(str(save_path), overlay)
                result["heatmap_path"] = str(save_path)
                logger.info("热力图已保存: %s", save_path.name)
            except Exception as e:
                logger.debug("热力图保存失败 (非致命): %s", e)
        else:
            result["heatmap_overlay"] = None
    else:
        result["heatmap_overlay"] = None

    return result
