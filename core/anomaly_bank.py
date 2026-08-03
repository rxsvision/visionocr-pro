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
_BANKS_DV_DIR = Path("data/banks_dinov2")  # DINOv2 库独立目录 (与 PatchCore 隔离)


def _safe_name(name: str) -> str:
    """清洗产品名，防止路径穿越攻击。"""
    import re
    # 移除所有路径分隔符和特殊字符
    s = re.sub(r'[\\/:*?"<>|.]', '_', name.strip())
    return s or '_'


def _validate_path(path: Path, root: Path) -> Path:
    """验证路径不越界。"""
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"路径越界: {path} 不在 {root} 内")
    return resolved


def list_banks() -> list[str]:
    """列出所有已建库的产品名。"""
    if not _BANKS_DIR.exists():
        return []
    return sorted(p.stem for p in _BANKS_DIR.glob("*.npz"))


def list_banks_dinov2() -> list[str]:
    """列出所有已建库的 DINOv2 特征库产品名。"""
    if not _BANKS_DV_DIR.exists():
        return []
    return sorted(p.stem for p in _BANKS_DV_DIR.glob("*.npz"))


def bank_path(product_name: str) -> Path:
    """获取产品特征库文件路径。"""
    safe = _safe_name(product_name)
    p = _BANKS_DIR / f"{safe}.npz"
    _validate_path(p, _BANKS_DIR)
    return p


def bank_path_dinov2(product_name: str) -> Path:
    """获取产品 DINOv2 特征库文件路径。"""
    safe = _safe_name(product_name)
    p = _BANKS_DV_DIR / f"{safe}.npz"
    _validate_path(p, _BANKS_DV_DIR)
    return p


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

    # DINOv2 双建库 (Union 第4源, best-effort: 失败不阻塞 PatchCore)
    dv_eng = registry.get("dinov2_anomaly")
    if dv_eng is not None:
        try:
            if not dv_eng.is_ready():
                registry.ensure_loaded("dinov2_anomaly")
            if dv_eng.is_ready():
                dv_meta = dv_eng.train(image_paths)
                if dv_meta.get("error"):
                    result["dinov2_error"] = dv_meta["error"]
                    logger.warning("DINOv2 建库失败: %s", dv_meta["error"])
                else:
                    _BANKS_DV_DIR.mkdir(parents=True, exist_ok=True)
                    dv_eng.save_bank(bank_path_dinov2(product_name),
                                     product_name=product_name)
                    result["dinov2"] = dv_meta
                    result["dinov2_saved_to"] = str(
                        bank_path_dinov2(product_name))
            else:
                result["dinov2_error"] = "DINOv2 模型加载失败 (不影响PatchCore)"
        except Exception as e:
            result["dinov2_error"] = str(e)
            logger.warning("DINOv2 建库异常 (不影响PatchCore): %s", e)
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


def load_product_bank_dinov2(registry, product_name: str) -> bool:
    """加载指定产品的 DINOv2 特征库到引擎。"""
    engine = registry.get("dinov2_anomaly")
    if engine is None:
        return False
    if not engine.is_ready():
        registry.ensure_loaded("dinov2_anomaly")
    if not engine.is_ready():
        return False
    return engine.load_bank(bank_path_dinov2(product_name))


# ─── SubspaceAD 辅助提示通道特征库 (v1.4.0) ─────────────────
# 定位: 快速换线辅助 — 仅分数+热力图提示, 不给自主判定, 人工复核。
_BANKS_SA_DIR = Path("data/banks_subspacead")


def list_banks_subspace() -> list[str]:
    """列出所有已建库的 SubspaceAD 产品名。"""
    if not _BANKS_SA_DIR.exists():
        return []
    return sorted(p.stem for p in _BANKS_SA_DIR.glob("*.npz"))


def bank_path_subspace(product_name: str) -> Path:
    """获取产品 SubspaceAD 特征库文件路径。"""
    safe = _safe_name(product_name)
    p = _BANKS_SA_DIR / f"{safe}.npz"
    _validate_path(p, _BANKS_SA_DIR)
    return p


def delete_bank_subspace(product_name: str) -> bool:
    """删除产品 SubspaceAD 特征库。"""
    p = bank_path_subspace(product_name)
    if p.exists():
        p.unlink()
        logger.info("已删除 SubspaceAD 特征库: %s", product_name)
        return True
    return False


def register_subspace_bank(registry, product_name: str,
                           image_paths: list[str]) -> dict:
    """注册 OK 样本并构建 SubspaceAD 子空间库。

    1-4 张触发快速换线模式 (旋转增广), ≥5 张标准模式。

    Returns:
        {"n_images", "pca_k", "mode", "saved_to", ...} 或 {"error": str}
    """
    engine = registry.get("subspace_ad")
    if engine is None:
        return {"error": "SubspaceAD 引擎未注册 (config: qc.subspacead)"}
    if not engine.is_ready():
        registry.ensure_loaded("subspace_ad")
    if not engine.is_ready():
        return {"error": "SubspaceAD 模型加载失败"}

    result = engine.train(image_paths)
    if result.get("error"):
        return result

    _BANKS_SA_DIR.mkdir(parents=True, exist_ok=True)
    engine.save_bank(bank_path_subspace(product_name),
                     product_name=product_name)
    result["product_name"] = product_name
    result["saved_to"] = str(bank_path_subspace(product_name))
    return result


def load_product_bank_subspace(registry, product_name: str) -> bool:
    """加载指定产品的 SubspaceAD 特征库到引擎。"""
    engine = registry.get("subspace_ad")
    if engine is None:
        return False
    if not engine.is_ready():
        registry.ensure_loaded("subspace_ad")
    if not engine.is_ready():
        return False
    return engine.load_bank(bank_path_subspace(product_name))


def run_subspace_detection(registry, image_path: str,
                           product_name: str = "") -> dict:
    """SubspaceAD 辅助提示检测 (不做自主判定)。

    快速模式自校准偏乐观 (KolektorSDD 实测), 故本通道仅输出
    分数 + 热力图叠加, 判定恒标注"仅供参考", 由人工复核。

    Returns:
        {"score", "anomaly_map", "pred_label" ("REVIEW"/"ERROR"),
         "heatmap_overlay", "mode", ...}
    """
    import cv2
    import numpy as np

    engine = registry.get("subspace_ad")
    if engine is None:
        return {"pred_label": "ERROR",
                "error": "SubspaceAD 引擎未注册 (config: qc.subspacead)"}
    if not engine.is_ready():
        registry.ensure_loaded("subspace_ad")
    if not engine.is_ready():
        return {"pred_label": "ERROR", "error": "SubspaceAD 模型加载失败"}

    # 自动加载产品库 (无产品上下文时自动发现唯一库)
    if not engine.has_bank:
        if product_name:
            if not load_product_bank_subspace(registry, product_name):
                return {"pred_label": "ERROR",
                        "error": f"产品「{product_name}」无 SubspaceAD 特征库, "
                                 f"请先注册 OK 样本"}
        else:
            available = list_banks_subspace()
            if len(available) == 1:
                logger.info("SubspaceAD: 自动加载唯一特征库「%s」",
                            available[0])
                load_product_bank_subspace(registry, available[0])
            elif len(available) > 1:
                return {"pred_label": "ERROR",
                        "error": f"存在 {len(available)} 个 SubspaceAD 特征库, "
                                 f"请在工程师面板指定产品"}
            else:
                return {"pred_label": "ERROR",
                        "error": "无 SubspaceAD 特征库, 请先注册 OK 样本"}
    if not engine.has_bank:
        return {"pred_label": "ERROR", "error": "特征库加载失败"}

    from core.infer_stats import Timer
    with Timer("subspace_ad"):
        result = engine.infer(image_path)
    if result.get("error"):
        return result

    # 热力图叠加 (辅助提示: 橙色 REVIEW 标注, 区别于 NG 红)
    anomaly_map = result.get("anomaly_map")
    overlay = None
    if anomaly_map is not None:
        from core.imutils import imread_unicode, imwrite_unicode
        img = imread_unicode(image_path)
        if img is not None:
            if img.ndim == 2:  # 灰度图 (工业相机常见) → BGR 才能叠彩热力图
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            h, w = img.shape[:2]
            amap = np.asarray(anomaly_map, dtype=np.float32)
            heatmap = cv2.resize(amap, (w, h))
            m0, m1 = heatmap.min(), heatmap.max()
            heatmap = (heatmap - m0) / (m1 - m0) if m1 > m0 else heatmap * 0
            heatmap_u8 = (heatmap * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)

            score = result.get("score", 0)
            mode = engine._bank_meta.get("mode", "standard")
            # cv2.putText 无法渲染中文, 标注用纯 ASCII
            tag = ("REVIEW" if mode == "fast"
                   else f"{'NG' if result.get('pred_label') == 'NG' else 'OK'}")
            color = (0, 165, 255) if mode == "fast" else (
                (0, 0, 255) if tag == "NG" else (0, 200, 0))
            cv2.putText(overlay, f"{tag} ({score:.3f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
            result["heatmap_overlay"] = overlay

            # 审计保存: 热力图 PNG 持久化 (产线追溯)
            try:
                import time as _time
                results_dir = Path(__file__).parent.parent / "results" \
                    / "heatmaps_subspace"
                results_dir.mkdir(parents=True, exist_ok=True)
                ts = _time.strftime("%Y%m%d_%H%M%S")
                stem = Path(image_path).stem
                save_path = results_dir / f"{stem}_{ts}_SA.png"
                imwrite_unicode(str(save_path), overlay)
                result["heatmap_path"] = str(save_path)
            except Exception as e:
                logger.debug("SubspaceAD 热力图保存失败 (非致命): %s", e)
    result.setdefault("heatmap_overlay", overlay)
    return result


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
            if img.ndim == 2:  # 灰度图 (工业相机常见) → BGR 才能叠彩热力图
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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
