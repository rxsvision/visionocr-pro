"""合同文档读取器 - PDF / 图片 → 文本

策略 (精度优先, 全离线):
1. PDF: 优先用 PyMuPDF 抽取内嵌文本层 (无损, 快)。
   - 若文本层过短 (扫描件), 逐页渲染为图片走 OCR 兜底。
2. 图片: 直接走 OCR 引擎 (registry 路由, 默认 rapidocr 兜底)。

返回统一结构:
    {"text": str, "pages": int, "source": "text"|"ocr", "images": [path,...]}
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger("visionocr.document_reader")

try:
    import cv2
    import numpy as np
    from core.image_preprocess import preprocess_image
    from core.imutils import imread_unicode
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# 文本层最少字符数阈值: 低于此值判定为扫描件, 走 OCR
_MIN_TEXT_CHARS = 30


def read_document(file_path: str, registry=None, ocr_engine: str = "rapidocr") -> dict:
    """读取合同文档, 返回文本与元信息。"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(file_path, registry, ocr_engine)
    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"):
        if ext in (".tiff", ".tif"):
            return _read_tiff(file_path, registry, ocr_engine)
        text = _ocr_image(file_path, registry, ocr_engine)
        return {"text": text, "pages": 1, "source": "ocr", "images": [file_path]}
    if ext in (".heic", ".heif"):
        return _read_heic(file_path, registry, ocr_engine)
    if ext == ".djvu":
        return {"text": "", "pages": 0, "source": "error", "images": [],
                "error": "DjVu 格式暂不支持, 请先转换为 PDF (推荐 djv2pdf 或在线工具)"}
    return {"text": "", "pages": 0, "source": "unknown", "images": [],
            "error": f"不支持的文件类型: {ext}"}


# ─── PDF ─────────────────────────────────────────────────────
def _read_pdf(file_path: str, registry, ocr_engine: str) -> dict:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return {"text": "", "pages": 0, "source": "error", "images": [],
                "error": "缺少 PyMuPDF, 请 `pip install pymupdf`"}

    tmpdir = None
    try:
        doc = fitz.open(file_path)
    except Exception as e:  # noqa: BLE001
        # 加密/损坏/零字节 PDF → 结构化错误, 不中断批量流程 (C2 修复)
        return {"text": "", "pages": 0, "source": "error", "images": [],
                "error": f"PDF 打开失败: {e}"}

    try:
        pages_text: list[str] = []
        rendered: list[str] = []
        need_ocr = False

        for i, page in enumerate(doc):
            txt = page.get_text("text") or ""
            if len(txt.strip()) >= _MIN_TEXT_CHARS:
                pages_text.append(txt)
                continue
            # 扫描件: 渲染为 300dpi 图片后 OCR (审查后从 200 提升)
            need_ocr = True
            if tmpdir is None:
                tmpdir = tempfile.mkdtemp(prefix="visionocr_pdf_")
            pix = page.get_pixmap(dpi=300)
            img_path = os.path.join(tmpdir, f"page_{i:03d}.png")
            pix.save(img_path)
            rendered.append(img_path)
            pages_text.append(_ocr_image(img_path, registry, ocr_engine))

        source = "ocr" if need_ocr else "text"
        return {
            "text": "\n\n".join(t for t in pages_text if t.strip()),
            "pages": len(pages_text),
            "source": source,
            "images": rendered,
        }
    except Exception as e:  # noqa: BLE001
        return {"text": "", "pages": 0, "source": "error", "images": [],
                "error": f"PDF 处理异常: {e}"}
    finally:
        doc.close()
        # C3 修复: 清理临时渲染图片 (OCR 已完成, 不再需要)
        if tmpdir:
            import shutil
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:  # noqa: BLE001
                pass


# ─── 多页 TIFF ──────────────────────────────────────────────
def _read_tiff(file_path: str, registry, ocr_engine: str) -> dict:
    """多页 TIFF: 逐页 OCR (历史扫描件常见格式)。"""
    try:
        from PIL import Image
    except ImportError:
        return {"text": "", "pages": 0, "source": "error", "images": [],
                "error": "缺少 Pillow, 请 `pip install Pillow`"}

    try:
        img = Image.open(file_path)
    except Exception as e:  # noqa: BLE001
        return {"text": "", "pages": 0, "source": "error", "images": [],
                "error": f"TIFF 打开失败: {e}"}

    pages_text: list[str] = []
    tmpdir = tempfile.mkdtemp(prefix="visionocr_tiff_")
    try:
        page_idx = 0
        while True:
            try:
                img.seek(page_idx)
            except EOFError:
                break
            page_path = os.path.join(tmpdir, f"page_{page_idx:03d}.png")
            img.convert("RGB").save(page_path, "PNG")
            pages_text.append(_ocr_image(page_path, registry, ocr_engine))
            page_idx += 1
    finally:
        img.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "text": "\n\n".join(t for t in pages_text if t.strip()),
        "pages": len(pages_text),
        "source": "ocr",
        "images": [file_path],
    }


# ─── HEIC/HEIF (iPhone 默认格式) ────────────────────────────
def _read_heic(file_path: str, registry, ocr_engine: str) -> dict:
    """HEIC/HEIF: 通过 pillow-heif 转为 PNG 后 OCR。"""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        from PIL import Image
    except ImportError:
        return {"text": "", "pages": 0, "source": "error", "images": [],
                "error": "缺少 pillow-heif, 请 `pip install pillow-heif`"}

    try:
        img = Image.open(file_path)
        tmpdir = tempfile.mkdtemp(prefix="visionocr_heic_")
        png_path = os.path.join(tmpdir, "converted.png")
        img.convert("RGB").save(png_path, "PNG")
        img.close()
        text = _ocr_image(png_path, registry, ocr_engine)
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        return {"text": text, "pages": 1, "source": "ocr", "images": [file_path]}
    except Exception as e:  # noqa: BLE001
        return {"text": "", "pages": 0, "source": "error", "images": [],
                "error": f"HEIC 处理失败: {e}"}


# ─── 图片 OCR ────────────────────────────────────────────────
def _ocr_image(image_path: str, registry, ocr_engine: str) -> str:
    if registry is None:
        return ""
    try:
        # 图像预处理 (对比度/纠偏/降噪/放大) — 提升工业相机/手机/扫描件识别率
        ocr_path = image_path
        if _HAS_CV2:
            img = imread_unicode(image_path)
            if img is not None:
                processed = preprocess_image(img)
                # 写入临时文件供引擎读取
                tmp_pp = image_path + ".pp.png"
                cv2.imwrite(tmp_pp, processed)
                ocr_path = tmp_pp

        text = _try_ocr_engine(registry, ocr_engine, ocr_path)

        # 推理级 fallback: 主引擎失败/空结果 → 降级 rapidocr
        if not text and ocr_engine != "rapidocr":
            fallback = "rapidocr"
            logger.info("[OCR] 主引擎 %s 无结果, 降级 %s", ocr_engine, fallback)
            text = _try_ocr_engine(registry, fallback, ocr_path)

        # 清理预处理临时文件
        if ocr_path != image_path and os.path.exists(ocr_path):
            try:
                os.remove(ocr_path)
            except OSError:
                pass
        return text
    except Exception as e:  # noqa: BLE001
        logger.error("[ContractReader] OCR 失败 %s: %s", image_path, e)
        return ""


def _try_ocr_engine(registry, engine_name: str, image_path: str) -> str:
    """尝试用指定引擎执行 OCR, 失败返回空字符串 (不抛异常)"""
    try:
        engine = registry.ensure_loaded(engine_name)
        if not engine.is_ready():
            return ""
        result = engine.infer(image_path)
        if isinstance(result, dict):
            if result.get("error"):
                logger.debug("[OCR] %s 返回错误: %s",
                             engine_name, result["error"])
                return ""
            return result.get("text", "")
        return ""
    except Exception as e:  # noqa: BLE001
        logger.debug("[OCR] %s 异常: %s", engine_name, e)
        return ""
