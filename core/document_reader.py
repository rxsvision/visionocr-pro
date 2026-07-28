"""合同文档读取器 - PDF / 图片 → 文本

策略 (精度优先, 全离线):
1. PDF: 优先用 PyMuPDF 抽取内嵌文本层 (无损, 快)。
   - 若文本层过短 (扫描件), 逐页渲染为图片走 OCR 兜底。
2. 图片: 直接走 OCR 引擎 (registry 路由, 默认 rapidocr 兜底)。

返回统一结构:
    {"text": str, "pages": int, "source": "text"|"ocr", "images": [path,...]}
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

# 文本层最少字符数阈值: 低于此值判定为扫描件, 走 OCR
_MIN_TEXT_CHARS = 30


def read_document(file_path: str, registry=None, ocr_engine: str = "rapidocr") -> dict:
    """读取合同文档, 返回文本与元信息。"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(file_path, registry, ocr_engine)
    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
        text = _ocr_image(file_path, registry, ocr_engine)
        return {"text": text, "pages": 1, "source": "ocr", "images": [file_path]}
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
            # 扫描件: 渲染为 200dpi 图片后 OCR
            need_ocr = True
            if tmpdir is None:
                tmpdir = tempfile.mkdtemp(prefix="visionocr_pdf_")
            pix = page.get_pixmap(dpi=200)
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


# ─── 图片 OCR ────────────────────────────────────────────────
def _ocr_image(image_path: str, registry, ocr_engine: str) -> str:
    if registry is None:
        return ""
    try:
        engine = registry.ensure_loaded(ocr_engine)
        if not engine.is_ready():
            engine = registry.ensure_loaded("rapidocr")
        result = engine.infer(image_path)
        return result.get("text", "") if isinstance(result, dict) else ""
    except Exception as e:  # noqa: BLE001
        print(f"[ContractReader] OCR 失败 {image_path}: {e}")
        return ""
