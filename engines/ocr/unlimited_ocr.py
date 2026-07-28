"""Unlimited-OCR - 长文档分页 OCR (40+ 页)

由于 unlimited-ocr 包未必可 pip 安装, 这里实现一个实用的等价方案:
    - PDF: 用 PyMuPDF (fitz) 逐页渲染为图片
    - 图片: 直接处理
    - 每页交给 RapidOCR 识别
    - 汇总为全文 + 分页结果

输出:
    {
        "text": str,              # 全文 (页间用分隔符)
        "pages": [                # 每页结果
            {"page": int, "text": str, "confidence": float},
            ...
        ],
        "page_count": int,
        "confidence": float,      # 全文平均置信度
        "engine": "unlimited_ocr",
    }
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

from engines.base import BaseEngine, EngineMeta, EngineState


class UnlimitedOCREngine(BaseEngine):
    """长文档分页 OCR (PDF 拆页 + RapidOCR)"""

    def __init__(self, config: dict):
        super().__init__(config)
        self._rapid = None

    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="unlimited_ocr",
            display_name="Unlimited-OCR (长文档)",
            category="ocr",
            vram_gb=6.0,
            license="MIT",
            description="超长文档分页 OCR, 支持百页级 PDF",
            tags=["长文档", "分页", "PDF"],
        )

    # ─── 生命周期 ────────────────────────────────────────────
    def load(self) -> None:
        self.state = EngineState.LOADING
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError as e:
            self.state = EngineState.ERROR
            print(
                "[UnlimitedOCR] 依赖缺失: 请执行 `pip install rapidocr_onnxruntime` "
                f"(PDF 支持另需 `pip install pymupdf`)。原始错误: {e}"
            )
            return
        try:
            self._rapid = RapidOCR()
            self.state = EngineState.READY
            print("[UnlimitedOCR] 就绪 (分页 + RapidOCR)")
        except Exception as e:  # noqa: BLE001
            self.state = EngineState.ERROR
            print(f"[UnlimitedOCR] 初始化失败: {e}")

    def unload(self) -> None:
        self._rapid = None
        self._model = None
        self.state = EngineState.UNLOADED

    # ─── 推理 ────────────────────────────────────────────────
    def infer(
        self,
        image_path: str,
        max_pages: int | None = None,
        dpi: int = 200,
        **kwargs: Any,
    ) -> dict:
        if not self.is_ready() or self._rapid is None:
            return self._empty("引擎未就绪, 请先调用 load()")
        if not image_path or not os.path.isfile(image_path):
            return self._empty(f"文件不存在: {image_path}")

        ext = os.path.splitext(image_path)[1].lower()
        try:
            if ext == ".pdf":
                page_images = self._pdf_to_images(image_path, dpi, max_pages)
            else:
                page_images = [(1, image_path, True)]  # (page_no, path, is_temp)
        except ImportError as e:
            return self._empty(f"PDF 解析依赖缺失: {e}")
        except Exception as e:  # noqa: BLE001
            return self._empty(f"读取文件失败: {e}")

        pages: list[dict] = []
        tmp_files: list[str] = []
        try:
            for page_no, path, is_temp in page_images:
                if is_temp:
                    tmp_files.append(path)
                page_result = self._ocr_one(path)
                pages.append(
                    {
                        "page": page_no,
                        "text": page_result["text"],
                        "confidence": page_result["confidence"],
                    }
                )
        finally:
            # 清理临时渲染图
            for f in tmp_files:
                try:
                    os.remove(f)
                except OSError:
                    pass

        if not pages:
            return self._empty("未提取到任何页面")

        full_text = "\n\n".join(
            f"--- 第 {p['page']} 页 ---\n{p['text']}" for p in pages
        )
        avg_conf = sum(p["confidence"] for p in pages) / len(pages)
        return {
            "text": full_text,
            "pages": pages,
            "page_count": len(pages),
            "confidence": round(avg_conf, 4),
            "engine": "unlimited_ocr",
        }

    # ─── 内部 ────────────────────────────────────────────────
    def _ocr_one(self, image_path: str) -> dict:
        """对单张图片执行 RapidOCR 并归一化"""
        from engines.ocr.rapidocr import RapidOCREngine

        try:
            result = self._rapid(image_path)
        except Exception as e:  # noqa: BLE001
            return {"text": "", "confidence": 0.0, "error": str(e)}
        normalized = RapidOCREngine._normalize(result)
        return {
            "text": normalized["text"],
            "confidence": normalized["confidence"],
        }

    @staticmethod
    def _pdf_to_images(
        pdf_path: str, dpi: int, max_pages: int | None
    ) -> list[tuple[int, str, bool]]:
        """用 PyMuPDF 把 PDF 每页渲染为临时 PNG

        Returns:
            [(page_no, tmp_png_path, is_temp=True), ...]
        """
        try:
            import fitz  # type: ignore  # PyMuPDF
        except ImportError as e:  # noqa: BLE001
            raise ImportError("请执行 `pip install pymupdf` 以支持 PDF") from e

        results: list[tuple[int, str, bool]] = []
        doc = fitz.open(pdf_path)
        try:
            total = doc.page_count
            limit = total if max_pages is None else min(total, max_pages)
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            tmp_dir = tempfile.mkdtemp(prefix="unlimited_ocr_")
            for i in range(limit):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=matrix)
                out_path = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
                pix.save(out_path)
                results.append((i + 1, out_path, True))
        finally:
            doc.close()
        return results

    def _empty(self, error: str) -> dict:
        return {
            "text": "",
            "pages": [],
            "page_count": 0,
            "confidence": 0.0,
            "engine": "unlimited_ocr",
            "error": error,
        }
