"""MinerU - 结构化文档解析 (magic-pdf)

优先使用 magic-pdf (MinerU) 做版面分析 + 结构化输出 (Markdown/JSON)。
若 magic-pdf 不可用, 降级为: RapidOCR 出文本 + 简单正则结构检测
(标题/列表/表格), 仍能产出可用的 Markdown 与 JSON。

输出:
    {"markdown": str, "json": dict, "engine": "mineru"}
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from engines.base import BaseEngine, EngineMeta, EngineState

logger = logging.getLogger("visionocr.mineru")


class MinerUEngine(BaseEngine):
    """MinerU 结构化解析 (含降级方案)"""

    def __init__(self, config: dict):
        super().__init__(config)
        self._backend = "magic_pdf"   # magic_pdf | fallback
        self._rapid = None            # 降级时的 RapidOCR 实例

    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="mineru",
            display_name="MinerU 2.5-Pro (结构化)",
            category="ocr",
            vram_gb=6.0,
            license="Apache-2.0+",
            description="结构化文档解析, 输出 Markdown/JSON, 支持复杂排版",
            tags=["结构化", "Markdown", "复杂排版"],
        )

    # ─── 生命周期 ────────────────────────────────────────────
    def load(self) -> None:
        self.state = EngineState.LOADING

        # 1) 尝试 magic-pdf
        try:
            import magic_pdf  # type: ignore  # noqa: F401

            self._backend = "magic_pdf"
            self.state = EngineState.READY
            logger.info("magic-pdf 可用, 使用结构化解析后端")
            return
        except ImportError:
            logger.warning("magic-pdf 未安装, 降级到 OCR+正则结构检测")
        except Exception as e:  # noqa: BLE001
            logger.warning("magic-pdf 加载异常 (%s), 降级到兜底后端", e)

        # 2) 降级: RapidOCR
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            self._rapid = RapidOCR()
            self._backend = "fallback"
            self.state = EngineState.READY
            logger.info("降级后端就绪 (RapidOCR + 正则结构检测)")
        except ImportError as e:
            self.state = EngineState.ERROR
            logger.error(
                "依赖缺失: 请执行 `pip install magic-pdf` "
                "或 `pip install rapidocr_onnxruntime`。原始错误: %s", e
            )
        except Exception as e:  # noqa: BLE001
            self.state = EngineState.ERROR
            logger.error("降级初始化失败: %s", e)

    def unload(self) -> None:
        self._model = None
        self._rapid = None
        self.state = EngineState.UNLOADED

    # ─── 推理 ────────────────────────────────────────────────
    def infer(self, image_path: str, **kwargs: Any) -> dict:
        if not self.is_ready():
            return self._empty("引擎未就绪, 请先调用 load()")
        if not image_path or not os.path.isfile(image_path):
            return self._empty(f"图片不存在: {image_path}")

        if self._backend == "magic_pdf":
            return self._infer_magic_pdf(image_path)
        return self._infer_fallback(image_path)

    # ─── magic-pdf 后端 ──────────────────────────────────────
    def _infer_magic_pdf(self, image_path: str) -> dict:
        """调用 magic-pdf 解析。API 在不同版本差异较大, 做多重兼容。"""
        try:
            # 新版 (>=1.x): magic_pdf.data.data_reader_writer + PipeResult
            try:
                return self._magic_pdf_new_api(image_path)
            except Exception:  # noqa: BLE001
                pass

            # 旧版/简化 API: magic_pdf 顶层函数
            import magic_pdf  # type: ignore

            for fn_name in ("parse", "extract", "run"):
                fn = getattr(magic_pdf, fn_name, None)
                if callable(fn):
                    res = fn(image_path)
                    md = self._extract_markdown(res)
                    return {
                        "markdown": md,
                        "json": self._md_to_json(md),
                        "engine": "mineru",
                    }
            return self._empty("magic-pdf 无可用解析入口, 请检查版本")
        except Exception as e:  # noqa: BLE001
            # magic-pdf 失败时不致命, 走兜底
            if self._rapid is None:
                try:
                    from rapidocr_onnxruntime import RapidOCR  # type: ignore

                    self._rapid = RapidOCR()
                except Exception:  # noqa: BLE001
                    return self._empty(f"magic-pdf 解析失败: {e}")
            logger.warning("magic-pdf 失败 (%s), 切换兜底后端", e)
            return self._infer_fallback(image_path)

    def _magic_pdf_new_api(self, image_path: str) -> dict:
        """magic-pdf 1.x 风格 API (DataWriter + PipeResult)"""
        from magic_pdf.data.data_reader_writer import FileBasedDataReader  # type: ignore
        from magic_pdf.data.dataset import PymuDocDataset  # type: ignore

        reader = FileBasedDataReader("")
        img_bytes = reader.read(image_path)
        # 图片需先转 PDF 才能走 DocDataset; 这里用 pymupdf 包装
        import fitz  # type: ignore

        doc = fitz.open(stream=img_bytes, filetype=os.path.splitext(image_path)[1].lstrip("."))
        pdf_bytes = doc.tobytes()
        ds = PymuDocDataset(pdf_bytes)
        infer_result = ds.apply(doc_parse_pipe_or_reader=None)  # 可能签名不同
        md = infer_result.get_markdown() if hasattr(infer_result, "get_markdown") else ""
        return {
            "markdown": md,
            "json": self._md_to_json(md),
            "engine": "mineru",
        }

    # ─── 兜底后端 ────────────────────────────────────────────
    def _infer_fallback(self, image_path: str) -> dict:
        try:
            result = self._rapid(image_path)
        except Exception as e:  # noqa: BLE001
            return self._empty(f"兜底 OCR 失败: {e}")

        from engines.ocr.rapidocr import RapidOCREngine

        normalized = RapidOCREngine._normalize(result)
        lines = [l["text"] for l in normalized.get("lines", []) if l["text"].strip()]
        markdown = self._lines_to_markdown(lines)
        return {
            "markdown": markdown,
            "json": self._md_to_json(markdown),
            "engine": "mineru(fallback)",
        }

    # ─── 结构检测 (正则启发式) ───────────────────────────────
    @staticmethod
    def _lines_to_markdown(lines: list[str]) -> str:
        """把 OCR 行文本用启发式规则转成 Markdown"""
        out: list[str] = []
        for raw in lines:
            s = raw.strip()
            if not s:
                continue
            # 标题: 短行 + 无标点结尾 + 全为文字
            if (
                len(s) <= 30
                and not re.search(r"[。，,.;；:：!！?？]$", s)
                and re.search(r"[\u4e00-\u9fa5A-Za-z]", s)
            ):
                # 已带数字编号的标题, e.g. "1. 概述" / "一、背景"
                if re.match(r"^([一二三四五六七八九十]+[、.]|\d+[\.、])", s):
                    out.append(f"## {s}")
                else:
                    out.append(f"### {s}")
            # 列表项
            elif re.match(r"^([-•·*]|\d+[)）.])\s*", s):
                item = re.sub(r"^([-•·*]|\d+[)）.])\s*", "", s)
                out.append(f"- {item}")
            else:
                out.append(s)
        return "\n\n".join(out)

    @staticmethod
    def _md_to_json(markdown: str) -> dict:
        """从 Markdown 抽取结构化 JSON"""
        headings, paragraphs, lists = [], [], []
        for block in markdown.split("\n\n"):
            b = block.strip()
            if not b:
                continue
            if b.startswith("#"):
                headings.append(b.lstrip("#").strip())
            elif b.startswith("- "):
                lists.extend(x[2:].strip() for x in b.splitlines() if x.startswith("- "))
            else:
                paragraphs.append(b)
        return {
            "headings": headings,
            "paragraphs": paragraphs,
            "list_items": lists,
            "table_detected": "|" in markdown,
        }

    @staticmethod
    def _extract_markdown(res: Any) -> str:
        """从 magic-pdf 各种返回结构里抠出 markdown 字符串"""
        if isinstance(res, str):
            return res
        if isinstance(res, dict):
            for key in ("markdown", "md_content", "content", "text"):
                if key in res and isinstance(res[key], str):
                    return res[key]
        for attr in ("get_markdown", "markdown"):
            v = getattr(res, attr, None)
            if callable(v):
                try:
                    return v()
                except Exception:  # noqa: BLE001
                    continue
            if isinstance(v, str):
                return v
        return str(res)

    def _empty(self, error: str) -> dict:
        return {
            "markdown": "",
            "json": {},
            "engine": "mineru",
            "error": error,
        }
