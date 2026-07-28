"""RapidOCR - PP-OCRv6 CPU 兜底方案

基于 rapidocr_onnxruntime 的轻量级 OCR 引擎, 纯 CPU 推理,
作为无 GPU 环境或其它引擎不可用时的最终兜底方案。

输出格式 (统一约定):
    {
        "text": str,                  # 全文 (按行拼接)
        "lines": [                    # 每一行的详细信息
            {"text": str, "box": [[x,y]*4], "confidence": float},
            ...
        ],
        "confidence": float,          # 平均置信度 0~1
        "engine": "rapidocr",
    }
"""
from __future__ import annotations

import os
from typing import Any

from engines.base import BaseEngine, EngineMeta, EngineState


class RapidOCREngine(BaseEngine):
    """PP-OCRv6 (ONNXRuntime) CPU 兜底引擎"""

    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="rapidocr",
            display_name="PP-OCRv6 (CPU兜底)",
            category="ocr",
            vram_gb=0.5,
            license="Apache-2.0",
            description="轻量级 CPU OCR, 无 GPU 时兜底方案",
            tags=["CPU", "轻量", "兜底"],
        )

    # ─── 生命周期 ────────────────────────────────────────────
    def load(self) -> None:
        """初始化 RapidOCR 实例 (按需加载 ONNX 模型)"""
        self.state = EngineState.LOADING
        try:
            # 延迟导入: 未安装时给出明确提示, 不污染全局
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError as e:
            self.state = EngineState.ERROR
            print(
                "[RapidOCR] 依赖缺失: 请执行 `pip install rapidocr_onnxruntime` "
                f"后重试。原始错误: {e}"
            )
            return
        except Exception as e:  # noqa: BLE001
            self.state = EngineState.ERROR
            print(f"[RapidOCR] 初始化失败: {e}")
            return

        try:
            # 从 config 读取可选参数, 兼容缺省
            ocr_cfg = (self.config or {}).get("ocr", {}).get("rapidocr", {})
            self._model = RapidOCR(**ocr_cfg)
            self.state = EngineState.READY
            print("[RapidOCR] 加载完成 (CPU/ONNX)")
        except Exception as e:  # noqa: BLE001
            self.state = EngineState.ERROR
            print(f"[RapidOCR] 模型加载失败: {e}")

    def unload(self) -> None:
        """释放模型对象"""
        self._model = None
        self.state = EngineState.UNLOADED

    # ─── 推理 ────────────────────────────────────────────────
    def infer(self, image_path: str, **kwargs: Any) -> dict:
        """对单张图片执行 OCR

        Args:
            image_path: 图片路径 (支持 jpg/png/bmp/webp 等)

        Returns:
            统一格式 dict, 见模块 docstring
        """
        # 状态校验
        if not self.is_ready() or self._model is None:
            return {
                "text": "",
                "lines": [],
                "confidence": 0.0,
                "engine": "rapidocr",
                "error": "引擎未就绪, 请先调用 load()",
            }

        if not image_path or not os.path.isfile(image_path):
            return {
                "text": "",
                "lines": [],
                "confidence": 0.0,
                "engine": "rapidocr",
                "error": f"图片不存在: {image_path}",
            }

        try:
            # rapidocr 接受路径 / np.ndarray / bytes / PIL.Image
            result = self._model(image_path)
        except Exception as e:  # noqa: BLE001
            return {
                "text": "",
                "lines": [],
                "confidence": 0.0,
                "engine": "rapidocr",
                "error": f"推理失败: {e}",
            }

        return self._normalize(result)

    # ─── 内部工具 ────────────────────────────────────────────
    @staticmethod
    def _normalize(result: Any) -> dict:
        """把 rapidocr 不同版本的返回结构归一化为统一格式

        兼容三种返回:
          A. 记录列表版 (本环境实测): result = (records, aux_scores)
             其中 records = [[box, text, score], ...], box 为 4 点 [[x,y],...]
          B. 并行列表版 (旧 API): result = (boxes, txts, scores)
          C. 对象版 (新 API): OcrResult, 含 .boxes / .txts / .scores
        """
        records: list[tuple] = []  # 统一为 (box, text, score) 序列

        try:
            if result is None:
                records = []
            elif hasattr(result, "txts") or hasattr(result, "boxes"):
                # C. 对象版
                boxes = list(getattr(result, "boxes", []) or [])
                txts = list(getattr(result, "txts", []) or [])
                scores = list(getattr(result, "scores", []) or [])
                scores = scores or [0.0] * len(txts)
                records = list(zip(boxes, txts, scores))
            elif isinstance(result, (tuple, list)) and len(result) > 0:
                head = result[0]
                first_rec = head[0] if isinstance(head, (list, tuple)) and len(head) > 0 else None
                # A. 记录列表版: result[0] = [[box, text, score], ...]
                if (
                    isinstance(first_rec, (list, tuple))
                    and len(first_rec) >= 2
                    and isinstance(first_rec[1], str)
                ):
                    for rec in head:
                        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                            records.append(
                                (rec[0], rec[1], rec[2] if len(rec) > 2 else 0.0)
                            )
                # B. 并行列表版: (boxes, txts, scores)
                elif len(result) >= 3 and isinstance(head, (list, tuple)):
                    boxes, txts, scores = result[0] or [], result[1] or [], result[2] or []
                    records = list(zip(boxes, txts, scores))
                else:
                    # 兜底: 直接尝试把 result 当作记录列表迭代
                    for rec in result:
                        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                            records.append(
                                (rec[0], rec[1], rec[2] if len(rec) > 2 else 0.0)
                            )
        except Exception as e:  # noqa: BLE001
            return {
                "text": "",
                "lines": [],
                "confidence": 0.0,
                "engine": "rapidocr",
                "error": f"结果解析失败: {e}",
            }

        lines = []
        for box, txt, score in records:
            # 跳过非文本记录 (例如误把辅助分数列表当记录)
            if not isinstance(txt, str):
                continue
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = 0.0
            lines.append(
                {
                    "text": txt,
                    "box": RapidOCREngine._norm_box(box),
                    "confidence": score_f,
                }
            )

        avg_conf = (
            sum(l["confidence"] for l in lines) / len(lines) if lines else 0.0
        )
        full_text = "\n".join(l["text"] for l in lines)

        return {
            "text": full_text,
            "lines": lines,
            "confidence": round(avg_conf, 4),
            "engine": "rapidocr",
        }

    @staticmethod
    def _norm_box(box: Any) -> list:
        """把 box 归一化为 [[x,y], ...]; 兼容 4 点 / flat / 空"""
        try:
            if box is None or len(box) == 0:
                return []
            first = box[0]
            if isinstance(first, (list, tuple)):
                return [[float(p[0]), float(p[1])] for p in box]
            pts = list(box)
            return [
                [float(pts[i]), float(pts[i + 1])]
                for i in range(0, len(pts) - 1, 2)
            ]
        except Exception:  # noqa: BLE001
            return []
