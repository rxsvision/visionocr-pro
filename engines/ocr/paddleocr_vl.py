"""PaddleOCR-VL - 相机照片/自然场景 OCR (含扭曲/倾斜/低光增强)

基于 PaddleOCR 3.x, 启用文档方向分类与去扭曲 (VL 模式),
专门处理手机相机拍摄的扭曲/倾斜/低光文档。

若 paddleocr 不可用, 自动降级到 RapidOCR (CPU 兜底)。

输出格式与 RapidOCREngine 一致:
    {"text": str, "lines": [...], "confidence": float, "engine": "paddleocr_vl"}
"""
from __future__ import annotations

import logging
import os
from typing import Any

from engines.base import BaseEngine, EngineMeta, EngineState

logger = logging.getLogger("visionocr.paddleocr_vl")


class PaddleOCRVLEngine(BaseEngine):
    """PaddleOCR 3.x VL 模式: 相机拍摄/扭曲文档专用"""

    def __init__(self, config: dict):
        super().__init__(config)
        self._backend = "paddleocr"   # paddleocr | rapidocr (降级)
        self._fallback = None         # 降级时持有的 RapidOCR 实例

    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="paddleocr_vl",
            display_name="PaddleOCR-VL-1.6 (相机照片)",
            category="ocr",
            vram_gb=4.0,
            license="Apache-2.0",
            description="相机拍摄/自然场景文档 OCR, 支持多语言",
            tags=["相机", "自然场景", "多语言"],
        )

    # ─── 生命周期 ────────────────────────────────────────────
    def load(self) -> None:
        """优先加载 PaddleOCR (GPU); DLL 冲突时走子进程; 最终降级 RapidOCR"""
        self.state = EngineState.LOADING
        device = (self.config or {}).get("device", "auto")
        use_gpu = self._should_use_gpu(device)
        self._use_gpu = use_gpu

        # 1) 尝试直接导入 PaddleOCR 3.x
        try:
            from paddleocr import PaddleOCR  # type: ignore

            ocr_cfg = (self.config or {}).get("ocr", {}).get("paddleocr", {})
            kwargs = dict(
                use_doc_orientation_classify=ocr_cfg.get(
                    "use_doc_orientation_classify", True
                ),
                use_doc_unwarping=ocr_cfg.get("use_doc_unwarping", True),
                use_textline_orientation=ocr_cfg.get(
                    "use_textline_orientation", True
                ),
                device="gpu" if use_gpu else "cpu",
            )
            if "lang" in ocr_cfg:
                kwargs["lang"] = ocr_cfg["lang"]

            self._model = PaddleOCR(**kwargs)
            self._backend = "paddleocr"
            self.state = EngineState.READY
            logger.info("加载完成 (device=%s, 直接导入)", "gpu" if use_gpu else "cpu")
            return
        except ImportError:
            logger.warning("paddleocr 未安装, 尝试子进程模式")
        except OSError as e:
            # torch + paddle cudnn DLL 冲突 (WinError 127)
            logger.warning("DLL 冲突 (%s), 切换子进程隔离模式", e)
            self._backend = "subprocess"
            self.state = EngineState.READY
            logger.info("加载完成 (device=%s, 子进程隔离)", "gpu" if use_gpu else "cpu")
            return
        except Exception as e:  # noqa: BLE001
            logger.warning("PaddleOCR 初始化失败 (%s), 尝试子进程模式", e)
            self._backend = "subprocess"
            self.state = EngineState.READY
            return

        # 2) 子进程模式 (worker 脚本存在即可)
        worker = self._worker_path()
        if worker.exists():
            self._backend = "subprocess"
            self.state = EngineState.READY
            logger.info("加载完成 (子进程隔离, worker=%s)", worker.name)
            return

        # 3) 最终降级: RapidOCR
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            self._fallback = RapidOCR()
            self._backend = "rapidocr"
            self.state = EngineState.READY
            logger.warning("已降级到 RapidOCR (CPU 兜底)")
        except ImportError as e:
            self.state = EngineState.ERROR
            logger.error(
                "依赖缺失: 请执行 `pip install paddleocr` "
                "或 `pip install rapidocr_onnxruntime`。原始错误: %s", e
            )
        except Exception as e:  # noqa: BLE001
            self.state = EngineState.ERROR
            logger.error("降级初始化失败: %s", e)

    def unload(self) -> None:
        self._model = None
        self._fallback = None
        self._backend = "paddleocr"
        self.state = EngineState.UNLOADED

    # ─── 推理 ────────────────────────────────────────────────
    def infer(self, image_path: str, **kwargs: Any) -> dict:
        if not self.is_ready():
            return self._empty("引擎未就绪, 请先调用 load()")
        if not image_path or not os.path.isfile(image_path):
            return self._empty(f"图片不存在: {image_path}")

        if self._backend == "paddleocr":
            return self._infer_paddle(image_path)
        if self._backend == "subprocess":
            return self._infer_subprocess(image_path)
        return self._infer_rapid_fallback(image_path)

    # ─── 子进程隔离推理 ─────────────────────────────────────
    @staticmethod
    def _worker_path():
        from pathlib import Path
        return Path(__file__).parent / "_paddle_worker.py"

    def _infer_subprocess(self, image_path: str) -> dict:
        """通过子进程调用 PaddleOCR, 规避 torch cudnn DLL 冲突"""
        import subprocess
        import sys
        import json

        worker = self._worker_path()
        if not worker.exists():
            return self._empty(f"Worker 脚本缺失: {worker}")

        device = "gpu" if getattr(self, "_use_gpu", False) else "cpu"
        cmd = [
            sys.executable, str(worker), image_path,
            "--device", device,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=120,
            )
            stdout = proc.stdout.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                try:
                    err_data = json.loads(stdout)
                    return self._empty(err_data.get("error", "unknown"))
                except (json.JSONDecodeError, ValueError):
                    stderr = proc.stderr.decode("utf-8", errors="replace")[:200]
                    return self._empty(
                        f"子进程失败 (rc={proc.returncode}): {stderr}")

            result = json.loads(stdout)
            if "error" in result:
                return self._empty(result["error"])
            return result

        except subprocess.TimeoutExpired:
            return self._empty("PaddleOCR 子进程超时 (120s)")
        except json.JSONDecodeError as e:
            return self._empty(f"子进程输出解析失败: {e}")
        except Exception as e:  # noqa: BLE001
            return self._empty(f"子进程调用异常: {e}")

    # ─── PaddleOCR 推理 ──────────────────────────────────────
    def _infer_paddle(self, image_path: str) -> dict:
        try:
            result = self._model.predict(image_path)  # 3.x 接口
        except AttributeError:
            try:
                result = self._model.ocr(image_path, cls=True)  # 2.x 兼容
            except Exception as e:  # noqa: BLE001
                return self._empty(f"PaddleOCR 推理失败: {e}")
        except Exception as e:  # noqa: BLE001
            return self._empty(f"PaddleOCR 推理失败: {e}")

        return self._normalize_paddle(result)

    def _normalize_paddle(self, result: Any) -> dict:
        """归一化 PaddleOCR 3.x / 2.x 返回结构"""
        lines: list[dict] = []
        try:
            # 3.x: result 是可迭代的 OCRResult, 每页含 rec_texts / rec_polys / rec_scores
            for page in result:
                texts = getattr(page, "rec_texts", None)
                polys = getattr(page, "rec_polys", None)
                scores = getattr(page, "rec_scores", None)

                if texts is None and isinstance(page, dict):
                    texts = page.get("rec_texts", [])
                    polys = page.get("rec_polys", page.get("dt_polys", []))
                    scores = page.get("rec_scores", [])

                if texts is None:
                    # 2.x: page = [ [box, (txt, score)], ... ]
                    if isinstance(page, (list, tuple)):
                        for item in page:
                            if isinstance(item, (list, tuple)) and len(item) == 2:
                                box, ts = item
                                txt, score = ts
                                lines.append(self._line(txt, box, score))
                    continue

                polys = polys if polys is not None else [[] for _ in texts]
                scores = scores if scores is not None else [0.0 for _ in texts]
                for txt, box, score in zip(texts, polys, scores):
                    lines.append(self._line(txt, box, score))
        except Exception as e:  # noqa: BLE001
            return self._empty(f"PaddleOCR 结果解析失败: {e}")

        return self._pack(lines, "paddleocr_vl")

    # ─── RapidOCR 降级推理 ───────────────────────────────────
    def _infer_rapid_fallback(self, image_path: str) -> dict:
        try:
            result = self._fallback(image_path)
        except Exception as e:  # noqa: BLE001
            return self._empty(f"RapidOCR 兜底推理失败: {e}")

        # 复用 RapidOCR 引擎的归一化逻辑
        from engines.ocr.rapidocr import RapidOCREngine

        normalized = RapidOCREngine._normalize(result)
        normalized["engine"] = "paddleocr_vl(fallback=rapidocr)"
        return normalized

    # ─── 工具 ────────────────────────────────────────────────
    @staticmethod
    def _should_use_gpu(device: str) -> bool:
        if device == "cpu":
            return False
        if device == "cuda":
            return True
        # auto: 探测 paddle 是否编译了 GPU
        try:
            import paddle  # type: ignore

            return bool(paddle.device.is_compiled_with_cuda())
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _line(txt: Any, box: Any, score: Any) -> dict:
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0
        norm_box: list = []
        try:
            if box is not None and len(box) > 0:
                # 可能是 4 点 [[x,y]*4] 或 flat [x1,y1,x2,y2,...]
                first = box[0]
                if isinstance(first, (list, tuple)):
                    norm_box = [[float(p[0]), float(p[1])] for p in box]
                else:
                    pts = list(box)
                    norm_box = [
                        [float(pts[i]), float(pts[i + 1])]
                        for i in range(0, len(pts) - 1, 2)
                    ]
        except Exception:  # noqa: BLE001
            norm_box = []
        return {"text": str(txt), "box": norm_box, "confidence": score_f}

    @staticmethod
    def _pack(lines: list[dict], engine: str) -> dict:
        avg = sum(l["confidence"] for l in lines) / len(lines) if lines else 0.0
        return {
            "text": "\n".join(l["text"] for l in lines),
            "lines": lines,
            "confidence": round(avg, 4),
            "engine": engine,
        }

    def _empty(self, error: str) -> dict:
        return {
            "text": "",
            "lines": [],
            "confidence": 0.0,
            "engine": "paddleocr_vl",
            "error": error,
        }
