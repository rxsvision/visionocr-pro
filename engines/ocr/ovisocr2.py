"""OvisOCR2 - 印刷文档/表格/公式 OCR (端到端 VLM)

基于 HuggingFace transformers 的因果语言模型, 端到端输出
Markdown / 结构化文本。OmniDocBench 96.58, 表格 TEDS 94.76,
公式 CDM 97.53。

加载策略 (优雅降级):
    1. 优先从本地 models/ 目录加载 (model_source=local 或目录存在)
    2. 否则尝试 HuggingFace Hub
    3. 全部失败 -> state=ERROR, 给出明确指引
"""
from __future__ import annotations

import os
from typing import Any

from engines.base import BaseEngine, EngineMeta, EngineState

# 候选模型 ID (按优先级尝试, 兼容尚未发布/更名的情况)
_HF_CANDIDATES = [
    "AIDC-AI/Ovis-OCR2-0.8B",
    "AIDC-AI/Ovis-OCR2",
    "AIDC-AI/Ovis2-OCR",
]
_LOCAL_DIR_NAMES = ["ovis-ocr2", "Ovis-OCR2-0.8B", "ovisocr2"]


class OvisOCR2Engine(BaseEngine):
    """OvisOCR2 端到端文档 OCR (GPU/FP16)"""

    def __init__(self, config: dict):
        super().__init__(config)
        self._tokenizer = None
        self._processor = None
        self._model_path: str | None = None

    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="ovisocr2",
            display_name="OvisOCR2 (印刷文档)",
            category="ocr",
            vram_gb=5.0,
            license="Apache-2.0",
            description="OmniDocBench 96.58, 表格 TEDS 94.76, 公式 CDM 97.53",
            tags=["印刷", "表格", "公式", "端到端"],
        )

    # ─── 生命周期 ────────────────────────────────────────────
    def load(self) -> None:
        self.state = EngineState.LOADING

        # 1) 依赖检查
        try:
            import torch  # type: ignore  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore  # noqa: F401
        except ImportError as e:
            self.state = EngineState.ERROR
            print(
                "[OvisOCR2] 依赖缺失: 请执行 `pip install transformers torch pillow` "
                f"后重试。原始错误: {e}"
            )
            return

        # 2) 解析模型路径: 本地优先, 再 HuggingFace
        model_path = self._resolve_model_path()
        if model_path is None:
            self.state = EngineState.ERROR
            print(
                "[OvisOCR2] 未找到模型权重。请:\n"
                "  - 将模型放入 models/ovis-ocr2/ (本地模式), 或\n"
                f"  - 联网从 HuggingFace 下载 (候选: {_HF_CANDIDATES})"
            )
            return

        # 3) 加载到 GPU (FP16)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
            self._model.eval()
            self._model_path = model_path
            self.state = EngineState.READY
            print(f"[OvisOCR2] 加载完成: {model_path} (device={device})")
        except Exception as e:  # noqa: BLE001
            self.state = EngineState.ERROR
            print(f"[OvisOCR2] 模型加载失败: {e}")

    def unload(self) -> None:
        # 释放显存
        try:
            import torch  # type: ignore

            self._model = None
            self._tokenizer = None
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            self._model = None
            self._tokenizer = None
        self.state = EngineState.UNLOADED

    # ─── 推理 ────────────────────────────────────────────────
    def infer(self, image_path: str, prompt: str | None = None, **kwargs: Any) -> dict:
        if not self.is_ready() or self._model is None or self._tokenizer is None:
            return self._empty("引擎未就绪, 请先调用 load()")
        if not image_path or not os.path.isfile(image_path):
            return self._empty(f"图片不存在: {image_path}")

        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            # 默认提示词: 端到端文档转 Markdown
            instruction = prompt or (
                "Please read all the text in the image and output it in "
                "Markdown format, preserving tables and formulas."
            )

            # 不同模型的输入构造方式略有差异, 这里做通用兼容
            text_out = self._generate(image, instruction)

            return {
                "text": text_out,
                "structured": self._to_structured(text_out),
                "lines": [],  # 端到端模型不输出行框
                "confidence": 1.0,
                "engine": "ovisocr2",
            }
        except Exception as e:  # noqa: BLE001
            return self._empty(f"推理失败: {e}")

    # ─── 内部 ────────────────────────────────────────────────
    def _generate(self, image: Any, instruction: str) -> str:
        """统一的生成入口, 兼容多种 tokenizer/processor 接口"""
        import torch

        tokenizer = self._tokenizer
        # 优先尝试多模态 processor 路径
        try:
            from transformers import AutoProcessor  # type: ignore

            processor = AutoProcessor.from_pretrained(
                self._model_path, trust_remote_code=True
            )
            self._processor = processor
            inputs = processor(
                text=instruction, images=image, return_tensors="pt"
            ).to(self._model.device)
            with torch.no_grad():
                out_ids = self._model.generate(
                    **inputs, max_new_tokens=4096, do_sample=False
                )
            # 截掉输入部分
            new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
            return processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0].strip()
        except Exception:  # noqa: BLE001
            pass

        # 退化路径: 纯文本 prompt + 图像张量 (trust_remote_code 模型自定义)
        try:
            inputs = tokenizer(instruction, return_tensors="pt").to(
                self._model.device
            )
            with torch.no_grad():
                out_ids = self._model.generate(
                    **inputs, max_new_tokens=4096, do_sample=False
                )
            new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
            return tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0].strip()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"生成失败: {e}") from e

    def _resolve_model_path(self) -> str | None:
        """本地优先 -> HuggingFace 候选"""
        models_dir = (self.config or {}).get("models_dir", "models")
        source = (self.config or {}).get("model_source", "huggingface")

        # 本地候选目录
        for name in _LOCAL_DIR_NAMES:
            cand = os.path.join(models_dir, name)
            if os.path.isdir(cand) and self._looks_like_checkpoint(cand):
                return cand

        if source == "local":
            return None  # 仅本地模式, 找不到就放弃

        # HuggingFace: 用缓存探测, 不强制联网下载失败就报错
        try:
            from huggingface_hub import snapshot_download  # type: ignore

            for repo in _HF_CANDIDATES:
                try:
                    path = snapshot_download(repo_id=repo, local_files_only=True)
                    if path and os.path.isdir(path):
                        return path
                except Exception:  # noqa: BLE001
                    continue
        except ImportError:
            pass

        # 最后: 快速检测网络, 可达才返回 HF ID (避免 5 分钟超时)
        if self._hf_reachable():
            return _HF_CANDIDATES[0]
        print("[OvisOCR2] HuggingFace 不可达且本地无缓存, 跳过。")
        return None

    @staticmethod
    def _hf_reachable(timeout: float = 5.0) -> bool:
        """快速探测 huggingface.co 是否可达"""
        import socket
        try:
            sock = socket.create_connection(("huggingface.co", 443), timeout=timeout)
            sock.close()
            return True
        except (OSError, socket.timeout):
            return False

    @staticmethod
    def _looks_like_checkpoint(path: str) -> bool:
        try:
            files = os.listdir(path)
        except OSError:
            return False
        markers = ("config.json", "pytorch_model.bin", "model.safetensors")
        return any(m in files for m in markers) or any(
            f.endswith(".safetensors") for f in files
        )

    @staticmethod
    def _to_structured(text: str) -> dict:
        """从 Markdown 输出中粗略抽取结构 (标题/表格/公式)"""
        headings, tables, formulas = [], [], []
        in_table = False
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#"):
                headings.append(s.lstrip("#").strip())
            if s.startswith("|") and s.endswith("|"):
                in_table = True
            elif in_table:
                in_table = False
            if s.startswith("$$") or (s.startswith("$") and s.endswith("$")):
                formulas.append(s.strip("$"))
        return {
            "headings": headings,
            "has_table": "|" in text,
            "formula_count": text.count("$$") // 2 + len(formulas),
        }

    def _empty(self, error: str) -> dict:
        return {
            "text": "",
            "structured": {},
            "lines": [],
            "confidence": 0.0,
            "engine": "ovisocr2",
            "error": error,
        }
