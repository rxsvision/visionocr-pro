"""HunyuanOCR - 腾讯混元手写体/复杂版式 OCR

需要独占约 12GB 显存, 仅在 GPU 环境加载。
加载策略与 OvisOCR2 一致: 本地 models/ 优先, 再 HuggingFace, 全部失败 -> ERROR。
"""
from __future__ import annotations

import os
from typing import Any

from engines.base import BaseEngine, EngineMeta, EngineState

# 候选模型 ID (腾讯混元 OCR 尚未稳定上架, 做多候选兼容)
_HF_CANDIDATES = [
    "tencent/HunyuanOCR",
    "tencent/Hunyuan-OCR",
    "Tencent-Hunyuan/HunyuanOCR",
]
_LOCAL_DIR_NAMES = ["hunyuan-ocr", "HunyuanOCR", "hunyuan_ocr"]


class HunyuanOCREngine(BaseEngine):
    """HunyuanOCR 手写体识别 (GPU 独占, ~12GB VRAM)"""

    def __init__(self, config: dict):
        super().__init__(config)
        self._tokenizer = None
        self._processor = None
        self._model_path: str | None = None

    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="hunyuan_ocr",
            display_name="HunyuanOCR (手写体)",
            category="ocr",
            vram_gb=12.0,
            license="腾讯开源",
            description="手写体/复杂版式 OCR, 腾讯混元大模型驱动",
            tags=["手写", "复杂版式", "大模型"],
        )

    # ─── 生命周期 ────────────────────────────────────────────
    def load(self) -> None:
        self.state = EngineState.LOADING

        # 1) 依赖检查
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore  # noqa: F401
        except ImportError as e:
            self.state = EngineState.ERROR
            print(
                "[HunyuanOCR] 依赖缺失: 请执行 `pip install transformers torch pillow` "
                f"后重试。原始错误: {e}"
            )
            return

        # 2) 显存检查: 需要独占 ~12GB
        if not torch.cuda.is_available():
            self.state = EngineState.ERROR
            print("[HunyuanOCR] 需要 CUDA GPU (约 12GB 显存), 当前无可用 GPU。")
            return
        try:
            free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            if free_gb < self.meta.vram_gb * 0.9:
                print(
                    f"[HunyuanOCR] 警告: 可用显存 {free_gb:.1f}GB "
                    f"< 需求 {self.meta.vram_gb}GB, 可能 OOM。"
                )
        except Exception:  # noqa: BLE001
            pass

        # 3) 解析模型路径
        model_path = self._resolve_model_path()
        if model_path is None:
            self.state = EngineState.ERROR
            print(
                "[HunyuanOCR] 未找到模型权重。请:\n"
                "  - 将模型放入 models/hunyuan-ocr/ (本地模式), 或\n"
                f"  - 联网从 HuggingFace 下载 (候选: {_HF_CANDIDATES})"
            )
            return

        # 4) 加载到 GPU (FP16, 独占)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True,
            )
            self._model.eval()
            self._model_path = model_path
            self.state = EngineState.READY
            print(f"[HunyuanOCR] 加载完成: {model_path} (GPU/FP16)")
        except Exception as e:  # noqa: BLE001
            self.state = EngineState.ERROR
            print(f"[HunyuanOCR] 模型加载失败: {e}")

    def unload(self) -> None:
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
            # 手写体识别提示词
            instruction = prompt or (
                "请识别图片中的所有文字 (含手写体), 按阅读顺序输出, "
                "保留段落结构。"
            )
            text_out = self._generate(image, instruction)

            return {
                "text": text_out,
                "lines": [],
                "confidence": 1.0,
                "engine": "hunyuan_ocr",
            }
        except Exception as e:  # noqa: BLE001
            return self._empty(f"推理失败: {e}")

    # ─── 内部 ────────────────────────────────────────────────
    def _generate(self, image: Any, instruction: str) -> str:
        import torch

        # 优先多模态 processor
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
            new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
            return processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0].strip()
        except Exception:  # noqa: BLE001
            pass

        # 退化: 纯文本 prompt
        try:
            inputs = self._tokenizer(instruction, return_tensors="pt").to(
                self._model.device
            )
            with torch.no_grad():
                out_ids = self._model.generate(
                    **inputs, max_new_tokens=4096, do_sample=False
                )
            new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
            return self._tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0].strip()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"生成失败: {e}") from e

    def _resolve_model_path(self) -> str | None:
        models_dir = (self.config or {}).get("models_dir", "models")
        source = (self.config or {}).get("model_source", "huggingface")

        for name in _LOCAL_DIR_NAMES:
            cand = os.path.join(models_dir, name)
            if os.path.isdir(cand) and self._looks_like_checkpoint(cand):
                return cand

        if source == "local":
            return None

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

        # 快速检测网络, 可达才返回 HF ID (避免 5 分钟超时)
        if self._hf_reachable():
            return _HF_CANDIDATES[0]
        print("[HunyuanOCR] HuggingFace 不可达且本地无缓存, 跳过。")
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

    def _empty(self, error: str) -> dict:
        return {
            "text": "",
            "lines": [],
            "confidence": 0.0,
            "engine": "hunyuan_ocr",
            "error": error,
        }
