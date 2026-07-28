"""Ollama VLM Provider - 本地大模型推理 (Phase 2 实现)

设计要点:
- 通过 Ollama HTTP API (/api/chat) 调用, 不依赖 ollama-python 包。
- 本地优先: 服务在线且目标模型已拉取时直接可用, 全离线。
- 图文理解: 支持 image_path (base64 内嵌) 与纯文本两种调用。
- 服务在线但模型缺失时, 标记 ERROR 并给出明确提示 (不静默假装就绪)。
"""
from __future__ import annotations

import base64
import os
from typing import Any

from engines.base import BaseEngine, EngineMeta, EngineState


class OllamaEngine(BaseEngine):
    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="ollama_vlm",
            display_name="Qwen3-VL-8B (Ollama)",
            category="llm",
            vram_gb=6.0,
            license="Apache-2.0",
            description="本地 Ollama 部署的视觉语言模型, 支持图文理解与字段抽取",
            tags=["VLM", "本地", "Ollama", "图文理解"],
        )

    # ─── 生命周期 ────────────────────────────────────────────
    def load(self) -> None:
        import requests

        self.state = EngineState.LOADING
        llm_cfg = (self.config or {}).get("llm", {}).get("ollama", {})
        self._host = llm_cfg.get("host", "http://localhost:11434").rstrip("/")
        self._model_name = llm_cfg.get("model", "qwen3-vl:8b")
        self._timeout = float(llm_cfg.get("timeout", 600))

        # 1. 服务在线检测 (快速失败)
        try:
            r = requests.get(f"{self._host}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as e:  # noqa: BLE001
            self.state = EngineState.ERROR
            print(f"[Ollama] 服务不可达 ({self._host}): {e}")
            return

        # 2. 模型存在检测 (含 tag 归一化, e.g. qwen3-vl:8b == qwen3-vl:8b-xxx)
        if not self._model_available(models):
            self.state = EngineState.ERROR
            print(
                f"[Ollama] 服务在线但模型 '{self._model_name}' 未拉取。"
                f" 可用: {models or '空'}。请先 `ollama pull {self._model_name}`。"
            )
            return

        self._models = models
        self.state = EngineState.READY
        print(f"[Ollama] 就绪: {self._model_name} @ {self._host}")

    def unload(self) -> None:
        self._model = None
        self.state = EngineState.UNLOADED

    # ─── 推理 ────────────────────────────────────────────────
    def infer(self, image_path: str | None = None, prompt: str = "", **kwargs: Any) -> dict:
        """图文理解推理。

        Args:
            image_path: 可选图片路径, 提供则做图文多模态调用
            prompt: 指令文本

        Returns:
            {"text": str, "confidence": float, "engine": "ollama_vlm", "error"?: str}
        """
        if not self.is_ready():
            return {"text": "", "confidence": 0.0, "engine": "ollama_vlm",
                    "error": "引擎未就绪, 请先 load()"}

        import requests

        message: dict[str, Any] = {"role": "user", "content": prompt or "请描述图片内容。"}
        if image_path and os.path.isfile(image_path):
            try:
                with open(image_path, "rb") as f:
                    message["images"] = [base64.b64encode(f.read()).decode("ascii")]
            except Exception as e:  # noqa: BLE001
                return {"text": "", "confidence": 0.0, "engine": "ollama_vlm",
                        "error": f"图片读取失败: {e}"}

        payload = {
            "model": self._model_name,
            "messages": [message],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": kwargs.get("max_tokens", 8192)},
        }
        try:
            r = requests.post(f"{self._host}/api/chat", json=payload, timeout=self._timeout)
            r.raise_for_status()
            data = r.json()
            text = self._extract_answer(data.get("message", {}))
            return {"text": text, "confidence": 1.0, "engine": "ollama_vlm"}
        except Exception as e:  # noqa: BLE001
            return {"text": "", "confidence": 0.0, "engine": "ollama_vlm",
                    "error": f"推理失败: {e}"}

    # ─── LLMProvider 兼容接口 ────────────────────────────────
    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """OpenAI 风格 chat 接口, 供合同抽取管线调用。"""
        if not self.is_ready():
            return ""
        import requests

        payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": kwargs.get("max_tokens", 8192)},
        }
        try:
            r = requests.post(f"{self._host}/api/chat", json=payload, timeout=self._timeout)
            r.raise_for_status()
            msg = r.json().get("message", {})
            return self._extract_answer(msg)
        except Exception as e:  # noqa: BLE001
            print(f"[Ollama] chat 失败: {e}")
            return ""

    @staticmethod
    def _extract_answer(msg: dict) -> str:
        """从 Ollama 响应提取正式回答。

        Qwen3-VL 等思考模型会把推理放在 thinking 字段, 正式答案在 content。
        若 content 为空 (token 预算被思考耗尽), 兜底从 thinking 剥离 <think>
        标签后取剩余正文。
        """
        content = (msg.get("content") or "").strip()
        if content:
            return content
        thinking = (msg.get("thinking") or "")
        if not thinking:
            return ""
        # 剥离 <think>...</think> 包裹, 取思考之后的正文
        import re
        stripped = re.sub(r"<think>.*?</think>", "", thinking, flags=re.DOTALL)
        stripped = stripped.replace("<think>", "").replace("</think>", "")
        return stripped.strip()

    def is_available(self) -> bool:
        return self.is_ready()

    # ─── 内部 ────────────────────────────────────────────────
    def _model_available(self, models: list[str]) -> bool:
        target = self._model_name
        if target in models:
            return True
        # 归一化: 去掉 :latest, 比较 name 前缀
        base = target.split(":")[0]
        for m in models:
            if m == target or m.split(":")[0] == base:
                return True
        return False
