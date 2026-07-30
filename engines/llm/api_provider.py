"""API VLM Provider - 云端大模型推理 (Phase 2 实现)

设计要点:
- OpenAI 兼容协议 (/v1/chat/completions), 支持 DashScope / OpenAI / 任意兼容端点。
- 作为本地 Ollama 不可用时的降级路径 (需联网 + API Key)。
- API Key 来源优先级: config.llm.api.api_key > 环境变量 VISIONOCR_API_KEY。
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any

from engines.base import BaseEngine, EngineMeta, EngineState

logger = logging.getLogger("visionocr.api_provider")


class APIEngine(BaseEngine):
    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="api_vlm",
            display_name="API VLM (云端)",
            category="llm",
            vram_gb=0.0,
            license="N/A",
            description="云端 API 视觉语言模型 (OpenAI 兼容), 无本地显存占用",
            tags=["VLM", "云端", "API", "多供应商"],
        )

    # ─── 生命周期 ────────────────────────────────────────────
    def load(self) -> None:
        self.state = EngineState.LOADING
        api_cfg = (self.config or {}).get("llm", {}).get("api", {})
        self._base_url = api_cfg.get("base_url", "").rstrip("/")
        self._model_name = api_cfg.get("model", "qwen-vl-max")
        self._api_key = api_cfg.get("api_key") or os.environ.get("VISIONOCR_API_KEY", "")
        self._timeout = float(api_cfg.get("timeout", 120))

        if not self._base_url:
            self.state = EngineState.ERROR
            logger.error("[API VLM] 未配置 base_url")
            return
        if not self._api_key:
            self.state = EngineState.ERROR
            logger.error("[API VLM] 未配置 API Key (config.llm.api.api_key 或 VISIONOCR_API_KEY)")
            return

        self.state = EngineState.READY
        logger.info("[API VLM] 就绪: %s @ %s", self._model_name, self._base_url)

    def unload(self) -> None:
        self._model = None
        self.state = EngineState.UNLOADED

    # ─── 推理 ────────────────────────────────────────────────
    def infer(self, image_path: str | None = None, prompt: str = "", **kwargs: Any) -> dict:
        if not self.is_ready():
            return {"text": "", "confidence": 0.0, "engine": "api_vlm",
                    "error": "引擎未就绪, 请先 load()"}

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt or "请描述图片内容。"}]
        if image_path and os.path.isfile(image_path):
            try:
                b64 = self._encode_image(image_path)
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            except Exception as e:  # noqa: BLE001
                return {"text": "", "confidence": 0.0, "engine": "api_vlm",
                        "error": f"图片读取失败: {e}"}

        text = self._chat([{"role": "user", "content": content}], **kwargs)
        if text:
            return {"text": text, "confidence": 1.0, "engine": "api_vlm"}
        return {"text": "", "confidence": 0.0, "engine": "api_vlm", "error": "API 调用失败"}

    # ─── LLMProvider 兼容接口 ────────────────────────────────
    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        if not self.is_ready():
            return ""
        return self._chat(messages, **kwargs)

    def is_available(self) -> bool:
        return self.is_ready()

    # ─── 内部 ────────────────────────────────────────────────
    def _chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        import requests

        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}",
                   "Content-Type": "application/json"}
        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            logger.error("[API VLM] 请求失败: %s", e)
            return ""

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
