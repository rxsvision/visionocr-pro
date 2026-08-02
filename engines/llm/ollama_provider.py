"""Ollama VLM Provider - 本地大模型推理 (Phase 2 实现)

设计要点:
- 通过 Ollama HTTP API (/api/chat) 调用, 不依赖 ollama-python 包。
- 本地优先: 服务在线且目标模型已拉取时直接可用, 全离线。
- 图文理解: 支持 image_path (base64 内嵌) 与纯文本两种调用。
- 服务在线但模型缺失时, 标记 ERROR 并给出明确提示 (不静默假装就绪)。
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any

from engines.base import BaseEngine, EngineMeta, EngineState

logger = logging.getLogger("visionocr.ollama_provider")


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
    @staticmethod
    def _resolve_host(llm_cfg: dict) -> str:
        """解析 Ollama 服务地址。

        OLLAMA_HOST 环境变量为标准 Ollama 约定, 优先于配置文件
        (部署/测试可重定向到备用实例, 无需改 config.yaml)。
        """
        host = os.environ.get("OLLAMA_HOST", "").strip() or llm_cfg.get(
            "host", "http://localhost:11434")
        if not host.startswith(("http://", "https://")):
            host = "http://" + host
        return host.rstrip("/")

    def load(self) -> None:
        import requests

        self.state = EngineState.LOADING
        llm_cfg = (self.config or {}).get("llm", {}).get("ollama", {})
        self._host = self._resolve_host(llm_cfg)
        self._model_name = llm_cfg.get("model", "qwen3-vl:8b")
        self._timeout = float(llm_cfg.get("timeout", 600))

        # 1. 服务在线检测 (快速失败)
        try:
            r = requests.get(f"{self._host}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as e:  # noqa: BLE001
            self.state = EngineState.ERROR
            logger.error("[Ollama] 服务不可达 (%s): %s", self._host, e)
            return

        # 2. 模型存在检测 (含 tag 归一化, e.g. qwen3-vl:8b == qwen3-vl:8b-xxx)
        if not self._model_available(models):
            self.state = EngineState.ERROR
            logger.error(
                "[Ollama] 服务在线但模型 '%s' 未拉取。 可用: %s。请先 `ollama pull %s`。",
                self._model_name, models or "空", self._model_name,
            )
            return

        self._models = models
        self.state = EngineState.READY
        logger.info("[Ollama] 就绪: %s @ %s", self._model_name, self._host)

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
                message["images"] = [
                    base64.b64encode(self._prepare_image_bytes(image_path))
                    .decode("ascii")]
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
            logger.error("[Ollama] chat 失败: %s", e)
            return ""

    # VLM 输入边长上限: 线扫图可达 15000x4096 (~180MB BMP), 原样 base64
    # 会产生 ~240MB 负载导致服务挂起; qwen3-vl 内部也会缩放, 预先降到该
    # 上限既保细节又控负载。
    MAX_VLM_SIDE = 1568

    def _prepare_image_bytes(self, image_path: str) -> bytes:
        """读取并按需下采样图片, 返回可 base64 编码的字节。"""
        import numpy as np

        with open(image_path, "rb") as f:
            raw = f.read()
        try:
            import cv2

            img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                return raw  # 无法解码 → 原样交给服务端
            h, w = img.shape[:2]
            long_side = max(h, w)
            if long_side <= self.MAX_VLM_SIDE:
                return raw
            scale = self.MAX_VLM_SIDE / float(long_side)
            img = cv2.resize(img, (max(1, int(w * scale)),
                                   max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img,
                                   [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                return raw
            logger.info("[Ollama] 大图下采样 %dx%d -> %dx%d",
                        w, h, img.shape[1], img.shape[0])
            return buf.tobytes()
        except Exception:  # noqa: BLE001 解码/缩放失败不应阻断推理
            return raw

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
