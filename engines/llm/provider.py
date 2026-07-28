"""LLM Provider 基类 - 统一的 LLM 调用接口"""
from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """LLM 提供者抽象基类, 统一 chat 接口"""

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]]) -> str:
        """
        发送消息列表并返回模型回复。

        Args:
            messages: OpenAI 格式消息列表,
                      e.g. [{"role": "user", "content": "..."}]

        Returns:
            模型回复文本
        """
        ...

    def is_available(self) -> bool:
        """检查 provider 是否可用 (服务在线 / key 有效)"""
        return False
