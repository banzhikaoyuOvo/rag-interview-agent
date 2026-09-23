"""DeepSeek LLM 封装 - 同步 + 流式"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class DeepSeekClient:
    def __init__(self) -> None:
        key = settings.deepseek_api_key
        if not key or key.startswith("sk-xxx") or key == "sk-your-key-here":
            raise ValueError(
                "DEEPSEEK_API_KEY 未配置，请检查 .env\n"
                "  获取地址: https://platform.deepseek.com/api_keys"
            )

        base_url = settings.deepseek_base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = "https://" + base_url

        self.base_url = base_url
        self.api_key = key
        self.model = settings.deepseek_model
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ── 同步调用（用于短输出：Query 改写、评分）──
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout: float = 60.0,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=self._headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ── 异步流式（用于最终答案 + SSE）──
    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", url, headers=self._headers, json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


# ── 单例 ──
_client: DeepSeekClient | None = None


def get_deepseek_client() -> DeepSeekClient:
    global _client
    if _client is None:
        _client = DeepSeekClient()
    return _client