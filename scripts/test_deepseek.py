"""DeepSeek 客户端验证"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.deepseek_client import get_deepseek_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def test_sync() -> None:
    print("\n" + "=" * 60)
    print("测试 1: 同步 chat()")
    print("=" * 60)

    client = get_deepseek_client()
    messages = [
        {"role": "system", "content": "你是简洁的助手，回答不超过 20 字。"},
        {"role": "user", "content": "什么是 RAG？"},
    ]
    answer = client.chat(messages, temperature=0.3)
    print(f"回答: {answer}")


async def test_stream() -> None:
    print("\n" + "=" * 60)
    print("测试 2: 异步流式 chat_stream()")
    print("=" * 60)

    client = get_deepseek_client()
    messages = [
        {"role": "system", "content": "你是简洁的助手。"},
        {"role": "user", "content": "用三句话介绍 LangGraph。"},
    ]
    print("流式输出: ", end="", flush=True)
    async for chunk in client.chat_stream(messages, temperature=0.3):
        print(chunk, end="", flush=True)
    print("\n")


def main() -> None:
    test_sync()
    asyncio.run(test_stream())


if __name__ == "__main__":
    main()