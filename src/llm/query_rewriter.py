"""Query 改写：把口语化问题改写成 2-3 个专业检索词"""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field, ValidationError

from src.llm.deepseek_client import get_deepseek_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 RAG 检索查询改写助手。把用户的口语化问题改写成 2-3 个更适合知识库检索的专业查询词。

规则：
1. 每个改写结果是一个独立的检索 query，用于向量检索和 BM25 检索
2. 保留原始问题的核心意图，不要偏离主题
3. 把简称补全为完整术语（如 astream → astream_events）
4. 把口语化表达改为专业表达（如 怎么用 → 使用方法 / API 用法）
5. 每个 query 不超过 30 字
6. 不要编造知识库里没有的概念

只输出 JSON，格式：
{"queries": ["query1", "query2", "query3"]}

不要输出任何解释、markdown 代码块或额外文字。"""


class RewriteOutput(BaseModel):
    queries: list[str] = Field(default_factory=list, min_length=1, max_length=5)


class QueryRewriter:
    def __init__(self, num_queries: int = 3) -> None:
        self.client = get_deepseek_client()
        self.num_queries = num_queries

    def rewrite(self, query: str) -> list[str]:
        """
        返回改写后的 query 列表，**一定包含原始 query**。
        如果 LLM 调用或解析失败，回退到 [原始 query]。
        """
        original = query.strip()
        if not original:
            return []

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"原始问题：{original}\n"
                    f"请改写为最多 {self.num_queries} 个检索 query。"
                ),
            },
        ]

        try:
            raw = self.client.chat(messages, temperature=0.1, max_tokens=512)
            parsed = self._parse_output(raw)
        except Exception as e:
            logger.warning(f"Query 改写失败，回退到原始 query: {e}")
            return [original]

        # 去重 + 限制长度 + 保留原始 query
        seen: set[str] = set()
        result: list[str] = []

        # 原始 query 优先
        seen.add(original)
        result.append(original)

        for q in parsed:
            q = q.strip()
            if not q or q in seen or len(q) > 50:
                continue
            seen.add(q)
            result.append(q)
            if len(result) >= self.num_queries:
                break

        return result

    @staticmethod
    def _parse_output(raw: str) -> list[str]:
        """解析 LLM 输出，容错 markdown 代码块包裹"""
        text = raw.strip()
        # 去掉 ```json ... ``` 包裹
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()

        data = json.loads(text)
        output = RewriteOutput.model_validate(data)
        return output.queries


# ── 单例 ──
_rewriter: QueryRewriter | None = None


def get_query_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter