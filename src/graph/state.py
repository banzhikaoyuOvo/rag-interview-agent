"""LangGraph RAG 状态定义"""
from __future__ import annotations

from typing import TypedDict

from src.db.hybrid_retriever import RetrievedDoc
from src.utils.citation import CitationCheck


class RAGState(TypedDict, total=False):
    """
    RAG 工作流全局状态。
    total=False 让所有字段可选，START 时只需提供 user_query。
    """

    # ── 输入 ──
    user_query: str
    target_collection: str | list[str]  # 支持单集合或集合列表
    visibility: str | None

    # ── Query 改写 ──
    rewritten_queries: list[str]

    # ── 检索 ──
    retrieved_docs: list[RetrievedDoc]

    # ── 上下文 ──
    context_text: str

    # ── 生成 ──
    final_answer: str

    # ── Citation 校验 ──
    citation_check: CitationCheck
    retry_count: int

    # ── 错误 ──
    error: str