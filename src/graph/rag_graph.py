"""LangGraph RAG 主图构建"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    citation_check_route,
    context_build_node,
    generate_node,
    hybrid_retrieve_node,
    query_rewrite_node,
)
from src.graph.state import RAGState

logger = logging.getLogger(__name__)


def build_rag_graph():
    """构建并编译 RAG 工作流"""
    builder = StateGraph(RAGState)

    # ── 注册节点 ──
    builder.add_node("query_rewrite", query_rewrite_node)
    builder.add_node("hybrid_retrieve", hybrid_retrieve_node)
    builder.add_node("context_build", context_build_node)
    builder.add_node("generate", generate_node)

    # ── 线性边 ──
    builder.add_edge(START, "query_rewrite")
    builder.add_edge("query_rewrite", "hybrid_retrieve")
    builder.add_edge("hybrid_retrieve", "context_build")
    builder.add_edge("context_build", "generate")

    # ── 条件边：Citation 校验 ──
    builder.add_conditional_edges(
        "generate",
        citation_check_route,
        {
            "pass": END,
            "retry": "generate",
        },
    )

    return builder.compile()


# 单例：编译后的可执行图
rag_graph = build_rag_graph()