"""LangGraph RAG 节点函数"""
from __future__ import annotations

import logging

from src.db.hybrid_retriever import HybridRetriever, RetrievedDoc
from src.graph.state import RAGState
from src.llm.deepseek_client import get_deepseek_client
from src.llm.query_rewriter import get_query_rewriter
from src.utils.citation import build_context, validate_citations

logger = logging.getLogger(__name__)

MAX_RETRY = 1  # 最多重试 1 次（总共生成 2 次）

SYSTEM_PROMPT = """你是一个专业的 AI 求职面试助手，面向 HR 和面试官。

规则：
1. 只能基于下方 CONTEXT 中的内容回答，不要编造。
2. 每个关键结论必须在句末标注来源，格式：[Source: 文件名, Chunk ID: xxx]
3. Chunk ID 必须严格从 CONTEXT 中复制，不要自己编造。
4. CONTEXT 中没有相关信息时，明确回答"公开资料中没有相关信息"。
5. 回答要简洁、具体、面向招聘，适当使用 Markdown。
6. 当问题涉及"自我介绍""优势总结""综合评价""你擅长什么"等需要组合概述的问题时：
   - 允许从多个 CONTEXT 片段提取信息，组合成流畅的概述。
   - 不要先拒答再补充，直接输出组合概述。
   - 每条概述仍须标注来源，格式：[Source: 文件名, Chunk ID: xxx]。
   - 若某条概述由多个 chunk 支撑，可连续标注多个来源。
7. 当问题涉及"和某个岗位匹配吗""适合投这个 JD 吗"等匹配分析类问题时：
   - 先列出 JD 的核心要求（来自 job_descriptions 的 chunk）。
   - 再逐条对标候选人的能力证据（来自 resume_advantages 的 chunk）。
   - 最后给出匹配点、差距项、面试建议三部分。
"""


# ── 单例 ──
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Node 1: Query 改写
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def query_rewrite_node(state: RAGState) -> dict:
    query = state.get("user_query", "").strip()
    if not query:
        return {"error": "user_query 为空", "rewritten_queries": []}

    rewriter = get_query_rewriter()
    queries = rewriter.rewrite(query)
    logger.info(f"[query_rewrite] '{query}' -> {queries}")
    return {"rewritten_queries": queries}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Node 2: 混合检索（多路 query 合并）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def hybrid_retrieve_node(state: RAGState) -> dict:
    queries = state.get("rewritten_queries", [])
    target = state.get("target_collection", "interview_notes")
    visibility = state.get("visibility")

    # 兼容单集合和集合列表
    collections = [target] if isinstance(target, str) else list(target)

    if not queries:
        return {"retrieved_docs": [], "error": "rewritten_queries 为空"}

    retriever = get_retriever()
    merged: dict[str, RetrievedDoc] = {}

    # 每个集合分配 top_k，总预算不变
    per_collection_k = max(3, 8 // len(collections))

    for collection in collections:
        for q in queries:
            docs = retriever.retrieve(
                q, collection, top_k=per_collection_k, visibility=visibility
            )
            for d in docs:
                if d.chunk_id not in merged or d.score > merged[d.chunk_id].score:
                    merged[d.chunk_id] = d

    top_docs = sorted(merged.values(), key=lambda d: d.score, reverse=True)[:8]
    logger.info(
        f"[hybrid_retrieve] {len(collections)} 集合 × {len(queries)} query "
        f"-> {len(top_docs)} 条"
    )
    return {"retrieved_docs": top_docs}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Node 3: 上下文构造
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def context_build_node(state: RAGState) -> dict:
    docs = state.get("retrieved_docs", [])
    context = build_context(docs)
    logger.info(f"[context_build] {len(docs)} 条 -> {len(context)} 字符")
    return {"context_text": context}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Node 4: LLM 生成 + Citation 校验
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def generate_node(state: RAGState) -> dict:
    context = state.get("context_text", "")
    query = state.get("user_query", "")
    docs = state.get("retrieved_docs", [])
    retry_count = state.get("retry_count", 0)

    if not context:
        return {
            "final_answer": "公开资料中没有相关信息，建议直接联系本人。",
            "retry_count": retry_count,
            "error": "context 为空",
        }

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"## CONTEXT\n{context}\n\n## 问题\n{query}",
        },
    ]

    # 重试时明确告诉 LLM 上次的问题
    if retry_count > 0:
        messages.append({
            "role": "user",
            "content": (
                "注意：上次回答缺少合法的来源引用。"
                "请确保每个关键结论都带 [Source: 文件名, Chunk ID: xxx]，"
                "且 Chunk ID 必须严格从 CONTEXT 中复制。"
            ),
        })

    client = get_deepseek_client()
    answer = client.chat(messages, temperature=0.3, max_tokens=2048)

    # 顺便做一次校验，写入 state
    check = validate_citations(answer, docs)
    logger.info(
        f"[generate] attempt={retry_count + 1} "
        f"citations={check.total} valid={check.valid} invalid={check.invalid}"
    )

    return {
        "final_answer": answer,
        "citation_check": check,
        "retry_count": retry_count + 1,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Conditional Edge: Citation 校验路由
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def citation_check_route(state: RAGState) -> str:
    check = state.get("citation_check")
    retry_count = state.get("retry_count", 0)

    # 没有校验结果，放行
    if check is None:
        return "pass"

    # 达到最大重试次数，放行
    if retry_count > MAX_RETRY:
        logger.warning(f"[citation_check] 达到最大重试，放行")
        return "pass"

    # 无引用或存在非法引用 → 重试
    if check.total == 0 or check.invalid > 0:
        logger.info(f"[citation_check] retry (invalid={check.invalid}, total={check.total})")
        return "retry"

    return "pass"