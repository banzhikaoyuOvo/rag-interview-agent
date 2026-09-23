"""SSE 流式生成器：推送节点级事件"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass

from src.graph.rag_graph import rag_graph

logger = logging.getLogger(__name__)


def _safe_json(obj) -> str:
    """把 state 里的对象转成可 JSON 序列化的形式"""
    if is_dataclass(obj) and not isinstance(obj, type):
        return json.dumps(asdict(obj), ensure_ascii=False, default=str)
    return json.dumps(obj, ensure_ascii=False, default=str)


def _summarize_node_output(node_name: str, output: dict) -> dict:
    """只提取对前端有用的小字段，避免推大 payload"""
    if node_name == "query_rewrite":
        return {"rewritten_queries": output.get("rewritten_queries", [])}
    if node_name == "hybrid_retrieve":
        docs = output.get("retrieved_docs", [])
        return {"doc_count": len(docs)}
    if node_name == "context_build":
        return {"context_chars": len(output.get("context_text", ""))}
    if node_name == "generate":
        check = output.get("citation_check")
        return {
            "has_answer": bool(output.get("final_answer")),
            "citation_total": getattr(check, "total", 0) if check else 0,
            "citation_invalid": getattr(check, "invalid", 0) if check else 0,
            "attempt_count": output.get("retry_count", 0),
        }
    return {}


async def stream_rag_events(
    query: str,
    collection: str,
    visibility: str | None,
) -> AsyncIterator[dict]:
    """
    SSE 事件流（yield dict，由 sse-starlette 框架负责编码）：
      {"event": "node",  "data": {"node": "...", "output": {...}}}
      {"event": "final", "data": {完整回答 + sources}}
      {"event": "error", "data": {"message": "..."}}
    """
    try:
        inputs = {
            "user_query": query,
            "target_collection": collection,
            "visibility": visibility,
        }

        final_state: dict = {}

        async for chunk in rag_graph.astream(inputs, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                if not isinstance(node_output, dict):
                    continue
                final_state.update(node_output)

                payload = {
                    "node": node_name,
                    "output": _summarize_node_output(node_name, node_output),
                }
                yield {"event": "node", "data": _safe_json(payload)}

        # 最终事件
        docs = final_state.get("retrieved_docs", [])
        check = final_state.get("citation_check")
        final_payload = {
            "answer": final_state.get("final_answer", ""),
            "sources": [
                {
                    "source_file": d.source_file,
                    "chunk_id": d.chunk_id,
                    "score": round(d.score, 4),
                    "text_preview": d.text[:200],
                    "retrieval_source": d.retrieval_source,
                }
                for d in docs
            ],
            "rewritten_queries": final_state.get("rewritten_queries", []),
            "citation_total": getattr(check, "total", 0) if check else 0,
            "citation_valid": getattr(check, "valid", 0) if check else 0,
            "citation_invalid": getattr(check, "invalid", 0) if check else 0,
            "attempt_count": final_state.get("retry_count", 0),
        }
        yield {"event": "final", "data": _safe_json(final_payload)}

    except Exception as e:
        logger.exception("stream_rag_events 失败")
        yield f"event: error\ndata: {_safe_json({'message': str(e)})}\n\n"