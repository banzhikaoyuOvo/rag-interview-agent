"""FastAPI 路由"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from src.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceItem,
)
from src.api.streaming import stream_rag_events
from src.db.milvus_manager import COLLECTIONS, get_client
from src.graph.rag_graph import rag_graph

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """健康检查：Milvus 连通性 + collection 列表"""
    try:
        client = await run_in_threadpool(get_client)
        collections = await run_in_threadpool(client.list_collections)
        return HealthResponse(
            status="ok",
            milvus="connected",
            collections=list(collections),
        )
    except Exception as e:
        logger.warning(f"health check 失败: {e}")
        return HealthResponse(status="degraded", milvus=f"error: {e}")


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """非流式问答"""
    if req.collection not in COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"未知 collection: {req.collection}，可选: {list(COLLECTIONS.keys())}",
        )

    try:
        result = await run_in_threadpool(
            rag_graph.invoke,
            {
                "user_query": req.query,
                "target_collection": req.collection,
                "visibility": req.visibility,
            },
        )
    except Exception as e:
        logger.exception("rag_graph.invoke 失败")
        raise HTTPException(status_code=500, detail=f"RAG 执行失败: {e}")

    docs = result.get("retrieved_docs", [])
    check = result.get("citation_check")

    return AskResponse(
        answer=result.get("final_answer", ""),
        sources=[
            SourceItem(
                source_file=d.source_file,
                chunk_id=d.chunk_id,
                score=round(d.score, 4),
                text_preview=d.text[:200],
                retrieval_source=d.retrieval_source,
            )
            for d in docs
        ],
        rewritten_queries=result.get("rewritten_queries", []),
        citation_total=getattr(check, "total", 0) if check else 0,
        citation_valid=getattr(check, "valid", 0) if check else 0,
        citation_invalid=getattr(check, "invalid", 0) if check else 0,
        attempt_count=result.get("retry_count", 0),
        error=result.get("error"),
    )


@router.get("/ask/stream")
async def ask_stream(
    query: str,
    collection: str = "resume_advantages",
    visibility: str | None = None,
) -> EventSourceResponse:
    """SSE 流式问答：节点级事件 + 最终结果"""
    if collection not in COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"未知 collection: {collection}",
        )

    return EventSourceResponse(
        stream_rag_events(query, collection, visibility),
        media_type="text/event-stream",
    )