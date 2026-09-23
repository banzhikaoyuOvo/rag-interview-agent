"""FastAPI Request / Response 模型"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="用户问题")
    collection: str = Field(
        default="resume_advantages",
        description="目标 collection: resume_advantages / job_descriptions / interview_notes",
    )
    visibility: str | None = Field(
        default=None,
        description="权限过滤: public / private / None(不过滤)",
    )
    top_k: int = Field(default=8, ge=1, le=20, description="最终返回文档数")


class SourceItem(BaseModel):
    source_file: str
    chunk_id: str
    score: float
    text_preview: str
    retrieval_source: str = "hybrid"


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    rewritten_queries: list[str] = Field(default_factory=list)
    citation_total: int = 0
    citation_valid: int = 0
    citation_invalid: int = 0
    attempt_count: int = 0
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    milvus: str
    collections: list[str] = Field(default_factory=list)