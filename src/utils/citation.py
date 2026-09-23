"""Citation 格式化与校验"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.db.hybrid_retriever import RetrievedDoc

# 匹配 [Source: xxx, Chunk ID: yyy] 或 [Source: xxx, ChunkID: yyy]
CITATION_PATTERN = re.compile(
    r"\[Source:\s*([^,\]]+?)\s*,\s*Chunk\s*ID:\s*([^\]]+?)\s*\]",
    re.IGNORECASE,
)


def format_citation(source_file: str, chunk_id: str) -> str:
    """生成单个 citation 标记"""
    return f"[Source: {source_file}, Chunk ID: {chunk_id}]"


def build_context(docs: list[RetrievedDoc], max_chars_per_doc: int = 1200) -> str:
    """
    把检索结果拼成 LLM 上下文。
    每个 chunk 前加编号和 citation，方便 LLM 引用。
    同时标注 source_file，让 LLM 知道哪条来自简历、哪条来自 JD。
    """
    if not docs:
        return ""

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        citation = format_citation(doc.source_file, doc.chunk_id)
        text = doc.text[:max_chars_per_doc]
        # 加 source_file 标签，让 LLM 分清数据来源
        source_tag = f" [{doc.source_file}]" if doc.source_file else ""
        parts.append(f"[文档 {i}]{source_tag} {citation}\n{text}")
    return "\n\n---\n\n".join(parts)

@dataclass
class CitationCheck:
    total: int
    valid: int
    invalid: int
    cited_chunk_ids: list[str]
    invalid_citations: list[tuple[str, str]]  # (source_file, chunk_id)


def validate_citations(answer: str, docs: list[RetrievedDoc]) -> CitationCheck:
    """
    检查回答中的 citation 是否都来自检索结果。
    用于生成后自动校验，防止 LLM 编造来源。
    """
    valid_pairs = {(d.source_file, d.chunk_id) for d in docs}
    matches = CITATION_PATTERN.findall(answer)

    cited_chunk_ids: list[str] = []
    invalid: list[tuple[str, str]] = []
    valid_count = 0

    for source_file, chunk_id in matches:
        source_file = source_file.strip()
        chunk_id = chunk_id.strip()
        cited_chunk_ids.append(chunk_id)
        if (source_file, chunk_id) in valid_pairs:
            valid_count += 1
        else:
            invalid.append((source_file, chunk_id))

    return CitationCheck(
        total=len(matches),
        valid=valid_count,
        invalid=len(invalid),
        cited_chunk_ids=cited_chunk_ids,
        invalid_citations=invalid,
    )


def extract_citations(answer: str) -> list[tuple[str, str]]:
    """从回答中提取所有 (source_file, chunk_id)"""
    return [
        (s.strip(), c.strip())
        for s, c in CITATION_PATTERN.findall(answer)
    ]