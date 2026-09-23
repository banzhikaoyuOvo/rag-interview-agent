"""Citation 模块验证"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.hybrid_retriever import HybridRetriever
from src.utils.citation import (
    build_context,
    extract_citations,
    format_citation,
    validate_citations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    # 1. format_citation
    print("=" * 60)
    print("测试 1: format_citation")
    print("=" * 60)
    print(format_citation("advantages.json", "abc123"))

    # 2. 检索 + build_context
    print("\n" + "=" * 60)
    print("测试 2: 检索 + build_context")
    print("=" * 60)
    r = HybridRetriever()
    docs = r.retrieve("astream_events 怎么用", "interview_notes", top_k=3, visibility="private")
    ctx = build_context(docs)
    print(f"检索 {len(docs)} 条，上下文长度 {len(ctx)} 字符")
    print("\n前 400 字符预览：")
    print(ctx[:400])

    # 3. 校验合法 citation
    print("\n" + "=" * 60)
    print("测试 3: 合法 citation 校验")
    print("=" * 60)
    if docs:
        d = docs[0]
        legal_answer = (
            f"astream_events 是 LangGraph 的流式接口 "
            f"{format_citation(d.source_file, d.chunk_id)}。"
        )
        check = validate_citations(legal_answer, docs)
        print(f"总引用: {check.total}  合法: {check.valid}  非法: {check.invalid}")

    # 4. 校验非法 citation（模拟 LLM 编造）
    print("\n" + "=" * 60)
    print("测试 4: 非法 citation 校验（模拟 LLM 编造）")
    print("=" * 60)
    fake_answer = (
        "根据文档 [Source: 假的.md, Chunk ID: fake999]，"
        "astream_events 可以这样用 [Source: another_fake.md, Chunk ID: xxx111]。"
    )
    check = validate_citations(fake_answer, docs)
    print(f"总引用: {check.total}  合法: {check.valid}  非法: {check.invalid}")
    print(f"非法列表: {check.invalid_citations}")

    # 5. extract_citations
    print("\n" + "=" * 60)
    print("测试 5: extract_citations")
    print("=" * 60)
    if docs:
        d = docs[0]
        mixed = (
            f"合法 {format_citation(d.source_file, d.chunk_id)} "
            f"非法 [Source: fake.md, Chunk ID: nope]"
        )
        print(extract_citations(mixed))


if __name__ == "__main__":
    main()