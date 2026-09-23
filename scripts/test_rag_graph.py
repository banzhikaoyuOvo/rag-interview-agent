"""LangGraph RAG 图验证"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.rag_graph import rag_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def run_case(query: str, collection: str, visibility: str) -> None:
    print(f"\n{'#'*60}")
    print(f"# Query: {query}")
    print(f"# Collection: {collection}  visibility={visibility}")
    print(f"{'#'*60}")

    result = rag_graph.invoke({
        "user_query": query,
        "target_collection": collection,
        "visibility": visibility,
    })

    print(f"\n改写后 queries: {result.get('rewritten_queries', [])}")
    print(f"检索到 docs: {len(result.get('retrieved_docs', []))}")

    check = result.get("citation_check")
    if check:
        print(f"Citation: 总 {check.total} / 合法 {check.valid} / 非法 {check.invalid}")
    print(f"重试次数: {result.get('retry_count', 0)}")

    print(f"\n{'─'*60}")
    print("最终回答：")
    print(f"{'─'*60}")
    print(result.get("final_answer", "(无)"))
    print()


def main() -> None:
    test_cases = [
        ("astream_events 怎么用", "interview_notes", "private"),
        ("LangGraph 用在哪", "resume_advantages", "public"),
        ("需要什么技能", "job_descriptions", "public"),
    ]

    for q, c, v in test_cases:
        run_case(q, c, v)


if __name__ == "__main__":
    main()