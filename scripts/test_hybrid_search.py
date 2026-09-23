"""混合检索效果验证"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def show(label: str, docs, score_name: str = "score"):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {score_name}={d.score:.4f} dense_rank={d.dense_rank} bm25_rank={d.bm25_rank}")
        print(f"    file={d.source_file} chunk={d.chunk_id}")
        print(f"    text={d.text[:80]}...")

def main():
    r = HybridRetriever()

    test_cases = [
        ("resume_advantages", "LangGraph 用在哪", "public"),
        ("interview_notes", "astream_events 怎么用", "private"),
        ("interview_notes", "Checkpointer 持久化", "private"),
        ("job_descriptions", "需要什么技能", "public"),
        ("job_descriptions", "RAG 优化", "public"),
    ]

    for collection, query, vis in test_cases:
        print(f"\n\n{'#'*60}\n# Query: {query}  (collection={collection}, vis={vis})\n{'#'*60}")

        dense = r.dense_search(query, collection, top_k=5, visibility=vis)
        show("Dense Only", dense, score_name="cosine")

        bm25 = r.bm25_search(query, collection, top_k=5, visibility=vis)
        show("BM25 Only", bm25, score_name="bm25")

        hybrid = r.retrieve(query, collection, top_k=5, visibility=vis)
        show("Hybrid (RRF)", hybrid, score_name="rrf")


if __name__ == "__main__":
    main()