"""Query 改写验证"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.query_rewriter import get_query_rewriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    r = get_query_rewriter()

    test_queries = [
        "astream怎么用",
        "checkpointer 是干嘛的",
        "LangGraph 用在哪",
        "需要什么技能",
        "他会 RAG 吗",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"原始: {q}")
        print(f"{'='*60}")
        result = r.rewrite(q)
        for i, rq in enumerate(result, 1):
            marker = "  [原始]" if i == 1 else "  [改写]"
            print(f"{marker} {rq}")


if __name__ == "__main__":
    main()