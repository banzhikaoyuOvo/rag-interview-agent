"""混合检索器 - Dense + BM25 + RRF 融合"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi
from src.config import settings
from src.db.milvus_manager import get_client
from src.llm.embedder import get_embedder
from src.utils.tokenizer import tokenize

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF 平滑常数，标准值


@dataclass
class RetrievedDoc:
    text: str
    source_file: str
    chunk_id: str
    category: str
    visibility: str
    score: float            # RRF 融合后分数
    dense_rank: int | None = None
    bm25_rank: int | None = None
    retrieval_source: Literal["dense", "bm25", "hybrid"] = "hybrid"


class HybridRetriever:
    def __init__(self) -> None:
        self.client: MilvusClient = get_client()
        self.embedder = get_embedder()
        self._bm25_cache: dict[str, tuple[BM25Okapi, list[dict]]] = {}

    # ── BM25 缓存加载 ─────────────────────────────────
    def _load_bm25(self, collection: str) -> tuple[BM25Okapi, list[dict]] | None:
        if collection in self._bm25_cache:
            return self._bm25_cache[collection]

        cache_path = Path(settings.bm25_cache_dir) / f"{collection}.pkl"
        if not cache_path.exists():
            logger.warning(f"BM25 缓存不存在: {cache_path}")
            return None

        with open(cache_path, "rb") as f:
            payload = pickle.load(f)

        bm25 = BM25Okapi(payload["tokenized"])
        chunks: list[dict] = payload["chunks"]
        self._bm25_cache[collection] = (bm25, chunks)
        return bm25, chunks

    # ── Dense 检索 ────────────────────────────────────
    def dense_search(
        self,
        query: str,
        collection: str,
        top_k: int = 20,
        visibility: str | None = None,
    ) -> list[RetrievedDoc]:
        vec = self.embedder.embed([query])[0]
        expr = f'visibility == "{visibility}"' if visibility else None

        results = self.client.search(
            collection_name=collection,
            data=[vec],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_k,
            output_fields=["text", "source_file", "chunk_id", "category", "visibility"],
            filter=expr, 
        )

        docs: list[RetrievedDoc] = []
        for hit in results[0]:
            entity = hit["entity"]
            docs.append(
                RetrievedDoc(
                    text=entity["text"],
                    source_file=entity["source_file"],
                    chunk_id=entity["chunk_id"],
                    category=entity["category"],
                    visibility=entity["visibility"],
                    score=float(hit["distance"]),
                    retrieval_source="dense",
                )
            )
        return docs

    # ── BM25 检索 ─────────────────────────────────────
    def bm25_search(
        self,
        query: str,
        collection: str,
        top_k: int = 20,
        visibility: str | None = None,
    ) -> list[RetrievedDoc]:
        loaded = self._load_bm25(collection)
        if not loaded:
            return []
        bm25, chunks = loaded

        tokens = tokenize(query)
        scores = bm25.get_scores(tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        docs: list[RetrievedDoc] = []
        for i in ranked_idx:
            if len(docs) >= top_k:
                break
            chunk = chunks[i]
            if visibility and chunk.get("visibility") != visibility:
                continue
            if scores[i] < 0:
                # 负分说明 query 词在文档中完全没出现，不参与 RRF
                continue
            docs.append(
                RetrievedDoc(
                    text=chunk["text"],
                    source_file=chunk["source_file"],
                    chunk_id=chunk["chunk_id"],
                    category=chunk["category"],
                    visibility=chunk["visibility"],
                    score=float(scores[i]),
                    retrieval_source="bm25",
                )
            )
        return docs

    # ── RRF 融合 ──────────────────────────────────────
    @staticmethod
    def rrf_fusion(
        dense_docs: list[RetrievedDoc],
        bm25_docs: list[RetrievedDoc],
        k: int = RRF_K,
        top_k: int = 8,
        w_dense: float = 0.7,
        w_bm25: float = 0.3,
    ) -> list[RetrievedDoc]:
        """RRF 加权融合

        w_dense / w_bm25: 权重，Dense 主导排序，BM25 补召回
        """
        by_id: dict[str, RetrievedDoc] = {}
        rrf_scores: dict[str, float] = {}

        for rank, doc in enumerate(dense_docs, start=1):
            cid = doc.chunk_id
            by_id.setdefault(cid, doc)
            by_id[cid].dense_rank = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + w_dense / (k + rank)

        for rank, doc in enumerate(bm25_docs, start=1):
            cid = doc.chunk_id
            if cid not in by_id:
                by_id[cid] = doc
            by_id[cid].bm25_rank = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + w_bm25 / (k + rank)

        merged: list[RetrievedDoc] = []
        for cid, doc in by_id.items():
            doc.score = rrf_scores[cid]
            doc.retrieval_source = "hybrid"
            merged.append(doc)

        merged.sort(key=lambda d: d.score, reverse=True)
        return merged[:top_k]
    # ── 对外统一入口 ──────────────────────────────────
    def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 8,
        visibility: str | None = None,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
    ) -> list[RetrievedDoc]:
        dense_docs = self.dense_search(query, collection, dense_top_k, visibility)
        bm25_docs = self.bm25_search(query, collection, bm25_top_k, visibility)
        return self.rrf_fusion(
            dense_docs, bm25_docs,
            k=RRF_K, top_k=top_k,
            w_dense=0.7, w_bm25=0.3,
        )