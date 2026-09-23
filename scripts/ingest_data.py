"""数据入库：解析 -> Embedding -> Milvus + BM25 缓存"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank_bm25 import BM25Okapi
from src.config import settings
from src.db.milvus_manager import COLLECTIONS, get_client, init_all_collections
from src.llm.embedder import get_embedder
from src.utils.document_parser import Chunk, parse_directory
from src.utils.tokenizer import tokenize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_DATA_DIR = {
    "resume_advantages": "resume",
    "job_descriptions": "job_descriptions",
    "interview_notes": "interview_notes",
}

COLLECTION_VISIBILITY = {
    "resume_advantages": "public",
    "job_descriptions": "public",
    "interview_notes": "private",
}


def build_bm25_cache(chunks: list[Chunk], collection: str) -> None:
    cache_dir = Path(settings.bm25_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenized = [tokenize(c.text) for c in chunks]
    payload = {"chunks": [c.to_dict() for c in chunks], "tokenized": tokenized}
    cache_path = cache_dir / f"{collection}.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"BM25 缓存已保存: {cache_path} ({len(chunks)} chunks)")


def ingest_collection(client, collection: str, embedder, rebuild: bool = False) -> None:
    logger.info(f"=== 入库 {collection} ===")
    data_dir = Path(settings.data_dir) / COLLECTION_DATA_DIR[collection]
    visibility = COLLECTION_VISIBILITY[collection]
    logger.info(f"数据目录: {data_dir.resolve()}")
    if data_dir.exists():
        files = sorted(f.name for f in data_dir.iterdir() if f.is_file())
        logger.info(f"目录下 {len(files)} 个文件: {files}")

    chunks = parse_directory(data_dir, collection, visibility)
    if not chunks:
        logger.warning(f"未找到数据: {data_dir}")
        return
    logger.info(f"解析得到 {len(chunks)} 个 chunk")

    # 去重
    seen: set[str] = set()
    unique: list[Chunk] = []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            unique.append(c)
    chunks = unique

    # 幂等：跳过已存在的 chunk_id
    if not rebuild:
        try:
            existing = client.query(
                collection_name=collection,
                filter="id >= 0",
                output_fields=["chunk_id"],
                limit=16384,
            )
            existing_ids = {r["chunk_id"] for r in existing}
            before = len(chunks)
            chunks = [c for c in chunks if c.chunk_id not in existing_ids]
            logger.info(f"跳过已存在: {before - len(chunks)}, 待插入: {len(chunks)}")
        except Exception as e:
            logger.warning(f"查询已存在失败，改为全量插入: {e}")

    # 插入 Milvus
    if chunks:
        logger.info(f"生成 embedding (provider={settings.embedding_provider}, dim={settings.embedding_dim})...")
        texts = [c.text for c in chunks]
        vectors = embedder.embed(texts)
        if len(vectors) != len(chunks):
            raise RuntimeError(f"Embedding 数量不匹配: {len(vectors)} != {len(chunks)}")
        if len(vectors[0]) != settings.embedding_dim:
            raise RuntimeError(
                f"Embedding 维度不匹配: 期望 {settings.embedding_dim}, 实际 {len(vectors[0])}"
            )

        rows = [
            {
                "text": c.text,
                "dense_vector": vec,
                "source_file": c.source_file,
                "chunk_id": c.chunk_id,
                "category": c.category,
                "visibility": c.visibility,
                "metadata_json": c.metadata_json,
            }
            for c, vec in zip(chunks, vectors)
        ]
        client.insert(collection_name=collection, data=rows)
        logger.info(f"插入 Milvus: {len(rows)} 条")
    else:
        logger.info("无新数据需要插入")

    # BM25 缓存始终基于全量数据重建
    all_chunks = parse_directory(data_dir, collection, visibility)
    build_bm25_cache(all_chunks, collection)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Interview Agent - 数据入库")
    parser.add_argument("--collection", default=None, help="指定 collection，不传则全部")
    parser.add_argument("--rebuild", action="store_true", help="删除并重建 collection")
    args = parser.parse_args()

    client = init_all_collections(drop_existing=args.rebuild)
    embedder = get_embedder()

    targets = [args.collection] if args.collection else list(COLLECTIONS.keys())
    for col in targets:
        if col not in COLLECTIONS:
            logger.error(f"未知 collection: {col}")
            continue
        ingest_collection(client, col, embedder, rebuild=args.rebuild)

    logger.info("=== 入库完成 ===")
    for col in COLLECTIONS:
        if client.has_collection(col):
            print(f"  {col}: {client.get_collection_stats(col)}")


if __name__ == "__main__":
    main()