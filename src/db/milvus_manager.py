"""Milvus Collection 管理器 - 方案 A：Dense + 元数据，BM25 在应用层"""
from __future__ import annotations

import logging

from pymilvus import DataType, MilvusClient

from src.config import settings

logger = logging.getLogger(__name__)

COLLECTIONS = {
    "resume_advantages": "个人优势 & 项目经历",
    "job_descriptions": "目标岗位 JD",
    "interview_notes": "面试技术笔记",
}


def get_client(uri: str | None = None) -> MilvusClient:
    target = uri or settings.milvus_db_path
    try:
        return MilvusClient(uri=target, timeout=5)
    except Exception as e:
        logger.error(
            f"无法连接 Milvus: {target}\n"
            f"  请检查：\n"
            f"  1. 容器是否运行：docker ps | Select-String milvus\n"
            f"  2. 容器未启动时执行：.\\standalone_embed.bat start\n"
            f"  3. 等 20-40 秒直到 STATUS 显示 (healthy)\n"
            f"  原始错误: {e}"
        )
        raise


def _build_schema(dim: int):
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("source_file", DataType.VARCHAR, max_length=512)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=128)
    schema.add_field("category", DataType.VARCHAR, max_length=128)
    schema.add_field("visibility", DataType.VARCHAR, max_length=16)
    schema.add_field("metadata_json", DataType.VARCHAR, max_length=4096)
    return schema


def _build_index(client: MilvusClient):
    idx = client.prepare_index_params()
    idx.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    return idx


def init_all_collections(drop_existing: bool = False) -> MilvusClient:
    client = get_client()
    dim = settings.embedding_dim

    for name, desc in COLLECTIONS.items():
        if drop_existing and client.has_collection(name):
            client.drop_collection(name)
            logger.info(f"已删除 collection: {name}")

        if not client.has_collection(name):
            schema = _build_schema(dim)
            index = _build_index(client)
            client.create_collection(name, schema=schema, index_params=index)
            logger.info(f"创建 collection: {name} - {desc} (dim={dim})")
        else:
            logger.info(f"collection 已存在: {name}")

    return client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    client = init_all_collections(drop_existing=True)
    for name in COLLECTIONS:
        client.flush(name)
        try:
            n = len(client.query(collection_name=name, filter="id >= 0", output_fields=["id"], limit=16384))
        except Exception:
            n = 0
        print(f"  {name}: {n} entities")