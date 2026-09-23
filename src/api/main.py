"""FastAPI app 入口"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.db.milvus_manager import COLLECTIONS, get_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时检查 Milvus 连接"""
    logger.info("启动中...")
    try:
        client = get_client()
        cols = client.list_collections()
        logger.info(f"Milvus 已连接，collections: {cols}")

        missing = [c for c in COLLECTIONS if c not in cols]
        if missing:
            logger.warning(
                f"以下 collection 缺失: {missing}\n"
                f"  请先跑: python scripts\\ingest_data.py --rebuild"
            )
    except Exception as e:
        logger.error(f"Milvus 连接失败: {e}")
        logger.error("  请检查: docker ps | Select-String milvus")

    yield

    logger.info("关闭中...")


app = FastAPI(
    title="RAG Interview Agent",
    description="个人求职 RAG 助手：混合检索 + Query 改写 + Citation 溯源",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS：允许本地开发的前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产要收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root() -> dict:
    return {
        "name": "RAG Interview Agent",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": ["/api/health", "/api/ask", "/api/ask/stream"],
    }