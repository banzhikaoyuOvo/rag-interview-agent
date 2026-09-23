"""全局配置 - pydantic-settings，从 .env 读取"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── DeepSeek / LLM ──
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # ── Embedding ──
    embedding_provider: str = "dummy"  # openai_compatible | local_bge | dummy
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 16

    # ── Milvus ──
    milvus_db_path: str = "http://localhost:19530"

    # ── BM25 ──
    bm25_cache_dir: str = str(ROOT_DIR / "data" / "bm25_cache")

    # ── Data ──
    data_dir: str = str(ROOT_DIR / "data")


settings = Settings()

# src/config.py 末尾追加
import logging

# 降低第三方库日志级别，避免刷屏
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)