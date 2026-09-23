"""Embedding 封装 - 支持 OpenAI 兼容接口 / 本地 BGE / Dummy"""
from __future__ import annotations

import hashlib
import logging
from typing import Protocol

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbedder:
    """任何 OpenAI 兼容的 /v1/embeddings 接口"""

    def __init__(self, base_url: str, api_key: str, model: str, dim: int, batch_size: int = 16):
        base_url = base_url.strip().rstrip("/")
        if base_url and not base_url.startswith(("http://", "https://")):
            base_url = "https://" + base_url
        if not base_url:
            raise ValueError("EMBEDDING_BASE_URL 为空，请检查 .env")
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        with httpx.Client(timeout=60.0) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                resp = client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda x: x["index"])
                out.extend(d["embedding"] for d in data)
        return out


class LocalBGEEmbedder:
    """本地 sentence-transformers 模型"""

    def __init__(self, model_name: str, dim: int):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


class DummyEmbedder:
    """仅供链路测试，返回确定性假向量，不要用于真实检索"""

    def __init__(self, dim: int):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec: list[float] = []
            while len(vec) < self.dim:
                for b in h:
                    vec.append((b / 255.0) * 2 - 1)
                    if len(vec) >= self.dim:
                        break
            out.append(vec)
        return out


def get_embedder() -> Embedder:
    provider = settings.embedding_provider.lower()
    if provider == "openai_compatible":
        if not settings.embedding_api_key or not settings.embedding_base_url:
            logger.warning("Embedding API 未配置，回退到 DummyEmbedder")
            return DummyEmbedder(settings.embedding_dim)
        return OpenAICompatibleEmbedder(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
        )
    if provider == "local_bge":
        return LocalBGEEmbedder(settings.embedding_model, settings.embedding_dim)
    logger.warning("使用 DummyEmbedder，向量检索结果无意义，仅用于跑通链路")
    return DummyEmbedder(settings.embedding_dim)