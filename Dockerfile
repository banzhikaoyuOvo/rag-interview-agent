# ─────────────────────────────────────────────
# RAG Interview Agent - 多用途镜像
# 同一镜像用于 FastAPI（api）和 Streamlit（web）
# ─────────────────────────────────────────────
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl 用于健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖声明（利用层缓存）
COPY pyproject.toml ./

# 安装依赖（不用 -e .，因为 src 还没复制）
RUN pip install --upgrade pip && \
    pip install \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.34.0" \
        "langgraph>=1.0.0" \
        "langchain-core>=1.0.0" \
        "langchain-text-splitters>=1.0.0" \
        "pymilvus>=3.0.0" \
        "httpx>=0.27.0" \
        "python-dotenv>=1.0.0" \
        "sse-starlette>=2.0.0" \
        "rank-bm25>=0.2.2" \
        "jieba>=0.42.1" \
        "pydantic>=2.7.4" \
        "pydantic-settings>=2.0.0" \
        "streamlit>=1.30.0"

# 复制源码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/

# 运行时目录
RUN mkdir -p data/bm25_cache

EXPOSE 8001 8501

# 默认 FastAPI（web 服务在 compose 里 override）
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]