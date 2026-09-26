# RAG Interview Agent

> 基于 LangGraph + Milvus + Hybrid Search 的企业级 RAG 求职知识库助手

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-orange)](https://langchain-ai.github.io/langgraph/)
[![Milvus](https://img.shields.io/badge/Milvus-2.5-blue)](https://milvus.io/)

一个面向求职场景的 RAG Agent：给 HR 用，可以提问候选人信息；给自己用，可以做 JD 匹配分析和面试准备。

## ✨ 核心亮点

- **混合检索（Dense + BM25 + RRF）**：解决纯向量检索对技术专有名词（如 `astream_events`）召回不稳定的问题
- **Query 改写 + 多路召回**：LLM 将口语化 query 改写成 2-3 个专业检索词，保留原始 query 参与检索
- **跨集合检索**：支持一次查询同时检索多个 Collection，用于 JD 匹配分析
- **Citation 溯源 + 生成后校验**：每条关键结论带 `[Source: 文件名, Chunk ID]`，非法引用触发重试
- **LangGraph 5 节点编排**：Query 改写 → 混合检索 → 上下文构造 → LLM 生成 → Citation 校验
- **公开 / 私有权限隔离**：`visibility` 字段在检索层强制过滤，防止 prompt injection

## 📐 架构演进：V1 → V2

| 维度 | V1 Parking-YOLO | V2 RAG Interview Agent |
|------|-----------------|----------------------|
| Web 框架 | Flask（同步 WSGI） | **FastAPI（异步 ASGI）** |
| 编排 | 无（线性调用） | **LangGraph StateGraph（5 节点）** |
| 检索 | 纯向量检索 | **Hybrid（Dense + BM25）+ RRF** |
| Query 处理 | 原始 query 直接检索 | **LLM Query Rewrite + 多路召回** |
| 引用溯源 | 无 | **`[Source: 文件名, Chunk ID]` + 生成后校验** |
| 向量库 | Milvus Lite | **Milvus Standalone (Docker)** |
| 分词 | jieba 单一 | **中英混合 tokenizer** |
| 权限 | 无 | **visibility 字段（public/private）** |
| 检索模式 | 单集合 | **单集合 + 跨集合（JD 匹配）** |

## 🚀 快速开始

### 前置依赖

- Python 3.13
- Docker Desktop（用于运行 Milvus Standalone）
- SiliconFlow API Key（[注册地址](https://cloud.siliconflow.cn/)，用于 BGE-M3 Embedding）
- DeepSeek API Key（[注册地址](https://platform.deepseek.com/)，用于 LLM）

### 1. 克隆项目

```bash
git clone https://github.com/banzhikaoyuOvo/rag-interview-agent.git
cd rag-interview-agent
```

### 2. 创建 Conda 环境

```bash
conda create -n rag-interview python=3.13 -y
conda activate rag-interview
pip install -e .
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入 API Key：

```env
# DeepSeek LLM
DEEPSEEK_API_KEY=sk-你的deepseek-key

# SiliconFlow Embedding（BGE-M3）
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_API_KEY=sk-你的siliconflow-key
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024

# Milvus
MILVUS_DB_PATH=http://localhost:19530
```

### 4. 启动 Milvus Standalone

```bash
# 拉取镜像（国内建议用镜像源）
docker pull docker.m.daocloud.io/milvusdb/milvus:v2.5.14
docker tag docker.m.daocloud.io/milvusdb/milvus:v2.5.14 milvusdb/milvus:v2.5.14

# 启动容器
# Windows: .\standalone_embed.bat start
# Linux/Mac: bash standalone_embed.sh start

# 等 20-40 秒直到 healthy
docker ps | grep milvus
```

### 5. 数据入库

```bash
python scripts/ingest_data.py --rebuild
```

**预期输出**：

```text
=== 入库 resume_advantages ===
解析得到 14 个 chunk
=== 入库 job_descriptions ===
解析得到 7 个 chunk
=== 入库 interview_notes ===
解析得到 76 个 chunk
=== 入库完成 ===
```

### 6. 启动服务

**终端 1：FastAPI 后端**

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
```

**终端 2：Streamlit 前端**

```bash
streamlit run src/web/streamlit_app.py --server.port 8501 --server.headless true
```

### 7. 打开浏览器

- **Streamlit 前端**：http://localhost:8501
- **Swagger UI**：http://localhost:8001/docs
- **健康检查**：http://localhost:8001/api/health

## 🧠 核心设计

### 混合检索 Hybrid Search

```text
用户 query
  ↓
Query 改写（LLM → 2-3 个专业检索词）
  ↓
┌──────────────────┬──────────────────┐
│ Dense 检索        │ BM25 检索         │
│ （BGE-M3 向量）   │ （jieba + rank_bm25）│
│ 抓语义相似        │ 抓关键词精确匹配    │
└────────┬─────────┴─────────┬────────┘
         │                   │
         └─────────┬─────────┘
                   ↓
              RRF 融合
         score = Σ 1/(k + rank)
                   ↓
              输出 Top-8
```

**为什么用 RRF 而不是加权求和**：Dense 是余弦相似度（0~1），BM25 是无界实数，量纲不同无法直接相加。RRF 只看排名，无需归一化、无需调参。

### Citation 溯源 + 生成后校验

```text
LLM 生成答案
  ↓
正则提取 [Source: xxx, Chunk ID: yyy]
  ↓
与检索结果的 (source_file, chunk_id) 集合比对
  ↓
┌─────────────┬─────────────┐
│ 全部合法     │ 存在非法引用 │
│ → END       │ → 重试生成   │
└─────────────┴─────────────┘
```

### 跨集合检索（JD 匹配）

```python
# 单集合（向后兼容）
rag_graph.invoke({
    "user_query": "astream_events 怎么用",
    "target_collection": "interview_notes",
    "visibility": "private",
})

# 多集合（JD 匹配分析）
rag_graph.invoke({
    "user_query": "快手的 AI 应用开发工程师岗位，我匹配吗？",
    "target_collection": ["resume_advantages", "job_descriptions"],
    "visibility": "public",
})
```

## 📊 数据规模

| Collection | 用途 | visibility | 条数 |
|-----------|------|-----------|------|
| `resume_advantages` | 个人简历 & 项目 & 技能 | public | 14 |
| `job_descriptions` | 目标岗位 JD | public | 7 |
| `interview_notes` | 面试技术笔记 | private | 76 |
| **总计** | | | **97** |

## 🎯 演示示例

### 示例 1：HR 提问候选人信息

**输入**：`你最擅长什么？`

**输出**：

```text
我擅长基于 LangGraph 的企业级多 Agent 编排与 RAG 系统开发。

1. LangGraph 多 Agent 编排：独立完成 PCB 质检多 Agent 协同平台，
   5-Agent Orchestrator-Worker 模式，Orchestrator 统一调度。
   [Source: advantages.json, Chunk ID: 37cd3600037888d9]

2. RAG 混合检索：个人求职 RAG Agent，Dense + BM25 + RRF 融合，
   解决纯向量对技术专有名词召回不稳定问题。
   [Source: advantages.json, Chunk ID: 32d2115206e3ac16]
```

### 示例 2：JD 匹配分析

**输入**：`快手的AI应用开发工程师岗位，我匹配吗？`

**输出**：五段式报告（JD 核心要求 → 逐条对标 → 匹配点 → 差距项 → 面试建议），**每条结论带 Citation**。

### 示例 3：面试技术笔记查询

**输入**：`Checkpointer 持久化怎么做？`

**输出**：

```text
Checkpointer 实现跨轮次记忆和断点恢复，支持 MemorySaver（开发用内存）、
SqliteSaver（单机持久化）、PostgresSaver（生产级）。每次节点执行后，
LangGraph 自动把 State 快照写入 Checkpointer；下次调用时用 thread_id
恢复上下文。
[Source: langgraph_notes.md, Chunk ID: 1c3069c7ad2ab399]
```

## 📁 项目结构

```text
rag-interview-agent/
├── README.md
├── pyproject.toml
├── .env.example
├── standalone_embed.bat          # Milvus 启动脚本（Windows）
│
├── data/                          # 知识库数据
│   ├── resume/
│   │   └── advantages.json       # 个人优势（14 条）
│   ├── job_descriptions/
│   │   └── sample_jds.json       # 岗位 JD（7 条）
│   ├── interview_notes/           # 面试笔记（76 段）
│   │   ├── langgraph_notes.md
│   │   ├── rag_notes.md
│   │   ├── fastapi_notes.md
│   │   ├── python_notes.md
│   │   ├── milvus_notes.md
│   │   └── project_notes.md
│   └── bm25_cache/                # BM25 索引缓存（自动生成）
│
├── src/
│   ├── config.py                  # 配置单例（pydantic-settings）
│   ├── db/                        # 数据库层
│   │   ├── milvus_manager.py     # Milvus 客户端 + Schema
│   │   └── hybrid_retriever.py   # Dense + BM25 + RRF
│   ├── llm/                       # LLM 层
│   │   ├── embedder.py           # Embedding（SiliconFlow）
│   │   ├── deepseek_client.py    # LLM 客户端（同步 + 流式）
│   │   └── query_rewriter.py     # Query 改写
│   ├── graph/                     # LangGraph 编排
│   │   ├── state.py              # RAGState TypedDict
│   │   ├── nodes.py              # 5 节点函数
│   │   └── rag_graph.py          # StateGraph 组装
│   ├── api/                       # FastAPI 接口
│   │   ├── main.py               # app + lifespan
│   │   ├── routes.py             # /ask /ask/stream /health
│   │   ├── schemas.py            # Pydantic 模型
│   │   └── streaming.py          # SSE 生成器
│   ├── utils/                     # 工具层
│   │   ├── document_parser.py    # JSON/MD 解析
│   │   ├── tokenizer.py          # 中英混合分词
│   │   └── citation.py           # Citation 格式化 + 校验
│   └── web/
│       └── streamlit_app.py      # Streamlit 前端
│
├── scripts/
│   ├── ingest_data.py            # 数据入库
│   ├── test_deepseek.py          # LLM 测试
│   ├── test_hybrid_search.py     # 混合检索测试
│   ├── test_query_rewriter.py    # Query 改写测试
│   ├── test_citation.py          # Citation 测试
│   └── test_rag_graph.py         # 端到端测试
│
└── docs/
    └── architecture_evolution.md  # V1 → V2 演进文档
```

## 🛠️ 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.13 | 运行时 |
| FastAPI | 0.141 | Web 框架 |
| LangGraph | 1.2 | Agent 编排 |
| Milvus Standalone | 2.5.14 | 向量库 |
| pymilvus | 3.0.2 | Milvus 客户端 |
| rank-bm25 | 0.2.2 | BM25 算法 |
| jieba | 0.42.1 | 中文分词 |
| SiliconFlow BGE-M3 | 1024 维 | Embedding |
| DeepSeek chat | - | LLM |

## 📖 相关文档

- [架构演进：V1 → V2](docs/architecture_evolution.md)
- [面试话术](docs/interview_qa.md)（待补）

## 📄 License

MIT