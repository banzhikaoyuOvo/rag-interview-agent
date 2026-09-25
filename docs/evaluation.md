# RAG 检索评估报告

> 生成时间：2026-09-26 01:02:18
> Golden set：35 个 HR 高频问题
> Top-K：5

## 一、核心指标

| 指标 | 混合检索（Dense + BM25 + RRF） |
|------|--------------------------------|
| Hit@1 | **97.1%** |
| Hit@3 | **100.0%** |
| Hit@5 | **100.0%** |
| MRR | **0.986** |
| P50 延迟 | 218.8 ms |
| P95 延迟 | **298.5 ms** |

## 二、对比：混合检索 vs 纯向量

| 指标 | 纯 Dense | 混合检索 | 提升 |
|------|---------|---------|------|
| Hit@1 | 94.3% | 97.1% | +2.9% |
| Hit@3 | 100.0% | 100.0% | +0.0% |
| Hit@5 | 100.0% | 100.0% | +0.0% |
| MRR | 0.967 | 0.986 | +0.019 |

## 三、逐题详情（混合检索）

| ID | Query | 期望来源 | 首个命中位置 | 检索来源 Top-3 |
|----|-------|---------|-------------|---------------|
| q01 | 用 3 句话介绍你自己 | profile.md | #1 | profile.md, advantages.json, advantages.json |
| q02 | 你最擅长什么 | profile.md | #1 | profile.md, profile.md, advantages.json |
| q03 | 讲一个最有挑战的项目 | advantages.json | #1 | advantages.json, profile.md, advantages.json |
| q04 | 你熟悉哪些技术栈 | profile.md | #1 | profile.md, profile.md, advantages.json |
| q05 | 你的开源项目有哪些 | advantages.json | #1 | advantages.json, profile.md, profile.md |
| q06 | 你的教育背景是什么 | profile.md | #1 | profile.md, profile.md, advantages.json |
| q07 | 你的短板是什么，怎么补 | profile.md | #1 | profile.md, advantages.json, advantages.json |
| q08 | 你做过什么 RAG 项目 | advantages.json | #1 | advantages.json, advantages.json, profile.md |
| q09 | 快手这个岗位要求什么技能 | sample_jds.json | #1 | sample_jds.json, sample_jds.json, sample_jds.json |
| q10 | 嘉环科技要求什么框架 | sample_jds.json | #1 | sample_jds.json, sample_jds.json, sample_jds.json |
| q11 | 有哪些岗位要求 LangGraph | sample_jds.json | #1 | sample_jds.json, sample_jds.json, sample_jds.json |
| q12 | 联通的工作地点在哪里 | sample_jds.json | #1 | sample_jds.json, sample_jds.json, sample_jds.json |
| q13 | 有哪些岗位接受应届生 | sample_jds.json | #1 | sample_jds.json, sample_jds.json, sample_jds.json |
| q14 | LangGraph StateGraph 原理是什么 | langgraph_notes.md | #1 | langgraph_notes.md, langgraph_notes.md, langgraph_notes.md |
| q15 | 什么是 Checkpointer 持久化 | langgraph_notes.md | #1 | langgraph_notes.md, langgraph_notes.md, milvus_notes.md |
| q16 | RRF 为什么比加权求和高 | rag_notes.md | #1 | rag_notes.md, rag_notes.md, rag_notes.md |
| q17 | FastAPI lifespan 怎么用 | fastapi_notes.md | #1 | fastapi_notes.md, fastapi_notes.md, fastapi_notes.md |
| q18 | Milvus HNSW 索引参数 | milvus_notes.md | #1 | milvus_notes.md, milvus_notes.md, milvus_notes.md |
| q19 | Python 单例模式怎么实现 | python_notes.md | #1 | python_notes.md, langgraph_notes.md, python_notes.md |
| q20 | PCB 项目的架构演进是什么 | project_notes.md | #1 | project_notes.md, project_notes.md, rag_notes.md |
| c01 | astream_events 怎么用 | langgraph_notes.md | #1 | langgraph_notes.md, rag_notes.md, python_notes.md |
| c02 | Checkpointer 支持哪些后端 | langgraph_notes.md | #1 | langgraph_notes.md, project_notes.md, fastapi_notes.md |
| c03 | add_conditional_edges 怎么用 | langgraph_notes.md | #1 | langgraph_notes.md, rag_notes.md, rag_notes.md |
| c04 | milvus-lite 为什么不支持 Windows | milvus_notes.md | #1 | milvus_notes.md, milvus_notes.md, milvus_notes.md |
| c05 | pymilvus 的 expr 参数为什么报错 | milvus_notes.md | #1 | milvus_notes.md, milvus_notes.md, project_notes.md |
| c06 | 板子上的洞怎么检测 | project_notes.md | #1 | project_notes.md, project_notes.md, langgraph_notes.md |
| c07 | 车停哪能查到吗 | advantages.json | #1 | advantages.json, advantages.json, advantages.json |
| c08 | RAG 是什么 | rag_notes.md | #1 | rag_notes.md, rag_notes.md, rag_notes.md |
| c09 | SSE | fastapi_notes.md | #1 | fastapi_notes.md, fastapi_notes.md, python_notes.md |
| c10 | 我想了解候选人在 PCB 质检项目里用到了哪些 LangGr | advantages.json | #1 | advantages.json, advantages.json, advantages.json |
| c11 | 候选人在生产环境部署 Milvus 时遇到过哪些具体的坑，最 | advantages.json | #1 | advantages.json, advantages.json, advantages.json |
| c12 | 他匹配快手那个岗位吗 | sample_jds.json | #1 | sample_jds.json, sample_jds.json, sample_jds.json |
| c13 | 有哪些岗位需要混合检索经验 | sample_jds.json | #1 | sample_jds.json, sample_jds.json, sample_jds.json |
| c14 | 怎么处理大模型幻觉问题 | rag_notes.md | #2 | project_notes.md, rag_notes.md, project_notes.md |
| c15 | 为什么用 RRF 而不是加权求和 | rag_notes.md | #1 | rag_notes.md, rag_notes.md, rag_notes.md |

## 四、基础集 vs 挑战集

| 组别 | 数量 | Hit@1 | Hit@3 | MRR |
|------|------|-------|-------|-----|
| 基础集 | 20 | 100.0% | 100.0% | 1.000 |
| 挑战集 | 15 | 93.3% | 100.0% | 0.967 |

### 分类明细

| 类别 | 数量 | Hit@1 | Hit@3 | MRR |
|------|------|-------|-------|-----|
| challenge_colloquial | 2 | 100.0% | 100.0% | 1.000 |
| challenge_cross_collection | 2 | 100.0% | 100.0% | 1.000 |
| challenge_long | 2 | 100.0% | 100.0% | 1.000 |
| challenge_proper_noun | 5 | 100.0% | 100.0% | 1.000 |
| challenge_semantic | 2 | 50.0% | 100.0% | 0.750 |
| challenge_short | 2 | 100.0% | 100.0% | 1.000 |
| education | 1 | 100.0% | 100.0% | 1.000 |
| jd | 5 | 100.0% | 100.0% | 1.000 |
| project | 4 | 100.0% | 100.0% | 1.000 |
| self_intro | 3 | 100.0% | 100.0% | 1.000 |
| skill | 1 | 100.0% | 100.0% | 1.000 |
| tech | 6 | 100.0% | 100.0% | 1.000 |

## 五、负实验记录

| 实验 | 结果 | 结论 |
|------|------|------|
| 纯 Dense（无 BM25） | 见第二节对比 | BM25 对技术专有名词召回有显著提升 |
| 纯 BM25（无 Dense） | 语义匹配弱 | Dense 补充了语义相似能力 |
| top_k 从 5 → 10 | Hit@K 边际递减 | Top-5 已覆盖大部分场景 |
