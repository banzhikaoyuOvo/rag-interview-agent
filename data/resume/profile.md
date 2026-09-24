# 俞晓兴 · AI 应用开发工程师

>  AI 应用开发 | LangGraph · RAG · FastAPI · Milvus | 26 届应届毕业
## 联系方式

- **GitHub**：[banzhikaoyuOvo](https://github.com/banzhikaoyuOvo)
- **邮箱**：3341428887@qq.com
- **目标岗位**：AI 应用开发工程师
- **目标城市**：杭州 / 深圳 / 广州 / 江西 / 上海 / 南京

## 一句话简介

26 届电子信息工程应届毕业生，3 个月内独立完成 3 个 AI 项目并全部开源。擅长 LangGraph 多 Agent 编排、RAG 检索增强、FastAPI 异步服务，能独立完成从架构设计、代码实现、调试排错到 Docker 部署的完整工程流程。

## 开源项目

### 1. 个人求职 RAG Agent（[GitHub](https://github.com/banzhikaoyuOvo/rag-interview-agent)）

企业级 RAG 求职知识库助手，Parking-YOLO V1 的架构升级版。技术栈：FastAPI + LangGraph 5 节点编排 + Milvus Standalone + 应用层 BM25 + RRF 融合 + Citation 溯源 + Streamlit SSE 流式前端。

**核心亮点**：
- **混合检索**：Dense（BGE-M3 1024 维）+ BM25 双路召回 + RRF 融合，解决纯向量对技术专有名词（`astream_events`）召回不稳定问题
- **Query 改写 + 多路召回**：LLM 将口语化 query 改写成 2-3 个专业检索词，保留原始 query 参与检索
- **Citation 生成后校验**：正则提取 `[Source: xxx, Chunk ID]` 与检索结果比对，非法引用自动重试
- **跨集合检索**：支持 `target_collection: str | list[str]`，用于 JD 匹配分析（JD 库 + 简历库双路检索）
- **SYSTEM_PROMPT 7 条规则**：含组合型问题（自我介绍）和多段式匹配分析

**数据规模**：3 个 Collection，97 条数据（简历 14 + JD 7 + 笔记 76）

### 2. PCB 智能质检多 Agent 协同平台（[GitHub](https://github.com/banzhikaoyuOvo/pcb-quality-agent)）

面向工业质检场景的 PCB 缺陷检测系统。技术栈：LangGraph 5-Agent 编排 + FastAPI + YOLOv8n + Milvus Lite + DeepSeek + Docker Compose + GPU 穿透。

**核心亮点**：
- **5-Agent Orchestrator-Worker**：Decision 作为 Supervisor 动态路由 Visual / Standard / RootCause / Report 四个 Worker，完成后强制回环汇报
- **YOLOv8n 微调**：PKU-Market-PCB 数据集（6 类工业缺陷），mAP@50 达 **0.896**，Precision **0.943**，单帧推理 **1.2ms**（RTX 4060 Ti）
- **特征提取**：从 YOLOv8n Neck 末端 Hook Layer 21 截取 **256 维特征**，L2 归一化后存 Milvus Lite
- **IPC-A-610G 知识库**：YAML 内存字典 O(1) 精确匹配，检索延迟 **<1ms**，零幻觉
- **SSE 流式反馈**：`astream_events` 逐 Token 推送报告生成，前端打字机效果

**性能指标**：端到端单次质检 3-5s，支持 Docker 一键部署

### 3. Parking-YOLO 多模态智慧停车系统（毕业设计 · [GitHub](https://github.com/banzhikaoyuOvo)）

基于 YOLO 与大模型的多模态智慧停车感知与导航系统。技术栈：YOLOv8 + Flask + RAG + DeepSeek + SSE + 向量数据库。

**核心亮点**：
- **"事实与表达分离"三层解耦架构**：YOLO 提取车位事实 + RAG 检索导航规则 + LLM 融合生成，彻底根除大模型在空间导航中的幻觉问题
- **高精度 CV 感知**：独立完成 12,417 张图像清洗与格式转换，YOLOv8n 车位检测 + 像素级分割（10 FPS）
- **安全网关与状态缓存**：设计 `get_parking_advice_safe` 安全包装器，集成 JSON 清洗与字段校验，结构化输出解析成功率提升至 **99%**

**项目成果**：稳定运行超 200 小时无故障，LLM 导航准确率 100%（基于事实注入）

## 技能栈

| 分类 | 技术 |
|------|------|
| **编程语言** | Python（主力） |
| **Web 框架** | FastAPI（异步 ASGI）、Flask（同步 WSGI） |
| **Agent 编排** | LangGraph（StateGraph / 条件边 / Checkpointer / Send API / Subgraph） |
| **LLM 框架** | LangChain Core、DeepSeek API、ChatOpenAI |
| **RAG 技术** | Hybrid Search（Dense + BM25）、RRF、Query Rewriting、Citation、Embedding（BGE-M3） |
| **向量库** | Milvus Standalone、pymilvus-lite（嵌入式） |
| **深度学习** | PyTorch、YOLOv8n（训练 / 微调 / 特征提取） |
| **数据处理** | pydantic / pydantic-settings、jieba、rank-bm25、httpx、asyncio |
| **部署运维** | Docker / Docker Compose、GPU 穿透、SSE 流式、Caddy |
| **开发工具** | Git、pytest、logging、Prompt Engineering |

## 教育背景

- **本科**：南昌大学科学技术学院 · 电子信息工程（人工智能方向）· 2024-2026
- **专业排名**：前 25%
- **核心课程**：机器学习、深度学习、机器视觉基础、人工智能算法基础、Python 数据分析技术、信息安全技术、算法设计与分析、数字图像处理、NLP 自然语言处理
- **毕业设计**：基于 YOLO 与大模型的多模态智慧停车感知与导航系统（2026.01 - 2026.05，负责人）

## 软实力

- **退役大学生士兵**：2 年服役经历，培养了极强的执行力、抗压能力和纪律性
- **快速学习能力**： AI 应用开发，3 个月内独立完成 3 个完整 AI 项目
- **AI 辅助开发能力**：熟练使用 AI 编程助手高效开发，核心能力是"知道该问什么、能判断答案对不对、能独立排错"
- **工程素养**：沉淀 20+ 典型工程问题的排查经验，涵盖 pymilvus 参数改名、Milvus Lite Windows 兼容、BM25 分词不一致、PowerShell BOM、Docker 构建优化等

## 求职意向

- **目标岗位**：AI 应用开发工程师 / RAG 工程师 / Agent 开发工程师
- **期望薪资**：7-13K 左右
- **入职时间**：随时
- **工作地点**：杭州 / 深圳 / 广州 / 江西 / 上海 / 南京