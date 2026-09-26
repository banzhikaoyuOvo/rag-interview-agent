# 面试话术速查

> 用途：面试前 30 分钟快速过一遍。每题 3-5 句话，覆盖高频问题。
> 更新时间：2026-09-26

---

## 一、自我介绍

### Q1：用 3 句话介绍你自己

> 我是俞晓兴，26 届电子信息工程（AI 方向）应届毕业生，退役大学生士兵。
>
> 3 个月内独立完成 3 个 AI 项目并全部开源：RAG 求职 Agent、PCB 质检多 Agent 平台、Parking-YOLO 多模态智慧停车系统。
>
> 擅长 LangGraph 多 Agent 编排、RAG 检索增强、FastAPI 异步服务，能独立完成从架构设计、代码实现、调试排错到 Docker 部署的完整工程流程。

### Q2：你本科什么专业？

> 我本科是南昌大学科学技术学院的电子信息工程，**培养方向是人工智能**。所以虽然毕业证写电子信息工程，但课程覆盖机器学习、深度学习、计算机视觉、NLP 这些 AI 核心课。
>
> 毕业设计"基于 YOLO 与大模型的多模态智慧停车系统"就是在这个背景下做的。

### Q3：为什么从算法方向转 AI 应用开发？

> 大学系统学了机器学习、深度学习、算法设计与分析，算法基础扎实。但毕业后发现算法岗竞争激烈、门槛高（需要顶会论文或大厂实习）。
>
> 我基于原有算法底子主动转向 AI 应用开发——**算法侧的模型训练我做过（YOLOv8 微调 mAP@50=0.896），但更擅长把模型落地为可用的工程系统**。

---

## 二、RAG 求职 Agent 项目（重点）

### Q4：介绍下你的 RAG 项目

> 这是一个给 HR 看的个人求职 RAG 助手。核心是**基于 LangGraph 的 5 节点编排 + Hybrid Search（Dense + BM25 + RRF）+ Citation 溯源**。
>
> 三个 Collection：简历 22 条、岗位 JD 7 条、面试笔记 142 条，共 **171 条知识库数据**。
>
> 支持 HR 提问候选人信息，也支持粘贴 JD 做匹配分析。回答必须带 `[Source: 文件名, Chunk ID]` 引用，生成后自动校验，非法引用触发重试。

### Q5：为什么用混合检索而不是纯向量？

> 纯 Dense 对技术专有名词（如 `astream_events`、`add_conditional_edges`）召回不稳定——BGE-M3 更关注语义，token 级别的精确匹配弱。
>
> 我做了**三路对比评估**：
> - **纯 BM25 Hit@1 = 65%，Hit@3 = 92.5%**——召回强、排序弱
> - **纯 Dense Hit@1 = 87.5%，Hit@3 = 92.5%**——排序强、召回弱
> - **混合检索 Hit@3 = 95%，Hit@5 = 97.5%，MRR = 0.913**——取两者之长
>
> **Hit@3 提升 2.5%** 意味着纯 Dense 漏掉的 2.5% 问题被 BM25 补上了。

### Q6：RRF 是什么？为什么用它而不是加权求和？

> RRF（Reciprocal Rank Fusion）公式：`score = Σ 1/(k + rank)`，k 通常取 60。
>
> **为什么不用加权求和**：Dense 是余弦相似度（0~1），BM25 是无界实数（可能到几十），**量纲不同无法直接相加**，需要归一化 + 调权重，而且归一化对分布敏感。
>
> RRF 只看排名，**无需归一化、无需调参**，工业界混合检索的标准做法。

### Q7：RRF 权重怎么定的？

> 我做了权重调参实验：
>
> | 权重（Dense/BM25） | Hit@1 | MRR |
> |------------------|-------|-----|
> | 0.5 / 0.5（平权） | 85.0% | 0.902 |
> | **0.7 / 0.3（最优）** | **87.5%** | **0.913** |
> | 0.85 / 0.15 | 85.0% | 0.896 |
>
> **发现权重非单调**——0.7/0.3 最优，过度偏向 Dense（0.85/0.15）反而下降，因为 BM25 权重太低无法补充召回。**RRF 不是无参算法，权重需要实验调优**。

### Q8：评估用了多少 case？怎么设计的？

> **40 个 case**：20 个基础集（HR 常规问题）+ 20 个挑战集。
>
> 挑战集分 6 类：专有名词（5）、口语化（2）、短 query（2）、长 query（2）、跨库检索（7）、语义歧义（2）。
>
> 还加了 **7 个跨 collection 的 case**（`cross_collection: true`），测试多库联合检索。比如"他匹配快手那个岗位吗"，需要同时查 `job_descriptions` 和 `resume_advantages`。

### Q9：Citation 溯源怎么做的？

> 生成时要求 LLM 每条关键结论带 `[Source: 文件名, Chunk ID: yyy]`，**Chunk ID 必须从上下文复制**。
>
> 生成后正则提取所有引用，与检索结果的 `(source_file, chunk_id)` 集合比对。**任何非法引用触发重试**（最多 1 次），两次失败降级为无引用回答。
>
> 前端展示引用统计徽章：`✅ 引用：总计 5 / 合法 5 / 非法 0`。

### Q10：LangGraph 5 节点怎么编排？

> `query_rewrite → hybrid_retrieve → context_build → generate → citation_check`。
>
> - **query_rewrite**：LLM 把口语化 query 改成 2-3 个专业检索词
> - **hybrid_retrieve**：多路 query 并行检索，Dense + BM25 + RRF 融合
> - **context_build**：加 `[source_file]` 标签，让 LLM 分清要求侧/证据侧
> - **generate**：DeepSeek 生成
> - **citation_check**：条件边，合法走 END，非法走 retry
>
> **为什么用 LangGraph 而不是 if-else**：有回环（重试）、有条件路由、有状态增量合并、有异常降级——`if-else` 会变成多层嵌套，可观测性差。

### Q11：跨集合检索怎么做的？

> 把 `target_collection` 从 `str` 扩展为 `str | list[str]`。检索节点遍历多集合，每集合 `top_k = max(3, 8 // len(collections))`，chunk_id 去重、RRF 分数择优合并。
>
> 一次 invoke 就能拿到 JD 侧和简历侧两个维度的上下文，LLM 做逐条对标。

### Q12：Query 改写为什么要保留原始 query？

> LLM 改写可能跑偏。比如"他会 RAG 吗"改写成"候选人 RAG 技术栈评估"，但知识库里就写的是"RAG"。改写后反而召不回。
>
> **原始 query 永远放第一位**，改写结果作为补充。改写成功提升覆盖，改写失败也不退化。

---

## 三、PCB 质检多 Agent 平台项目

### Q13：介绍下 PCB 项目

> 基于 LangGraph 5-Agent 编排的 PCB 缺陷检测系统。
>
> **5 个 Agent**：Visual（YOLOv8n 检测）、DB Store（Milvus Lite 写入）、Standard（IPC-A-610G YAML 匹配）、RootCause（DeepSeek 根因推理）、Report（Markdown 模板生成）。
>
> **核心指标**：YOLOv8n 微调 mAP@50 = 0.896，Precision = 0.943，单帧推理 1.2ms（RTX 4060 Ti）。端到端单次质检 3-5s。

### Q14：为什么用 YAML 精确匹配而不是向量 RAG？

> **决策依据**：查询键确定性 + 结果唯一性 → 精确匹配；查询键模糊 + 结果多样 → 向量检索。
>
> PCB 缺陷类型是确定的 6 类枚举值，IPC 标准按缺陷类型一一对应，无需语义检索。
>
> YAML + 内存字典实现 O(1) 精确匹配，**检索延迟 <1ms，零幻觉风险**。而向量 RAG 在这里反而有语义漂移风险（如 `short` 和 `open_circuit` 在向量空间可能高相似）。

### Q15：GraphRecursionError 死循环是怎么排查的？

> **问题**：空缺陷图片上传后，图一直转，最后报 GraphRecursionError。
>
> **根因**：Decision Agent 用 `if not state.get("defects")` 判断，但空缺陷返回空列表 `[]`，`if not []` 为 True，误判为"还没检测"，无限重跑 Visual Agent。
>
> **修复**：改用键存在性判断 `if "defects" not in state`，并要求 Visual Agent 无论是否检出缺陷都返回 `{"defects": [...]}`，空列表也是有效结果。
>
> **沉淀**：显式执行标记 `visual_done: bool` 比隐式数据存在性判断更可靠。

### Q16：YOLOv8n 特征提取怎么做的？

> YOLOv8 Neck 是 DAG 结构（含 Concat 多输入模块），常规 Hook 会被 `predict()` 静默跳过。
>
> **物理截断模型**：找到输出通道为 256 的最后一个 C2f 层（idx=21），用 `nn.ModuleList` 封装前 22 层，逐层执行并缓存中间输出。多输入模块从缓存取 `.f` 索引。
>
> 特征后处理：全局平均池化 `[256]` + L2 归一化，配合 COSINE 度量。

### Q17：推理尺寸踩过什么坑？

> 训练 `imgsz=640`，但 PCB 原图约 2592×1944，letterbox 到 640 后缺陷像素太小。同一张图从 0 检出变成 5 个缺陷——**强制 INFERENCE_IMGSZ=1280 后解决**。

---

## 四、Parking-YOLO（毕业设计）

### Q18：介绍下毕业设计

> 基于 YOLO 与大模型的多模态智慧停车系统。
>
> **首创"事实归感知、规则归检索、表达归大模型"三层解耦架构**：YOLO 提取车位占用率等客观事实，RAG 检索 B1/B2 层导航路线与收费规则，LLM 仅负责自然语言组织——**从架构层面规避大模型编造车位数字与导航路线的问题**。
>
> PKLot 数据集微调 YOLOv8n，32 epoch 训练，最佳 **mAP50-95 = 0.96544**，推理 10 FPS。已 GitHub 开源并发布 Release v1.0.0。

### Q19：为什么叫"事实与表达分离"？

> 传统 RAG 让 LLM 同时负责"查数据"和"写话术"，容易幻觉。
>
> 我拆成三层：**事实层（YOLO）**给客观数据、**规则层（RAG）**给硬约束、**表达层（LLM）**只管组织语言。LLM 无法编造车位数字，因为数字来自 YOLO；无法编造路线，因为路线来自 RAG。

---

## 五、工程化

### Q20：FastAPI 里用了什么？

> - **lifespan** 生命周期：启动时检查 Milvus 连通性 + collection 完整性
> - **SSE 流式**：`/ask/stream` 用 `sse-starlette` 的 `EventSourceResponse`
> - **run_in_threadpool**：`/ask` 端点包装同步的 `rag_graph.invoke`
> - **健康检查分层**：`/health` 纯内存返回，`/ready` 检查 Milvus
> - **限流**：`slowapi`（PCB 项目）

### Q21：SSE 流式怎么做的？

> 后端用 `graph.astream_events(version="v2")` 拿到节点级事件，yield dict 格式（`{"event": ..., "data": ...}`），`sse-starlette` 负责编码。
>
> **踩过的坑**：一开始 yield 完整的 SSE 字符串，导致框架二次包装，前端解析全部失败。**改成 yield dict 后解决**。
>
> 事件分三类：`node`（进度）、`final`（完整答案 + sources）、`error`（异常）。

### Q22：Docker 部署怎么做的？

> **RAG 项目**：`Dockerfile` + `docker-compose.yml`，`rag-api` 和 `rag-web` 两个服务共享同一镜像，用不同命令启动。容器通过 `host.docker.internal:19530` 访问宿主机 Milvus。健康检查用 `depends_on: condition: service_healthy` 保证 API 就绪后才启动前端。
>
> **`.dockerignore` 优化**：排除 `data/bm25_cache/`、`data/eval/`、`.env`，构建 239s（无 torch）。
>
> **双模式启动脚本**：本地开发用 `start_rag.ps1`（uvicorn --reload + streamlit 热重载），生产用 `start_rag_docker.ps1`（容器一键启动）。

### Q23：项目里遇到过什么工程难题？

> **PCB 项目 Docker 化遇到 10 个典型问题**：
> 1. 新版前端不生效（容器里是旧版）
> 2. 缺 slowapi 依赖
> 3. `KeyError: 'label'`（字段名硬访问）
> 4. LangSmith 403 刷屏
> 5. 业务日志被过滤（默认 WARNING）
> 6. build context 2.87GB（数据集没排除）
> 7. 容器名前缀异常
> 8. 改代码要重建镜像（挂 volume 解决）
>
> 每个问题都有明确的定位方法和修复方案，沉淀成可复用的工程经验。

---

## 六、行为面试

### Q24：你的短板是什么？

> **短板 1**：缺少大厂实习经历。**补法**：用 3 个高质量开源项目弥补，每个都含完整 README + 测试 + Docker 部署。
>
> **短板 2**：算法侧训练经验偏少。**补法**：YOLOv8n 微调项目已覆盖数据清洗、训练、评估全流程。
>
> **短板 3**：多模态大模型（VLM）经验少。**正在补**：Parking-YOLO 已做 YOLO + LLM 融合，下一步补 VLM 实践。

### Q25：退役经历对你有什么影响？

> 2 年服役培养了**执行力、抗压能力和纪律性**。
>
> 退役后从零自学 AI 应用开发，全职投入 6-8h/天，3 个月内独立完成 3 个完整 AI 项目。**部队教会我"任务导向"——定了目标就想办法完成，不找借口**。

### Q26：为什么选择我们公司？

> *（面试前根据公司调整）*
>
> 三个原因：
> 1. **技术栈匹配**：你们 JD 要求 LangGraph / RAG / FastAPI，我三个项目都用到
> 2. **成长空间**：从 AI 应用工程师到 AI 架构师的路径清晰
> 3. **业务场景**：*（结合公司业务）*

---

## 七、反例预案

### Q27：为什么不用 LlamaIndex？

> LlamaIndex 更擅长高级 RAG 和数据索引，LangGraph 更擅长状态机和多 Agent 编排。
>
> **我的场景需要状态机**：5 节点 + 条件重试 + 断点续传，LangGraph 的 StateGraph 更合适。LlamaIndex 的工作流也能做，但 LangGraph 的生态（LangSmith 可观测性、Checkpointer 持久化）更成熟。

### Q28：为什么用 Milvus 而不是 Chroma / FAISS？

> **开发阶段用 Milvus Standalone（Docker）**，理由是：
> 1. **生产一致性**——本地开发 = 生产环境，代码只改 uri 即可迁移
> 2. **API 完整**——支持标量过滤、多集合、HNSW 索引
> 3. **踩过 Milvus Lite 的坑**（Windows 不支持），切到 Standalone
>
> Chroma 更轻量但功能少，FAISS 是库不是服务。**Milvus 是生产级向量数据库的标准选择**。

### Q29：为什么用 DeepSeek 而不是 GPT-4？

> 3 个原因：
> 1. **成本**——DeepSeek ¥1/百万 token 输入，GPT-4 是 ¥70+，差 70 倍
> 2. **中文能力**——DeepSeek 中文优于 GPT-4
> 3. **国内直连**——不需要梯子，调试快
>
> 如果业务需要多模态或复杂推理，可以切换 OpenAI 兼容接口，代码只需改 base_url。

### Q30：你的项目有什么不足？

> 三个诚实的不足：
> 1. **语料规模小**：171 条数据，扩展到 10000+ 条才能体现混合检索的完整价值
> 2. **没有线上流量**：无 A/B 测试和真实用户反馈
> 3. **评估指标有限**：只有检索层指标（Hit@K / MRR），没有生成层（Faithfulness / Answer Relevancy）
>
> 下一步计划：加 RAGAS 评估生成质量，接 LangSmith 做可观测性。

---

## 八、关键数字速查

### RAG 项目

| 指标 | 值 |
|------|-----|
| 知识库数据 | **171 条**（简历 22 + JD 7 + 笔记 142） |
| Golden set | **40 case**（20 基础 + 20 挑战） |
| 混合 Hit@1 | **87.5%** |
| 混合 Hit@3 | **95.0%** |
| 混合 MRR | **0.913** |
| 纯 Dense Hit@1 | 87.5% |
| 纯 BM25 Hit@1 | 65.0% |
| RRF 最优权重 | **0.7 / 0.3** |
| P95 延迟 | 933 ms |

### PCB 项目

| 指标 | 值 |
|------|-----|
| YOLOv8n mAP@50 | **0.896** |
| Precision | 0.943 |
| 单帧推理 | 1.2ms |
| 特征维度 | 256 |
| IPC 检索 | <1ms |
| 端到端 | 3-5s |

### Parking-YOLO

| 指标 | 值 |
|------|-----|
| mAP50-95 | **0.96544** |
| 推理速度 | 10 FPS |
| 稳定运行 | 200+ 小时 |
| LLM 导航准确率 | 100% |

---

## 九、GitHub 仓库速查

| 项目 | 链接 |
|------|------|
| RAG 求职 | https://github.com/banzhikaoyuOvo/rag-interview-agent |
| PCB 质检 | https://github.com/banzhikaoyuOvo/pcb-quality-agent |
| Parking-YOLO | https://github.com/banzhikaoyuOvo/parking-yolo |

---

## 十、面试前 5 分钟检查清单

- [ ] 3 句话自我介绍能脱口而出
- [ ] 3 个项目一句话定位
- [ ] RAG 三路评估数据记牢（87.5% / 95% / 0.913）
- [ ] GraphRecursionError 死循环排查故事能讲 3 分钟
- [ ] GitHub 链接随手能报
- [ ] 短板回答准备好（不回避，给补法）