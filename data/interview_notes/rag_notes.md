# RAG 架构笔记

## 混合检索与 RRF 融合
纯向量检索对技术专有名词（如 astream_events、LangGraph）召回不稳定，因 Embedding 模型更关注语义而非精确 token。混合检索同时运行 Dense（向量）和 Sparse（BM25），两路结果用 RRF（Reciprocal Rank Fusion）融合。Dense 抓语义相似，BM25 抓关键词精确匹配，互补性强。RRF 算法：score = Σ 1/(k + rank_i)，k 通常取 60。RRF 优于加权求和的原因：Dense 是余弦相似度（0~1），BM25 是无界实数，量纲不同无法直接相加；RRF 只看排名，无需归一化，无需调参。评估指标：Recall@K、MRR。目标：Recall@5 > 90%。

## Query 改写与多路召回
Query 改写用于提升检索召回：将用户口语化描述改写为标准化术语（如"板子断了" → "PCB 开路"，"astream怎么用" → "astream_events 使用方法"）。实现要点：① 保留原始 query 一定参与检索（LLM 改写可能跑偏）；② 限制 2-3 个改写结果（多了增加成本、边际收益下降）；③ 用 Pydantic schema + JSON mode 强制结构化输出；④ 解析失败回退到 [原始 query]，不能中断链路；⑤ 温度 0.0-0.2 保证稳定。多路 query 并行检索后用 chunk_id 去重，RRF 分数择优合并。

## Citation 溯源与生成后校验
RAG 系统的核心承诺是"有据可查"，但 LLM 天然会编造引用。Citation 机制要求每条关键结论在句末标注 [Source: 文件名, Chunk ID: xxx]，Chunk ID 必须严格从 CONTEXT 复制。生成后校验：正则提取回答中的 [Source: xxx, Chunk ID: yyy]，与检索结果的 (source_file, chunk_id) 集合比对。任何非法引用触发重试，两次失败则降级为无引用回答。前端展示"总计 5 / 合法 5 / 非法 0"徽章，把可信度量化。这是 RAG 可信度的关键防线。

## Embedding 模型选型
中文 RAG 场景推荐 BGE-M3（本地可跑，中文友好，1024 维）。Milvus Collection 创建：vector(1024) + scalar fields(metadata)。相似度度量用 COSINE。检索评估构建 50 条"缺陷描述→对应标准条款"测试集，计算 Recall@3/5/10、MRR。目标 Recall@5 > 90%。混合检索 Dense + Sparse 用 RRF 融合，k 取 60。Embedding 层抽象三档 provider（openai_compatible / local_bge / dummy），换模型不影响上层 RAG 链路。在线方案用 SiliconFlow 或 OpenAI 兼容接口，离线用 sentence-transformers。

## PDF 文档解析与 Chunking 策略
质检标准 PDF 解析方案：Camelot 提取表格（质检阈值表），PyMuPDF 提取正文条款。Chunking 策略：按"条款层级"智能切片而非固定 token 数，保留 parent-child 关系。每条 chunk 添加 metadata：{standard_id, clause_no, defect_type, severity}。Chunk 大小 500-800 字，overlap 100。Markdown 按 ## 二级标题切分，每段一个 chunk。JSON 每条记录一个 chunk，字段用 \n 拼接。BM25 分词和 Embedding 编码用同一份 chunk 文本，保证 token 空间一致。

## 向量特征提取与 L2 归一化
YOLOv8 Neck 层输出 [1, 256, H, W] 特征图。全局平均池化：feat_map.mean(dim=(2,3)).squeeze(0) 得到 [256] 向量。L2 归一化：vec / np.linalg.norm(vec)，使 COSINE 相似度计算更稳定。最终 normalized.tolist() 转为 Python list，兼容 JSON 序列化和 isinstance 检查。向量数值范围约 [-0.06, 0.27]，符合单位向量特征。维度校验：db.py 中严格校验 if not isinstance(vector, list) or len(vector) != VECTOR_DIM: continue，跳过无效向量并记录 warning。

## 结构化知识库选型对比
向量 RAG 与 YAML 字典的核心差异：向量 RAG 适合非结构化文档（PDF/Markdown）的语义检索，但存在语义漂移风险；YAML 字典适合结构化标准条目的精确匹配，零幻觉但需人工结构化。PCB 质检场景中，缺陷类型是确定的枚举值（6 类），IPC 标准按缺陷类型一一对应，无需语义检索。选型决策依据：查询键确定性 + 结果唯一性 → 精确匹配；查询键模糊 + 结果多样 → 向量检索。YAML + 内存字典实现 O(1) 精确匹配，检索延迟 <1ms，准确率 100%。

## 三层评测体系与指标矩阵
Layer 1 单 Agent 评测（mAP、F1、Recall@K、MRR）；Layer 2 Pipeline 端到端评测（最终判定准确率、漏检率 FNR、误报率 FPR）；Layer 3 生产指标监控（平均处理耗时 < 12s/工单、Token 成本 < ¥0.2/工单、人工复核率 < 15%）。构建 300 条 Gold Standard 测试集（100 合格 + 150 缺陷 + 50 边界 case）。使用 LLM-as-Judge 对报告质量打分（1-5 分制）。对比实验：有 RAG vs 无 RAG，有反思 vs 无反思，量化每个优化的收益。