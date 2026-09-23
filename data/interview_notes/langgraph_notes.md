# LangGraph 核心概念笔记

## StateGraph 状态图原理
StateGraph 是 LangGraph 核心抽象，把 Agent 工作流建模为有向图。通过定义 State TypedDict 作为全局状态容器，每个节点函数读取所需字段、返回增量更新（dict），避免直接修改 state，保证状态变更可追溯。关键 API：StateGraph(StateSchema)、add_node(name, fn)、add_edge(from, to)、add_conditional_edges(from, router_fn, mapping)、set_entry_point(name)、compile()。相比 LangChain 的线性 Chain，StateGraph 天然支持循环、分支、条件跳转，是构建复杂 Agent 的标准方式。PCB 项目中 graph = build_graph() 模块级编译为 CompiledStateGraph，FastAPI 通过 graph.invoke({"image_path": ...}) 原生 dict 驱动执行。

## Checkpointer 持久化与断点续传
Checkpointer 实现跨轮次记忆和断点恢复。支持 MemorySaver（开发用内存）、SqliteSaver（单机持久化）、PostgresSaver（生产级）。核心机制：每次节点执行后 LangGraph 自动把 State 快照写入 Checkpointer；下次调用时用 thread_id 恢复上下文。PCB 项目的 State 设计天然支持断点续传：Decision Agent 作为入口，通过检查 State 中关键键（defects/matched_standards/root_cause/final_report）存在性判断当前执行进度。若服务在 Standard Agent 执行后崩溃，重启后重新 invoke 同一 State，Decision 检测到前置键已存在、后续键缺失，直接路由至对应节点继续，无需重跑前置节点。

## astream_events 细粒度事件流
astream_events 是 LangGraph 推荐的流式接口，支持 token 级实时推送。graph.astream_events(input, version="v2") 返回细粒度事件流。关键事件类型：on_chain_start（节点开始，event["metadata"]["langgraph_node"] 获取节点名）、on_chat_model_stream（LLM 逐 Token 输出，event["data"]["chunk"].content）、on_chain_end（节点结束，event["data"]["output"] 为节点返回值）。注意：若节点用 f-string 生成内容而非 LLM 调用，永远不会触发 on_chat_model_stream。需确保 ChatOpenAI 初始化时 streaming=True。配合 FastAPI SSE 实现企业级流式响应。

## Human-in-the-Loop 人机协同
通过 interrupt_before / interrupt_after 参数在指定节点前后暂停图执行，等待人工审批。典型场景：敏感操作确认、低置信度结果审核、多轮追问。实现流程：① compile(checkpointer=saver, interrupt_before=["dangerous_node"])；② 首次 invoke 到该节点前暂停，State 存盘；③ 人工审批后调用 graph.update_state(config, {"approved": True})；④ 用 graph.invoke(None, config) 从断点恢复。PCB 项目中置信度 > 0.85 自动放行，否则进入人工复核节点。这是 Agent 安全性的关键机制。

## 条件边与 Supervisor 动态路由
add_conditional_edges(source_node, router_fn, mapping) 实现动态路由。router_fn 接收 State 返回字符串 key，LangGraph 根据 mapping 字典决定下一节点。router_fn 必须是纯函数，只读 State 不做副作用。PCB 质检系统中 Decision Agent 作为 Supervisor，路由逻辑：先判断 error 短路，再判断 visual_done 标志，然后按 defects→db_store→standard→rootcause→report 顺序推进，最终 END。非法路由值兜底返回 END。条件边必须配合回边使用——Worker 完成后回边至 Decision，由 Decision 再次触发条件边，形成"调度-执行-汇报"闭环。

## Orchestrator-Worker 多 Agent 模式
LangGraph 实现多 Agent 协同的核心模式是 Orchestrator-Worker：Decision Agent 作为 Supervisor 维护全局 State，根据 State 字段变化动态路由至 Visual/Standard/RootCause/Report 四个 Worker 节点。Worker 执行完毕后通过回边 add_edge("visual", "decision") 强制回到 Decision 汇报，由 Decision 统一决定下一步。Report 作为终点节点直接连接 END，不回环。该模式保证流程完全可控，天然支持断点续传——任何节点中断后，Decision 可通过 State 键存在性判断恢复执行位置。PCB 项目 5-Agent 编排采用此模式。

## State Reducer 防覆盖机制
LangGraph StateGraph 中列表字段默认采用覆盖语义。当 Decision Agent 触发多轮循环时，Visual Agent 重复写入 defects 会导致数据丢失。解决方案：使用 Annotated[list[dict], operator.add] 实现追加语义。对于 feature_vectors 这类嵌套列表，operator.add 会导致无限拼接，需自定义 Reducer 函数，逻辑为"新值非空则替换，为空则保留"。关键 API：from typing import Annotated、import operator。配合"节点最小返回原则"——中间节点只返回其产生的增量状态，绝对不返回已存在的字段，否则 Reducer 会误触发导致数据翻倍。

## State 键存在性路由防死循环
Decision Agent 区分"未执行"与"执行后结果为空"是路由正确性的关键。若用 if not state.get("defects") 判断，空列表会被视为"未检测"，导致无限重跑 Visual Agent 形成死循环。正确做法是用键存在性判断：if "defects" not in state: return {"next_step": "visual"}。Visual Agent 无论是否检出缺陷，都必须返回 {"defects": [...]}（空列表也是有效结果），确保键被写入 State。该设计解决了空缺陷图片反复触发检测的 GraphRecursionError。显式执行标记（visual_done: bool）比隐式数据存在性判断更可靠。

## GraphRecursionError 排查
LangGraph 默认递归限制 25 步，超出抛 GraphRecursionError。PCB 项目曾因 Decision 用 if not state.get("defects") 判断导致空缺陷图片无限重跑 Visual Agent 触发该错误。排查方法：观察日志中节点执行序列是否重复循环，定位路由条件是否将"有效空结果"误判为"未执行"。修复策略：改用键存在性判断，或为 State 增加 visual_done: bool 标志位。生产环境可调高 recursion_limit 配置，但根本解法是修正路由逻辑。哨兵值模式：写入成功返回 >0，未执行返回 0，失败返回 -1，Decision 识别 -1 时自动降级跳过。

## 节点异常隔离与模块级单例
Worker 节点使用 try-except 包裹核心逻辑，捕获异常后返回 {"error": str(e), "next_step": "END"} 而非抛出。Decision Agent 在每次路由前检查 if state.get("error") 实现错误短路，立即终止流程。该设计确保单个 Agent 故障不阻塞整体工作流。模型加载等重资源操作采用模块级单例，在 graph.py 顶部初始化 _extractor = FeatureExtractor()，所有节点函数复用同一实例。若在节点函数内部初始化，每次请求都会重新加载 YOLOv8 权重（约 3 秒），GPU 显存也会反复分配释放。单例后节点开销仅剩推理时间（~12ms）。

## 可观测性与 LangSmith 接入
LangSmith 是 LangChain 官方的 Agent 可观测性平台。零侵入接入：设置环境变量 LANGCHAIN_TRACING_V2=true、LANGCHAIN_ENDPOINT、LANGCHAIN_API_KEY、LANGCHAIN_PROJECT。代码零修改，LangChain 底层自动 hook 所有 Runnable 调用。config 中 metadata={"trace_id": trace_id} 透传请求级标识。面板展示完整 DAG 拓扑、每节点耗时和 Token 用量。Web UI 可视化整条链路，支持按 thread_id 回溯、按用户筛选、对比不同版本的 Prompt 效果。生产环境排查 Agent 问题必备。