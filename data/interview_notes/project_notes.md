# 项目经验笔记

## PCB 质检多 Agent 协同平台概述
项目名称：智能制造质检多 Agent 协同平台（PCB 缺陷检测子场景）。核心架构：5-Agent LangGraph 编排（Visual / Standard / RootCause / Decision / Report），实现缺陷检测→标准匹配→根因分析→处置决策→报告生成全链路。技术栈：LangGraph + FastAPI + YOLOv8 + LlamaIndex + pymilvus-lite + Postgres + Redis + Docker。数据集：KolektorSDD（6 类 PCB 缺陷：missing_part/bent_lead/crack/short_circuit/wrong_placement/contamination）。知识库：IPC-A-610G 标准 + 自造企业 SOP。对标 7 份 JD，覆盖 Agent 编排、RAG 工程化、多模态、评测体系、工业场景、私有化部署。

## 项目架构演进决策记录
PCB 智能质检 Agent 项目经历 32 轮迭代：R1-11 环境搭建（Docker/WSL2）→ R12 Milvus 排障+架构切换（pymilvus-lite）→ R13-16 依赖对齐（nullable/grpcio/multipart）→ R17-22 数据管线（PKU-Market-PCB 下载+转换+训练）→ R23 特征提取（Hook Layer 21, 256维）→ R24-25 LangGraph 编排+Visual/Standard Agent 真实化 → R26 磁盘校准 → R27 DeepSeek LLM 根因接入 → R28 结构化报告生成 → R29 报告优化 → R30 项目清理 → R31 Docker 化准备 → R32 GPU 穿透验证。核心决策：pymilvus-lite 嵌入式、VECTOR_DIM=256、YAML 内存字典（非向量 RAG）、Orchestrator-Worker 模式。

## PCB 质检 Agent State Schema
使用 Pydantic 定义 LangGraph 全局状态 PCBQualityState，包含 image_path、batch_id、visual_result、standard_result、root_cause_result、decision_result、report_result 字段。各 Agent 输出模型：VisualAgentResult（defect_type/confidence/bbox）、StandardAgentResult（is_compliant/violation_clause/severity）、RootCauseAgentResult（probable_cause/contributing_factors）、DecisionAgentResult（final_disposition/reasoning）、ReportAgentResult（summary/recommendations）。DefectType 枚举定义 6 类缺陷。严格类型定义保证 Agent 间数据传递一致性。

## 截断模型提取特征方案
YOLOv8 的 Neck 是 DAG 结构（含 Concat 多输入模块），Hook 在 predict() 内部会被静默跳过，embed API 对 numpy 输入返回 None。最终方案：物理截断模型。通过 _find_neck_layer() 遍历 self.model.model.model，找到输出通道为 256 的最后一个 C2f 层（idx=21）。用 nn.ModuleList 封装前 22 层，逐层执行并缓存中间输出 y[i]=x，遇到多输入模块从缓存取 .f 索引。100% 可控，不依赖框架内部行为。特征提取后全局平均池化 feat_map.mean(dim=(2,3)).squeeze(0) → [256]，L2 归一化。

## 推理尺寸与置信度调优
detect() 方法使用 imgsz=self.INFERENCE_IMGSZ（1280），extract_from_image() 漏传 imgsz 默认 640。3034×1586 的 PCB 图在 640 下小目标全部丢失，raw_yolo_boxes=0。修复：extract_from_image 中统一 imgsz=self.INFERENCE_IMGSZ。DISPLAY_CONF_THRESHOLD 从 0.30 降至 0.20，保留 conf=0.273 的 open_circuit 缺陷。双重保险：extractor 内部过滤 + graph 层过滤。保底逻辑：若全部低于阈值，保留最高分缺陷作为"疑似缺陷"，防止下游 Agent 收到空列表报错。推理层保持 CONF_THRESHOLD=0.1 低阈值保证召回。

## 多轮循环状态管理
Decision Agent 每轮判断：error 短路 → visual_done 检查 → db_write_count 检查 → matched_standards 存在性 → root_cause 存在性 → final_report 存在性 → END。visual_done 防止重复视觉检测，db_write_count=-1 防止死循环，matched_standards/root_cause/final_report 存在性检查防止重复调用。每轮回到 decision 统一路由。显式执行标记替代隐式数据存在性判断，彻底消除状态机歧义。

## Docker GPU 穿透验证
Windows + WSL2 + Docker Desktop 环境下，容器内使用 GPU 需满足：NVIDIA 驱动 ≥ 某版本、WSL2 内核支持 GPU passthrough、Docker Desktop 启用 GPU 支持、docker run --gpus all 或 compose deploy.resources.reservations.devices。验证方法：docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi。若容器启动即退出，可用 --entrypoint tail -f /dev/null 保持容器存活，再 docker exec 验证 torch.cuda.is_available()。PCB 项目实测：Torch 2.14.0+cu126 / CUDA True / RTX 4060 Ti。

## OpenCV 容器依赖修复
python:3.13-slim 基础镜像缺少 OpenCV 运行所需的 X11 库，报 ImportError: libxcb.so.1。修复方案：① Dockerfile 安装系统依赖 apt-get install -y libxcb1 libxext6 libxrender1 libgl1-mesa-glx libglib2.0-0 libsm6 libgomp1；② requirements.txt 改用 opencv-python-headless（无 GUI 依赖，体积小 30MB+）。后端图像处理推荐方案 ②，除非需要 cv2.imshow 窗口显示。

## Docker 镜像拉取与基础镜像对齐
国内 Docker Hub 拉取 postgres/redis 等官方镜像常超时（284s+ 无进展）。解决方案优先级：① 配置 registry-mirrors（docker.1ms.run / docker.m.daocloud.io）；② 换时段（凌晨 1-7 点）；③ 离线导入（外部设备 docker save → 拷贝 tar → docker load）；④ 手机热点。多镜像源可能互相干扰，建议只保留一个。R31 将 Dockerfile 的 FROM python:3.11-slim 升级为 FROM python:3.13-slim，与本地 Miniconda Python 3.13 环境对齐，避免"本地能跑，容器报错"的 ABI 不兼容问题。

## PowerShell 编码陷阱
PowerShell 5.x 的 Set-Content -Encoding UTF8 会写入 UTF-8 with BOM，Python 读取时可能报错。修改含中文的 .py 文件应使用 [System.IO.File]::WriteAllText(path, content, [System.Text.UTF8Encoding]::new($false)) 确保无 BOM。Get-Content 默认按系统 GBK 解码，读取 UTF-8 文件显示乱码——加 -Encoding UTF8 参数。PowerShell 5.x 的 @'...'@ Here-String 嵌入含中文的多行内容时，Set-Content -Encoding UTF8 产生 BOM + GBK 双重编码污染。修复：改用 WriteAllText + UTF8Encoding($false)。

## .dockerignore 与 requirements 分离
.dockerignore (374 B) 排除四类内容：① .env（防止 DEEPSEEK_API_KEY 泄露到镜像层）；② __pycache__/ + *.pyc（Windows 编译产物 Linux 容器不可用）；③ milvus_data.db/ + uploaded_images/（运行时数据应通过 volume 挂载）；④ archive/ + runs/（历史脚本和训练产出）。缺失该文件会导致镜像体积从数百 MB 膨胀到 GB 级，并存在密钥泄露的安全漏洞。requirements.txt 保留 CPU 版 torch 供 Docker 构建；requirements-dev.txt 追加 torch==2.14.0+cu126 供本地 GPU 开发。

## 全链路端到端验证方法论
日志节点顺序核对：Decision→Visual→Decision→DB→Decision→Standard→Decision→RootCause→Decision→Report→END。API 响应字段交叉验证：visual_done=true、db_write_count>0、defects 不翻倍、feature_vectors 长度与 defects 一致。双路径预期对照：有缺陷走完整链路，无缺陷跳过 db_store/standard/rootcause 直达 report。10 步全链路通过，无断点无死循环。诊断日志分级：graph 层确认是否调用，extractor 层确认返回值，逐层缩小问题范围。

## 环境变量泄露排查
uvicorn 启动报 OpenAIError：api_key 未设置。根因：graph.py 模块级初始化 ChatOpenAI，main.py 缺少 load_dotenv()。修复：main.py 顶部 from dotenv import load_dotenv; load_dotenv()，必须在 import graph 之前执行。--reload 模式旧进程残留会造成 /health 正常但业务 500 假象，需 Get-Process -Name python,uvicorn | Stop-Process -Force 彻底杀死残留进程。

## Docker Compose 服务依赖编排
depends_on 配合 condition: service_healthy 确保 Postgres 完全就绪后 API 才启动。Postgres healthcheck：test: ["CMD-SHELL", "pg_isready -U pcb"]，interval 10s，retries 5。API healthcheck：test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')"]，start_period 15s 给模型加载留时间。Redis healthcheck：test: ["CMD", "redis-cli", "ping"]。docker compose ps 显示 (healthy) 表示健康检查通过。孤儿容器通过 docker compose down --remove-orphans -v 清理。

## 磁盘真值校准方法论
锚点中的估算值（~xxx B）只能作为临时占位，磁盘实测永远是唯一真值来源。每阶段进入前执行 PowerShell 清查：Get-ChildItem -Path D:\smart-agent -Recurse -File | Where-Object { $_.FullName -notmatch '(datasets|runs|archive|__pycache__|milvus_data|\.git|node_modules|uploaded_images)' } | Select-Object @{N='File';E={$_.FullName.Replace('D:\smart-agent\','')}}, @{N='Size(B)';E={$_.Length}} | Format-Table -AutoSize。PowerShell 中 find -exec {} + 报 missing argument to '-exec'，改用 docker run --rm -v pcb_api_app:/app alpine sh -c "find /app -type d -name '__pycache__' -exec rm -rf {} +"。

## 简历话术：向量数据库部署与架构决策
在 Windows + WSL2 环境下部署 Milvus v2.4.17 时，通过逐层日志分析定位 4 个嵌套问题：jemalloc 动态库路径错误、etcd Raft 选举超时（WSL2 I/O 延迟）、Docker Named Volume 权限不一致、RocksDB 跨文件系统 POSIX 兼容性缺陷。最终采用 pymilvus-lite 嵌入式方案消除所有底层依赖，环境搭建时间从 2 天缩短至 5 分钟，与生产环境 API 完全兼容，迁移成本为零。面试高频追问预备：etcd 选举超时原因、Named Volume 权限问题、pymilvus-lite 生产可用性。

## 简历话术：PCB 缺陷检测模型训练与优化
基于 PKU-Market-PCB 数据集（6 类真实工业缺陷），编写 COCO→YOLOv8 格式转换管线，在 RTX 4060 Ti 上完成 YOLOv8n 微调。最终模型 mAP@50 达 0.896、Precision 0.943，单帧推理 1.2ms，满足工业质检实时性要求。针对 short 类召回率偏低（0.694）问题，通过混淆矩阵分析定位形态多变导致的漏检，为后续数据增强提供量化依据。

## 简历话术：多智能体系统架构设计与实现
基于 LangGraph 设计 Orchestrator-Worker 模式 5-Agent 协同质检系统。通过 TypedDict 定义全局共享状态 PCBQualityState，利用 add_conditional_edges 实现动态路由。Decision Agent 作为 Supervisor 依次调度 Visual→Standard→RootCause→Report，Worker 完成后强制回环汇报，实现流程完全可控与断点续传能力。State Reducer 防覆盖机制 + 节点最小返回原则解决多轮循环数据翻倍问题。

## 简历话术：工业知识库架构选型与落地
针对 PCB 质检标准检索场景，对比向量 RAG 与确定性检索方案后，采用 YAML 内存字典实现 IPC-A-610G 标准的 O(1) 精确匹配。覆盖 6 类缺陷的判定标准、物理成因及返工建议，消除 LLM 幻觉风险，检索响应时间 <1ms，较向量检索方案降低 99% 延迟。选型决策依据：查询键确定性 + 结果唯一性 → 精确匹配；查询键模糊 + 结果多样 → 向量检索。

## 简历话术：LLM 工程化集成与根因推理
将 DeepSeek API 封装为 LangGraph RootCause Agent 节点，采用模块级单例避免重复初始化。设计结构化 Prompt 模板注入缺陷列表与 IPC 标准证据链，实现基于工业知识的工程根因推理。冒烟测试验证 557 字动态分析输出，API 调用延迟 <3s，结果完全基于 IPC 标准而非通用知识。ChatOpenAI 需 streaming=True，节点用 async def + await _llm.ainvoke(prompt) 激活 on_chat_model_stream 事件。

## 简历话术：结构化质检报告自动生成
设计纯 Python f-string Markdown 模板引擎，将多 Agent 协同结果（YOLO 缺陷列表、IPC-A-610G 标准证据链、DeepSeek 根因分析）实时组装为 Markdown 质检报告。生成延迟 <1ms、零额外 API 成本，输出包含缺陷明细表（含 IPC 等级引用与返工/报废处置建议）、根因证据链及综合判定意见，较 LLM 生成方案提速 1000x 且结果完全可控可复现。