# Python 笔记

## TypedDict 与 Pydantic 职责边界
LangGraph 共享状态用 TypedDict 定义（state.py），运行时就是 dict，支持 partial update（节点返回增量 dict 自动 merge）。FastAPI 接口层用 Pydantic BaseModel（schemas.py）做请求/响应校验和序列化。两者严禁混用——TypedDict 不可实例化，Pydantic 模型不支持 partial update。PCB 项目曾因 schemas.py 残留旧版 PCBQualityState (BaseModel) 与 state.py (TypedDict) 同名冲突，导致 ImportError。修复方案：schemas.py 仅保留 DefectType 枚举，State 定义唯一来源为 state.py。此架构兼顾性能与安全。

## asyncio 混合异步策略
同步阻塞库（PyTorch/YOLO、pymilvus-lite）使用 asyncio.to_thread() 卸载到线程池，避免霸占事件循环。原生异步库（LangChain ChatOpenAI 底层 httpx）直接使用 await ainvoke()。asyncio.to_thread() 是 Python 3.9+ 推荐写法，替代 loop.run_in_executor()。PCB 项目全链路异步化：visual_agent 用 await asyncio.to_thread(self.extract_from_image, image_path)；db_store_agent 用 await asyncio.to_thread(upsert_defect_features, ...)；rootcause_agent 用 await analyze_root_cause(...)；report_agent 用 await _llm.ainvoke(prompt)。高并发下 SSE 推送零卡顿。

## 模块级单例实现
StandardKnowledgeBase 通过 __new__ 控制实例创建：_instance 类变量缓存首次创建的实例，后续调用返回同一对象。__init__ 需配合 _initialized 标志防止重复初始化。模块级 kb = StandardKnowledgeBase() 在 import 时执行，所有引用 from app.knowledge_base import kb 的模块共享同一实例。该模式确保重资源（YAML 解析、Milvus 连接、YOLOv8 权重）仅初始化一次。MilvusDB 同样模式。若在节点函数内部初始化，每次请求都会重新加载 YOLOv8 权重（约 3 秒），单例后节点开销仅剩推理时间（~12ms）。

## 环境变量与 .env 加载顺序
root_cause_agent.py 使用 load_dotenv() 在模块顶层加载 .env 文件，os.getenv("DEEPSEEK_API_KEY") 读取密钥。graph.py 模块级初始化 _llm = ChatOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"))。若 main.py 在 from app.graph import graph 之前没有 load_dotenv()，则环境变量未注入，ChatOpenAI 初始化抛 OpenAIError。修复：main.py 顶部 from dotenv import load_dotenv; load_dotenv() 必须在所有 app 模块 import 之前执行。.env 必须加入 .gitignore 和 .dockerignore，禁止提交到仓库或打入镜像。

## 确定性 ID 与幂等性设计
使用 hashlib.md5(f"{image_path}__defect_{index}".encode()).hexdigest()[:16] 生成确定性 ID。取前 16 位十六进制，碰撞概率极低。同一图片重复检测时 ID 不变，Milvus upsert 自动覆盖旧数据，实现业务层幂等性。相比随机 UUID，避免了重复向量污染检索库。配合指数退避重试（1s/2s/4s），解决 RocksDB 文件锁冲突。defect_id 由 hash 生成确保相同图片相同位置的缺陷产生相同 ID。该设计支持幂等写入，服务重启后重新处理同一图片不会产生重复向量。

## 指数退避重试机制
_upsert_with_retry(data, max_retries=3)。for attempt in range(max_retries)：成功 return True，失败 wait_time = 2 ** attempt（1s/2s/4s），time.sleep(wait_time) 后重试。最终失败 return False 并记录 error。纯手写，无需 tenacity 库。解决 Milvus Lite 在容器内偶发的 RocksDB 文件锁冲突和 I/O 抖动。哨兵值模式：写入成功返回 >0，未执行返回 0，失败返回 -1，Decision Agent 识别 -1 时自动降级跳过 DB 节点，不阻塞主业务流程。

## 缓存污染与 __pycache__
__pycache__ 目录存放 Python 编译后的字节码文件（.pyc），文件名格式为 <模块名>.cpython-<版本>.pyc。例如 db.cpython-313.pyc 表示 db.py 在 Python 3.13 下编译成功。该目录的存在证明对应模块已被解释器成功 import 过（语法无误），是验证代码是否运行的可靠证据。Python 修改顶层 import/类名/全局变量后，旧的 .pyc 字节码缓存可能导致 NameError 等诡异错误。修复：Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force 清除所有缓存目录。

## Pathlib 跨平台路径拼接
MODEL_PATH = Path(__file__).resolve().parent.parent / "runs" / "detect" / "weights" / "best.pt" 使用 pathlib 运算符重载实现跨平台路径拼接。__file__ 获取当前文件路径，.resolve() 转为绝对路径，.parent.parent 上溯两级到项目根目录。该写法兼容 Windows（反斜杠）和 Linux（正斜杠），Docker 容器内路径自动适配。环境变量化后：os.getenv("MODEL_PATH", str(Path(...))) 优先读环境变量，回退到相对路径默认值。避免硬编码绝对路径导致容器内路径失效。

## pytest 测试体系搭建
PCB 项目在 tests/ 目录构建三层测试：test_feature_extractor.py 用 pytest.fixture(scope="module") 复用单例 FeatureExtractor，断言 Hook Layer 索引 >9、向量 shape 为 (256,)、L2 模长 ≈1.0；test_knowledge_base.py 验证单例模式（kb is StandardKnowledgeBase()）、YAML 加载条数、已知缺陷匹配、未知缺陷降级、大小写容错（MISSING_HOLE → missing_hole）；test_root_cause_agent.py 覆盖两个 early return 分支 + LLM 真实调用（@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"))）。实测 pytest tests/ -v -s 输出 11 passed。

## 死代码陷阱与控制流审计
_extract_vector 中存在两个 return：第一个 return vec / norm 提前返回 ndarray，下方 return normalized.tolist() 成为不可达死代码。Python 不报语法错误，静默忽略。grep 只能验证符号存在，必须用 sed -n '95,102p' 查看上下文确认控制流连通性。修复：删除提前 return，统一变量名为 normalized，确保 .tolist() 为唯一出口。该问题在重构代码时高频出现，建议每次重构后用 python -m py_compile 和最小单元测试验证控制流。