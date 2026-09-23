# FastAPI 笔记

## lifespan 生命周期管理
FastAPI 0.115+ 推荐用 @asynccontextmanager 的 lifespan 替代已弃用的 @app.on_event("startup")。示例：@asynccontextmanager async def lifespan(app): db.init_db(); yield; db.close()，然后 app = FastAPI(lifespan=lifespan)。lifespan 支持异步上下文管理器，在 yield 前执行启动逻辑，yield 后执行关闭逻辑。相比 on_event 更清晰，且支持依赖注入。PCB 项目在 lifespan 中执行 UPLOAD_DIR.mkdir(exist_ok=True) 和 db.init_db()，启动时初始化 Milvus Lite 集合；yield 后执行 db.client.close() 释放连接。开发阶段仍可用 on_event，生产建议迁移。

## SSE 流式推送与事件协议
FastAPI 通过 sse-starlette 库或 StreamingResponse 实现 SSE（Server-Sent Events）流式推送，用于实时审核进度展示。使用 EventSourceResponse(event_generator(), media_type="text/event-stream") 包装异步生成器，逐条推送事件。事件格式：data: {json.dumps(...)}\n\n。关键响应头：Cache-Control: no-cache、Connection: keep-alive、X-Accel-Buffering: no（防 Nginx 缓冲）。4 种事件类型：start（任务接收）、node_start（节点开始）、llm_token（Token 流）、complete（含 final_state）、error。前端用 EventSource 或 response.body.getReader() 监听。

## 文件上传与 python-multipart
FastAPI 使用 UploadFile 接收上传文件时，需要 python-multipart 库支持。若未安装会报 RuntimeError: Form data requires "python-multipart" to be installed.。安装命令：pip install python-multipart -i https://pypi.tuna.tsinghua.edu.cn/simple。该库不在 FastAPI 核心依赖中，需单独安装。文件上传接口示例：@app.post("/inspect") 配合 async def inspect(image: UploadFile = File(...))，通过 await image.read() 读取文件内容。三重安全校验：文件大小（10MB 上限）、格式白名单（.jpg/.jpeg/.png/.bmp）、先读入内存再校验避免二次读流。

## 依赖注入与中间件
FastAPI 依赖注入通过 Depends 实现，用于共享数据库连接、认证等。示例：def get_db(): return MilvusClient(uri="milvus_data.db")，路由中 def endpoint(db=Depends(get_db))。中间件用 @app.middleware("http") 注册，处理请求/响应日志、CORS、限流。run_in_threadpool 用于在异步路由中执行同步阻塞函数（如 YOLO 推理），避免阻塞事件循环。CORS 中间件开发阶段放开便于前端联调，生产环境收紧到具体域名。

## 健康检查分层设计
/health 为纯内存 Liveness Probe，零 I/O、零 DB 访问，仅返回 {"status": "ok"}。/ready 为深度 Readiness Probe，检查 Milvus 连接等外部依赖。此设计避免 gRPC too_many_pings 限流，符合 K8s 探针最佳实践。pymilvus gRPC 默认 keepalive=10s，Docker 健康检查每 30s 访问 /health，若访问 Milvus 触发 I/O 会导致 ENHANCE_YOUR_CALM 限流。探针本身不应成为系统负担，Liveness 只验活，Readiness 才验依赖。

## 接口限流与安全防护
使用 slowapi 实现 IP 级限流。初始化：limiter = Limiter(key_func=get_remote_address)，app.state.limiter = limiter。路由装饰器：@limiter.limit("5/minute")，需在路由参数中加 request: Request。限流异常处理：@app.exception_handler(RateLimitExceeded) 返回 429 JSON。文件安全：读取后校验大小（10MB 上限）和格式白名单（.jpg/.jpeg/.png/.bmp）。三层防护防止恶意刷 LLM Token：限流 + 文件大小校验 + 格式白名单。

## 环境变量运行时注入
敏感配置严禁 COPY 进镜像。通过 docker-compose.yml 的 env_file 在运行时注入。Python 端使用 load_dotenv() 加载 .env 文件，必须在 import graph 之前执行，否则 graph.py 模块级初始化 ChatOpenAI 时读不到 API Key，报 OpenAIError：api_key 未设置。Dockerfile 中删除所有 COPY .env 指令，符合 12-Factor App 原则。.env 必须加入 .gitignore 和 .dockerignore，禁止提交到仓库或打入镜像。.env.example 提供脱敏模板。

## 原生 dict 驱动 graph.invoke
main.py 的 /inspect 端点使用 initial_state = {"image_path": str(save_path)} 原生 dict 调用 graph.invoke(initial_state)。严禁使用 PCBQualityState(image_path=...) 构造——PCBQualityState 是 TypedDict，运行时就是 dict，不存在可实例化的类。graph.invoke() 返回最终 State dict，直接作为 JSON 响应体返回。FastAPI 自动序列化 dict 为 JSON，无需 .model_dump() 转换。该设计保持 API 层轻薄，业务逻辑全部封装在 Graph 内部。

## 双端点向后兼容设计
保留原 /inspect 同步 JSON 端点，新增 /inspect_stream SSE 端点。同步端点内部调用 graph.invoke(initial_state) 返回完整结果，适合脚本调用和自动化测试。SSE 端点调用 graph.astream_events() 推送流式结果，适合浏览器交互。两者并行不冲突，API 兼容保底。生产环境去掉 --reload 参数，使用多 worker 模式。uvicorn app.main:app --reload --host 0.0.0.0 --port 8000，--host 0.0.0.0 允许局域网访问，默认 127.0.0.1 仅本机可访问。