# Milvus 笔记

## Milvus Standalone 四层嵌套排障
在 Windows + WSL2 部署 Milvus v2.4.17 Standalone 时，连续遭遇 4 层嵌套问题，采用逐层剥离法定位。第 1 层：容器退出码 134 (SIGABRT)，根因是 LD_PRELOAD 指定的 jemalloc 路径在容器内不存在，修正为 /milvus/lib/libjemalloc.so。第 2 层：etcd 报 apply request took too long (168ms>100ms)，根因是 WSL2 9P 文件系统 I/O 延迟 100-200ms 突破 Raft 心跳阈值，通过自定义 milvus.yaml 放宽 requestTimeout/dialTimeout 至 30s。第 3 层：mkdir /var/lib/milvus/rdbms_meta: permission denied，根因是 Named Volume 权限归 Docker daemon，RocksDB CGO 子进程 UID 不一致，改为 Bind Mount + user: "0:0"。第 4 层：RocksMQ 初始化 SIGABRT，根因是 RocksDB 依赖 mmap/fallocate POSIX 语义，NTFS→WSL2 桥接不支持，属内核层缺陷无法配置绕过，最终架构切换为 pymilvus-lite。

## pymilvus-lite 嵌入式方案
pymilvus-lite 是 Milvus 官方提供的嵌入式引擎，通过 pip install milvus-lite 安装。使用 MilvusClient(uri="./milvus_data.db") 初始化，数据持久化到本地 .db 文件。不依赖 etcd、MinIO、RocksMQ 等外部组件，适合本地开发/测试/原型验证。单进程嵌入式，不支持分布式部署、多副本、RBAC 权限管理，但向量检索、插入、删除、标量过滤等核心功能与完整版 API 完全兼容。适合百万级向量以内场景，生产环境建议部署完整版 Milvus 集群。Milvus Lite 3.2.1 是纯 Python 嵌入式向量数据库，数据以目录形式持久化（milvus_data.db/collections/pcb_defects/），含 manifest.json/schema.json/wal 等文件。

## pymilvus-lite nullable 兼容性问题
pymilvus 2.4.13 客户端在创建集合时发送的 proto schema 包含 nullable 字段，但旧版 milvus-lite 的 translator 代码（milvus_lite/adapter/grpc/translators/schema.py 中 _decode_field 函数）未支持该字段，导致 AttributeError: nullable。解决方案：升级 milvus-lite 到最新版，或降级 pymilvus 到 2.4.8。验证方法：python -c "from milvus_lite.adapter.grpc.translators.schema import _decode_field; import inspect; print('nullable' in inspect.getsource(_decode_field))" 输出 True 表示已支持。该问题本质是版本对齐问题，非代码 bug。pymilvus 3.0.1 + milvus-lite 3.2.1 组合已验证。

## Milvus 日志排查方法论
排查 Milvus 崩溃时需区分"现场照片"与"死亡原因"：Go panic 的 goroutine stack trace 只是崩溃瞬间的快照，真正的 root cause 在日志更靠前的 ERROR/FATAL 行中。使用 docker logs <container> 2>&1 | Select-String -Pattern "panic|fatal|abort|SIGABRT|error" -Context 3,3 精准定位致命错误。逐层剥离法：每修复一个 bug 才暴露下一个，绝不跳步。日志中 tini 打印帮助信息说明容器找不到主程序，通常是镜像标签错误（缺少 -standalone 后缀）。Docker 容器退出码 134 = 128 + 6 (SIGABRT)，表示进程主动 abort。

## HNSW 索引与相似度度量
Milvus 支持 HNSW（Hierarchical Navigable Small World）索引，适合高维向量近似最近邻搜索。相似度度量支持 COSINE、L2、IP。创建集合时指定 metric_type="COSINE"，dimension 根据 Embedding 模型输出确定（bge-m3 为 1024 维，YOLO 特征可能为 256 或 512 维）。HNSW 参数：M（每层最大连接数，通常 16-64）、efConstruction（构建时搜索范围，通常 100-500）、ef（查询时搜索范围，通常 50-200）。Milvus Lite 小数据集推荐 FLAT（暴力检索）精度最高，大规模场景改 HNSW 以少量精度换 10x 检索速度。

## 标量过滤与分区
Milvus 支持标量字段过滤（如 defect_type == "crack"、severity == "critical"），结合向量检索实现混合查询。分区（Partition）用于按业务维度隔离数据（如按批次号、产品型号），提升查询效率。Collection 创建时定义 scalar fields 的 metadata：{standard_id, clause_no, defect_type, severity}。检索时用 filter 参数指定过滤条件，如 client.search(collection_name, data, filter='severity == "critical"', limit=10)。pymilvus 3.x 过滤参数是 filter 而不是 expr（后者是 AnnSearchRequest 的参数名）。

## VECTOR_DIM 维度对齐
YOLOv8n Neck 最终输出层 C2f 通道数为 256，通过 m.model.model[21].cv2.conv.weight.shape[0] 验证。Milvus 集合创建时 VECTOR_DIM = 256 必须与特征提取输出严格一致，否则 upsert/search 报维度不匹配。曾因假设值 512 导致集合重建——client.drop_collection("pcb_defects") 后以 256 重建。该维度由模型结构决定，不可随意修改。特征提取后 L2 归一化，配合 COSINE 度量实现方向相似度匹配。Embedding 维度校验：len(vectors[0]) != settings.embedding_dim 时抛 RuntimeError。

## 向量 Upsert 与幂等性
db.upsert_defect_feature(defect_id, vector, defect_class, confidence) 使用 upsert 而非 insert，保证同一缺陷重复检测时覆盖旧记录而非主键冲突。defect_id 由 hashlib.md5(f"{image_path}_{class_name}_{bbox_str}") 生成，确保相同图片相同位置的缺陷产生相同 ID。该设计支持幂等写入，服务重启后重新处理同一图片不会产生重复向量。批量写入校验：defects 与 vectors 长度一致，向量维度 256。失败重试：指数退避（1s/2s/4s），最多 3 次。写入失败返回 -1 哨兵值。

## 集合 Schema 与索引设计
完整 Schema 包含：id（VARCHAR 主键）、vector（FLOAT_VECTOR dim=256）、defect_class（VARCHAR）、confidence（FLOAT）、image_path（VARCHAR）。索引：FLAT（精确搜索）+ COSINE（余弦相似度）。Milvus Lite 小数据集推荐 FLAT，无需调参。快捷 API client.create_collection(collection_name, dimension=256, metric_type="COSINE") 仅创建 id+vector，无法标量过滤，企业级需完整 Schema。init_db() 首先 client.has_collection(COLLECTION_NAME) 检查，已存在则打印日志跳过，不存在则创建完整 Schema 和索引。

## Milvus Lite 文件锁机制与运维 SOP
Milvus Lite 底层 RocksDB 是单进程独占锁模型。FastAPI 运行时持有 milvus_data.db 锁，docker exec 启动新进程会触发 DataDirLockedError。运维 SOP：docker compose stop 释放锁 → docker compose run --rm 临时容器执行 DB 操作 → docker compose up -d 重启。嵌入式 DB 运维必须先停服，临时容器独立进程空间不争抢锁。全程停机约 3 秒，零数据丢失。milvus_data.db 是 RocksDB 存储目录而非单文件，PowerShell 删除时必须加 -Recurse：Remove-Item .\milvus_data.db -Recurse -Force。

## Docker Named Volume 权限问题
Docker Named Volume 存储于 WSL2 的 ext4.vhdx 虚拟磁盘内，文件所有权由 Docker daemon 管理。Milvus 内部 RocksDB（CGO 调用的 C++ 库）可能 fork 出子进程以不同 UID 执行，导致 mkdir /var/lib/milvus/rdbms_meta: permission denied。解决方案：将 Named Volume 改为 Bind Mount（./milvus-data:/var/lib/milvus），权限继承自宿主机文件系统更可控；同时设置 user: "0:0" 强制以 root 运行。Bind Mount 在 WSL2 下仍可能触发 RocksDB 文件系统兼容性问题。