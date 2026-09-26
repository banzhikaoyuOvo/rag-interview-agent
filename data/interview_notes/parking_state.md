# Parking-YOLO 项目状态锚点

> 最后更新：2026-09-23 19:35
> 状态：✅ 已完成
> 位置：.local/STATE.md（被 .gitignore 忽略，不进 Git）

## 一、项目基本信息

| 项 | 值 |
|---|---|
| 项目名 | Parking-YOLO（GitHub 用英文名） |
| 产品中文名 | 小鱼智能停车场（仅前端界面 + LLM 提示词） |
| GitHub 仓库 | https://github.com/banzhikaoyuOvo/parking-yolo.git |
| 本地根目录 | D:\Install software\pyCharm\project\parking-yolo |
| 项目性质 | 毕业设计 |
| 开发时间 | 2026 年 3 月 ~ 2026 年 9 月 |
| 作者 | 俞晓兴（GitHub: banzhikaoyuOvo，邮箱: 3341428887@qq.com） |
| 协作者 | 仅作者本人 |
| 部署状态 | 未部署（仅 GitHub 开源）；有部署计划 |
| 运行环境 | conda 环境 D:\Install software\envs\parking_yolo |

## 二、技术栈

| 层 | 技术 | 版本 |
|---|---|---|
| 视觉模型 | YOLOv8 (ultralytics) | 8.2.0 起步，后续升级 |
| 后端 | Flask + flask-cors | 3.0.0 / 4.0.0 |
| LLM | DeepSeek Chat API (openai SDK) | openai 3.19.0 |
| RAG | Jaccard 相似度 + 中文 bigram 分词 | 零依赖实现 |
| 图像 | Pillow | 10.3.0 |
| 前端 | 原生 HTML/CSS/JS（单文件） | - |
| 日志 | 结构化 JSON | - |

## 三、模型训练信息

| 项 | 值 |
|---|---|
| 数据集 | PKLot（巴西 UFPR + PUCPR 联合发布的停车场车位数据集） |
| 预处理图片数 | 12,417 张 |
| 基础模型 | YOLOv8n |
| 训练轮数 | 32 epoch（最佳点 epoch 29） |
| 训练环境 | 本机 GPU |
| 最佳 mAP50-95 | 0.96544（epoch 29） |
| 最佳 mAP50 | 0.9945 |
| 最佳 Precision | 0.99819 |
| 最佳 Recall | 0.99832 |
| 推理速度 | 10 FPS（本机 GPU） |
| 模型大小 | 17.6 MB (best.pt) |
| 类别数 | 3（spaces / space-empty / space-occupied） |
| SHA256 | 2278cd45fe1613bd1b4e2b0188b7fb9a6eb47c58fa97471fa7685b7a80026c1f |

## 四、Git 状态

### 5 条 commit（按时间倒序）
985637e (HEAD -> main, origin/main) docs: add project screenshots
d8dff87 docs: remove gif demo placeholder
1d3ab0e docs: fix HTML entity in README code blocks
964b244 docs: add training results, fix .gitignore encoding
ff76950 Initial commit: Parking-YOLO (clean repo, model weights via Release)

text

- 本地 main = 远程 origin/main = 985637e
- 工作区干净
- Git 跟踪 26 个文件

## 五、Release 信息

- URL：https://github.com/banzhikaoyuOvo/parking-yolo/releases/tag/v1.0.0
- Tag：v1.0.0
- Target：main
- 附件：best.pt（17.6 MB）
- SHA256：2278cd45fe1613bd1b4e2b0188b7fb9a6eb47c58fa97471fa7685b7a80026c1f

## 六、已完成改造（完整清单）

### 项目结构
1. 备份：_backup_pythonProject_20260923_165353
2. 归档：_archive（毕业设计 / 人工智能方向 / runs / yolov8n.pt）
3. 扁平化：xiaoyu-parking 内容上移到根
4. 本地文件夹改名 pythonProject -> parking-yolo
5. 删除 .venv（统一用 conda 环境）

### 文档
6. README 重写（含训练结果章节 + 2 张截图）
7. LICENSE（MIT 2026）
8. .env.example
9. docs/ARCHITECTURE.md
10. docs/screenshots/（overview.png + detections.png）
11. 修复 README 里的 HTML 实体转义（&#x20;）
12. 删除 GIF 占位段落

### 代码
13. rag_service.py 注释修正（TF-IDF -> Jaccard）
14. app.py 去重 import json

### Git
15. .gitignore 重写（英文注释，排除 best.pt / .env / .local）
16. Git 历史重写（移除 best.pt，减小仓库体积）
17. Push 到 GitHub（覆盖旧 commit）
18. Release v1.0.0 + best.pt 上传

### 依赖兼容
19. ultralytics 升级（解决 torch.load weights_only 问题）
20. openai 升级到 3.19.0 + httpx 0.28.1（解决 proxies 参数冲突）

### 本地工具
21. 创建 start.bat（一键启动 + 4 秒后自动打开浏览器）

## 七、全链路验证结果

- YOLO 检测：真实推理，B1 空位 97、B2 空位 28
- RAG 检索：命中导航知识（6 条）
- LLM 流式：真 DeepSeek 输出（无 MOCK 字样）
- 前端：大屏 + 详情页 + 检测框 + 打字机效果全部正常
- start.bat 双击启动成功

## 八、后续待办（可选，不紧急）

- [ ] 部署（未部署，有部署计划）
- [ ] 可选优化：
  - config.py 拆分（配置 / 提示词 / 知识）
  - app.py 的 debug=True 改读环境变量
  - CORS 收紧到指定域名
  - 多 worker 时 _screen_state 改用 Redis
- [ ] 可选：README 里 Release 下载链接改为 best.pt 直接下载地址
- [ ] 可选：给 start.bat 创建桌面快捷方式

## 九、关键路径

| 用途 | 路径 |
|---|---|
| 项目根 | D:\Install software\pyCharm\project\parking-yolo |
| conda 环境 | D:\Install software\envs\parking_yolo |
| 完整备份 | D:\Install software\pyCharm\project\_backup_pythonProject_20260923_165353 |
| 归档目录 | D:\Install software\pyCharm\project\_archive |
| 启动脚本 | D:\Install software\pyCharm\project\parking-yolo\start.bat |

## 十、恢复方法

### 完全恢复原状
cd "D:\Install software\pyCharm\project"
Remove-Item "parking-yolo" -Recurse -Force
Copy-Item "_backup_pythonProject_20260923_165353" "parking-yolo" -Recurse
cd parking-yolo
git fetch origin
git reset --hard origin/main

text

### 仅恢复 Git（保留当前文件）
git fetch origin
git reset --soft origin/main

text

## 十一、下次中断后快速恢复上下文

给新对话的开场白：
【Parking-YOLO 上下文】

路径：D:\Install software\pyCharm\project\parking-yolo

仓库：banzhikaoyuOvo/parking-yolo

状态：v1.0.0 已发布，全链路通过

运行环境：conda 环境 parking_yolo

详见：.local/STATE.md

需求：<新需求>

text
