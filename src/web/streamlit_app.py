"""RAG Interview Agent - Streamlit 前端 v5（暗色科技风 + Mock 兜底 + 视觉精修）"""
from __future__ import annotations

import json
import os

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8001")

import re

# 匹配 [Source: xxx, Chunk ID: yyy]
_CITATION_PATTERN = re.compile(r'\s*\[Source:\s*[^\]]+\]')


def clean_citations(text: str) -> str:
    """去除回答中的 inline citation（来源已单独在下方 expander 展示）"""
    cleaned = _CITATION_PATTERN.sub('', text)
    # 清理标点前的多余空格
    cleaned = re.sub(r'\s+([，。！？,.!?])', r'\1', cleaned)
    # 合并多余空格
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    return cleaned.strip()


st.set_page_config(
    page_title="俞晓兴 · AI 应用开发工程师",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Mock 兜底数据（后端不可用时使用）──
MOCK_ANSWERS = {
    "用 3 句话介绍你自己": """我是俞晓兴，26 届电子信息工程（AI 方向）应届毕业生，退役大学生士兵。

3 个月内独立完成 3 个 AI 项目并全部开源：RAG 求职 Agent、PCB 质检多 Agent 平台、Parking-YOLO 多模态智慧停车系统。

擅长 LangGraph 多 Agent 编排、RAG 检索增强、FastAPI 异步服务，能独立完成从架构设计到 Docker 部署的完整工程流程。""",

    "你最匹配这个岗位的三个证据": """**1. Agent 编排深度**：独立完成 PCB 质检多 Agent 平台，基于 LangGraph 5-Agent Orchestrator-Worker 架构，掌握条件边、State Reducer、异常隔离、断点续传。

**2. RAG 检索优化**：个人求职 RAG Agent 完整落地 Hybrid Search（Dense + BM25 + RRF）、Query 改写、Citation 溯源、跨集合检索。

**3. 工程落地能力**：3 个项目均含 Docker 部署、单元测试、完整 README，覆盖 FastAPI / Milvus / YOLOv8 / DeepSeek 全链路。""",

    "讲一个最有挑战的项目": """**PCB 质检多 Agent 平台 - GraphRecursionError 死循环排查**。

**问题**：空缺陷图片上传后，图一直转，最后报 GraphRecursionError。

**根因**：Decision Agent 用 `if not state.get("defects")` 判断，但空缺陷返回空列表 `[]`，`if not []` 为 True，误判为"还没检测"，无限重跑 Visual Agent。

**修复**：改用键存在性判断 `if "defects" not in state`，并要求 Visual Agent 无论是否检出缺陷都返回 `{"defects": [...]}`，空列表也是有效结果。

**沉淀**：显式执行标记 `visual_done: bool` 比隐式数据存在性判断更可靠。""",

    "你熟悉哪些技术栈，项目里怎么用的": """**LangGraph**：PCB 项目 5-Agent 编排、RAG 项目 5 节点状态机。

**RAG 全链路**：Hybrid Search（Dense + BM25 + RRF）、Query 改写、Citation 溯源、Embedding（BGE-M3）。

**向量库**：Milvus Standalone（RAG）、pymilvus-lite（PCB 嵌入式）。

**Web 框架**：FastAPI（异步 ASGI）、Flask（同步 WSGI）。

**深度学习**：YOLOv8n 训练 / 微调 / 特征提取。

**部署**：Docker Compose、GPU 穿透、SSE 流式。""",

    "你的短板是什么，怎么补": """**短板 1**：缺少大厂实习经历。**补法**：用 3 个高质量开源项目弥补，每个都含完整 README + 测试 + Docker 部署。

**短板 2**：算法侧训练经验偏少。**补法**：YOLOv8n 微调项目已覆盖数据清洗、训练、评估全流程。

**短板 3**：多模态大模型（VLM）经验少。**正在补**：Parking-YOLO 已做 YOLO + LLM 融合，下一步补 VLM 实践。""",
}


def get_mock_answer(query: str) -> str | None:
    """从 Mock 数据中匹配答案"""
    for key, ans in MOCK_ANSWERS.items():
        if key in query or query in key:
            return ans
    return None


# ── 全局样式 ──
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 6rem;
    max-width: 1280px;
}

/* ── 全局背景（深灰，非纯黑）── */
.stApp {
    background: #0F1115;
}
body { background: #0F1115; }

/* ── Hero 区 ── */
.hero-wrap {
    position: relative;
    background: linear-gradient(135deg, #1a1b2e 0%, #1e1b4b 40%, #2a2560 100%);
    border-radius: 20px;
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    overflow: hidden;
    border: 1px solid rgba(99, 102, 241, 0.18);
    box-shadow: 0 20px 60px -20px rgba(99, 102, 241, 0.3);
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60%; right: -15%;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.28) 0%, transparent 70%);
    pointer-events: none;
    animation: breathe 8s ease-in-out infinite;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -40%; left: -5%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.15) 0%, transparent 70%);
    pointer-events: none;
}
@keyframes breathe {
    0%, 100% { opacity: 0.7; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.08); }
}
.hero-content { position: relative; z-index: 1; }
.hero-name {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 60%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.75rem 0;
    line-height: 1.05;
}
.hero-title {
    font-size: 1.25rem;
    color: #a5b4fc;
    font-weight: 500;
    margin-bottom: 1.75rem;
    letter-spacing: 0.01em;
}
.hero-desc {
    font-size: 0.98rem;
    color: #cbd5e1;
    line-height: 2;
    margin-bottom: 2rem;
    max-width: 900px;
}
.hero-contacts {
    display: flex;
    flex-wrap: wrap;
    gap: 2rem;
    font-size: 0.9rem;
    color: #9CA3AF;
    margin-bottom: 2rem;
}
.hero-contacts .item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.hero-contacts a {
    color: #a5b4fc;
    text-decoration: none;
    border-bottom: 1px dashed rgba(165, 180, 252, 0.35);
    transition: all 0.2s;
}
.hero-contacts a:hover {
    color: #22d3ee;
    border-bottom-color: #22d3ee;
}
.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    padding: 0.55rem 1.15rem;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.28);
    border-radius: 10px;
    font-size: 0.85rem;
    color: #c7d2fe;
    font-weight: 500;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.hero-tag:hover {
    background: rgba(34, 211, 238, 0.15);
    border-color: rgba(34, 211, 238, 0.5);
    color: #67e8f9;
    transform: translateY(-2px);
}

/* ── 区块标题 ── */
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 2.5rem 0 1.5rem 0;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.section-title .accent {
    color: #22d3ee;
    font-size: 0.9rem;
    text-shadow: 0 0 20px rgba(34, 211, 238, 0.6);
}

/* ── 项目卡片 ── */
.proj-card {
    position: relative;
    background: linear-gradient(180deg, #171923 0%, #1a1a2e 100%);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 16px;
    padding: 1.75rem 1.6rem 1.5rem 1.6rem;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
    min-height: 320px;
    display: flex;
    flex-direction: column;
}
.proj-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #22d3ee, transparent);
    opacity: 0;
    transition: opacity 0.35s;
}
.proj-card:hover {
    border-color: rgba(34, 211, 238, 0.4);
    transform: translateY(-6px);
    box-shadow: 0 16px 40px -12px rgba(34, 211, 238, 0.25);
}
.proj-card:hover::before { opacity: 1; }
.proj-emoji {
    font-size: 1.8rem;
    margin-bottom: 0.85rem;
    display: block;
}
.proj-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.01em;
}
.proj-meta {
    font-size: 0.8rem;
    color: #818cf8;
    font-family: 'SF Mono', Menlo, monospace;
    margin-bottom: 1rem;
    letter-spacing: 0.02em;
}
.proj-desc {
    font-size: 0.88rem;
    color: #9CA3AF;
    line-height: 1.75;
    margin-bottom: 1.25rem;
    flex-grow: 1;
}
.proj-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: auto;
}
.proj-badge {
    display: inline-block;
    padding: 0.28rem 0.65rem;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.22);
    border-radius: 6px;
    font-size: 0.73rem;
    color: #a5b4fc;
    font-weight: 500;
}

/* ── Streamlit 按钮重样式 ── */
.stButton > button {
    background: rgba(99, 102, 241, 0.08) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    color: #c7d2fe !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.25s !important;
    padding: 0.65rem 1rem !important;
    font-size: 0.88rem !important;
}
.stButton > button:hover {
    background: rgba(34, 211, 238, 0.15) !important;
    border-color: rgba(34, 211, 238, 0.5) !important;
    color: #67e8f9 !important;
    transform: translateY(-1px);
}
.stButton > button[kind="secondary"] {
    background: rgba(23, 25, 35, 0.6) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    color: #cbd5e1 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(34, 211, 238, 0.1) !important;
    border-color: rgba(34, 211, 238, 0.45) !important;
    color: #67e8f9 !important;
}

/* Link button 样式 */
.stLinkButton > a {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1)) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    color: #a5b4fc !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    text-decoration: none !important;
    transition: all 0.25s !important;
    font-size: 0.82rem !important;
    padding: 0.4rem 0.9rem !important;
}
.stLinkButton > a:hover {
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.25), rgba(139, 92, 246, 0.25)) !important;
    border-color: rgba(34, 211, 238, 0.6) !important;
    color: #67e8f9 !important;
    transform: translateY(-2px);
    box-shadow: 0 0 24px rgba(34, 211, 238, 0.3), 0 4px 12px rgba(34, 211, 238, 0.15);
}

/* ── 对话气泡（v5 加强）── */
[data-testid="stChatMessage"] {
    padding: 1.15rem 1.4rem !important;
    border-radius: 14px !important;
    margin-bottom: 1rem !important;
    border: 1px solid transparent !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(100, 116, 139, 0.08), rgba(100, 116, 139, 0.04)) !important;
    border-left: 3px solid #64748b !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.06), rgba(34, 211, 238, 0.04)) !important;
    border-left: 3px solid #8b5cf6 !important;
    border-color: rgba(139, 92, 246, 0.2) !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    margin-bottom: 0.8rem !important;
    line-height: 1.75 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child {
    margin-bottom: 0 !important;
}

/* ── 代码块样式（统一青紫色调）── */
code {
    background: rgba(99, 102, 241, 0.15) !important;
    color: #67e8f9 !important;
    padding: 0.15rem 0.4rem !important;
    border-radius: 5px !important;
    font-size: 0.88em !important;
    border: 1px solid rgba(99, 102, 241, 0.2);
}
pre {
    background: #0a0b10 !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
pre code {
    background: transparent !important;
    color: #cbd5e1 !important;
    border: none !important;
}
[data-testid="stMarkdownContainer"] .k,
[data-testid="stMarkdownContainer"] .kd,
[data-testid="stMarkdownContainer"] .kn {
    color: #c7d2fe !important;
}
[data-testid="stMarkdownContainer"] .s,
[data-testid="stMarkdownContainer"] .s1,
[data-testid="stMarkdownContainer"] .s2 {
    color: #67e8f9 !important;
}
[data-testid="stMarkdownContainer"] .nf,
[data-testid="stMarkdownContainer"] .nc {
    color: #fbbf24 !important;
}

/* ── 底部输入框毛玻璃加强 ── */
[data-testid="stChatInput"] {
    backdrop-filter: blur(16px) !important;
    background: rgba(15, 17, 21, 0.85) !important;
    border: 1px solid rgba(34, 211, 238, 0.18) !important;
    border-radius: 14px !important;
    box-shadow: 0 -4px 24px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(34, 211, 238, 0.05) inset !important;
    transition: all 0.25s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(34, 211, 238, 0.4) !important;
    box-shadow: 0 -4px 24px rgba(0, 0, 0, 0.4), 0 0 24px rgba(34, 211, 238, 0.15) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #E5E7EB !important;
}

/* ── 演示模式徽章（v5：位置由 Python 控制）── */
.demo-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.8rem;
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.25);
    border-radius: 8px;
    color: #fbbf24;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}
</style>
""", unsafe_allow_html=True)

# ── Hero 区 ──
st.markdown("""
<div class="hero-wrap">
    <div class="hero-content">
        <h1 class="hero-name">俞晓兴</h1>
        <div class="hero-title">AI 应用开发工程师 · 26 届电子信息工程（AI 方向）</div>
        <div class="hero-desc">
            退役大学生士兵，3 个月独立完成 3 个 AI 项目并全部开源。<br>
            专注 LangGraph 多 Agent 编排、RAG 检索增强、FastAPI 异步服务。<br>
            能独立完成从架构设计、代码实现、调试排错到 Docker 部署的完整工程流程。
        </div>
        <div class="hero-contacts">
            <span class="item">📍 杭州 / 深圳 / 广州 / 江西 / 上海 / 南京</span>
            <span class="item">📧 <a href="mailto:3341428887@qq.com">3341428887@qq.com</a></span>
            <span class="item">🐙 <a href="https://github.com/banzhikaoyuOvo" target="_blank">github.com/banzhikaoyuOvo</a></span>
        </div>
        <div class="hero-tags">
            <span class="hero-tag">LangGraph</span>
            <span class="hero-tag">RAG</span>
            <span class="hero-tag">Milvus</span>
            <span class="hero-tag">FastAPI</span>
            <span class="hero-tag">Hybrid Search</span>
            <span class="hero-tag">BM25 + RRF</span>
            <span class="hero-tag">DeepSeek</span>
            <span class="hero-tag">Docker</span>
            <span class="hero-tag">YOLOv8</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 核心项目 ──
st.markdown('<div class="section-title"><span class="accent">◆</span> 核心项目</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class="proj-card">
        <span class="proj-emoji">🔍</span>
        <div class="proj-name">RAG 求职 Agent</div>
        <div class="proj-meta">FastAPI · LangGraph · Milvus</div>
        <div class="proj-desc">
            企业级 RAG 知识库助手，支持 Hybrid Search + RRF、Query 改写、Citation 溯源、跨集合检索、JD 匹配分析。
        </div>
        <div class="proj-badges">
            <span class="proj-badge">Hybrid Search</span>
            <span class="proj-badge">RRF</span>
            <span class="proj-badge">105 条数据</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.link_button("→ GitHub", "https://github.com/banzhikaoyuOvo/rag-interview-agent", use_container_width=True)

with col2:
    st.markdown("""
    <div class="proj-card">
        <span class="proj-emoji">🔬</span>
        <div class="proj-name">PCB 质检平台</div>
        <div class="proj-meta">LangGraph 5-Agent · YOLOv8n</div>
        <div class="proj-desc">
            工业级 PCB 缺陷检测系统，5-Agent 编排 + YOLOv8n 微调（mAP@50=0.896）+ Docker 一键部署。
        </div>
        <div class="proj-badges">
            <span class="proj-badge">5-Agent</span>
            <span class="proj-badge">YOLOv8n</span>
            <span class="proj-badge">Docker</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.link_button("→ GitHub", "https://github.com/banzhikaoyuOvo/pcb-quality-agent", use_container_width=True)

with col3:
    st.markdown("""
    <div class="proj-card">
        <span class="proj-emoji">🚗</span>
        <div class="proj-name">Parking-YOLO</div>
        <div class="proj-meta">YOLOv8 · RAG · Flask</div>
        <div class="proj-desc">
            多模态智慧停车系统（毕业设计），"事实与表达分离"三层解耦架构，已发布 Release v1.0.0。
        </div>
        <div class="proj-badges">
            <span class="proj-badge">mAP@50=0.965</span>
            <span class="proj-badge">Release v1.0.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.link_button("→ GitHub", "https://github.com/banzhikaoyuOvo/parking-yolo", use_container_width=True)

# ── 对话区 ──
st.markdown('<div class="section-title"><span class="accent">◆</span> AI 助手对话</div>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("### ⚙️ 配置")
    collection = st.selectbox(
        "知识库",
        options=["resume_advantages", "job_descriptions", "interview_notes"],
        format_func=lambda x: {
            "resume_advantages": "📄 简历 & 项目",
            "job_descriptions": "💼 岗位 JD",
            "interview_notes": "📝 面试笔记",
        }[x],
    )
    vis_option = st.radio("可见性", ["public", "private", "全部"], index=0)
    visibility = None if vis_option == "全部" else vis_option
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

NODE_LABELS = {
    "query_rewrite": "🔍 Query 改写",
    "hybrid_retrieve": "📚 混合检索",
    "context_build": "📝 上下文构造",
    "generate": "🤖 LLM 生成",
}


def format_node_output(node: str, output: dict) -> str:
    if node == "query_rewrite":
        return f"改写成 {len(output.get('rewritten_queries', []))} 个 query"
    if node == "hybrid_retrieve":
        return f"检索到 {output.get('doc_count', 0)} 条"
    if node == "context_build":
        return f"上下文 {output.get('context_chars', 0)} 字符"
    if node == "generate":
        total = output.get("citation_total", 0)
        invalid = output.get("citation_invalid", 0)
        return f"引用 {total} 处，非法 {invalid} 处"
    return ""


# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

PRESETS = [
    "用 3 句话介绍你自己",
    "你最匹配这个岗位的三个证据",
    "讲一个最有挑战的项目",
    "你熟悉哪些技术栈，项目里怎么用的",
    "你的短板是什么，怎么补",
]

cols = st.columns(len(PRESETS))
for i, q in enumerate(PRESETS):
    with cols[i]:
        if st.button(q, use_container_width=True, key=f"preset_{i}"):
            if st.session_state.pending_query != q:
                st.session_state.pending_query = q
                st.rerun()

# 渲染历史
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "⚡"
    with st.chat_message(msg["role"], avatar=avatar):
        content = msg["content"] if msg["role"] == "user" else clean_citations(msg["content"])
        st.markdown(content)
        if msg.get("sources"):
            with st.expander(f"📚 来源引用 ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source_file']}** `{s['chunk_id']}` score=`{s['score']}`")
                    # 用 code 样式展示纯文本，避免 Markdown 标题渲染
                    preview = s["text_preview"].replace("\n", " ")
                    st.markdown(f"<div style='color:#94a3b8;font-size:0.82rem;line-height:1.6;padding:0.4rem 0.6rem;background:rgba(15,17,21,0.5);border-radius:6px;border-left:2px solid rgba(99,102,241,0.3);'>{preview}</div>", unsafe_allow_html=True)

# 输入
user_input = st.chat_input("问我任何关于我的问题...")

if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="⚡"):
        status = st.status("正在思考...", expanded=True)
        answer_text = ""
        sources_data: list[dict] = []
        citation_info: dict = {}
        is_demo_mode = False

        params = {"query": user_input, "collection": collection}
        if visibility:
            params["visibility"] = visibility

        try:
            with httpx.stream("GET", f"{API_BASE}/api/ask/stream", params=params, timeout=15.0) as resp:
                resp.raise_for_status()
                current_event: str | None = None
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("event: "):
                        current_event = line[7:].strip()
                    elif line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        if current_event == "node":
                            node = data.get("node", "")
                            output = data.get("output", {})
                            label = NODE_LABELS.get(node, f"⚙️ {node}")
                            summary = format_node_output(node, output)
                            status.write(f"✓ {label} — {summary}")
                        elif current_event == "final":
                            answer_text = data.get("answer", "")
                            sources_data = data.get("sources", [])
                            citation_info = {
                                "total": data.get("citation_total", 0),
                                "valid": data.get("citation_valid", 0),
                                "invalid": data.get("citation_invalid", 0),
                            }
            status.update(label="✅ 完成", state="complete", expanded=False)

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError):
            is_demo_mode = True
            mock_ans = get_mock_answer(user_input)
            if mock_ans:
                answer_text = mock_ans
            else:
                answer_text = f"**演示模式下暂不支持此问题。**\n\n完整功能需启动后端服务（FastAPI + Milvus）。当前可尝试以下预设问题：\n\n" + "\n".join(f"- {q}" for q in MOCK_ANSWERS.keys())
            status.update(label="🎭 演示模式", state="complete", expanded=False)

        except Exception as e:
            is_demo_mode = True
            mock_ans = get_mock_answer(user_input)
            answer_text = mock_ans or f"服务暂时不可用。可尝试以下问题：\n\n" + "\n".join(f"- {q}" for q in MOCK_ANSWERS.keys())
            status.update(label="🎭 演示模式", state="complete", expanded=False)

        if answer_text:
            st.markdown(clean_citations(answer_text))
            # v5：演示模式徽章移到答案下方
            if is_demo_mode:
                st.markdown(
                    '<div class="demo-badge" style="margin-top: 0.75rem;">🎭 演示模式 · 后端未连接</div>',
                    unsafe_allow_html=True,
                )
            c_total = citation_info.get("total", 0)
            c_valid = citation_info.get("valid", 0)
            c_invalid = citation_info.get("invalid", 0)
            if c_total > 0:
                badge = "✅" if c_invalid == 0 else "⚠️"
                st.caption(f"{badge} 引用：总计 {c_total} / 合法 {c_valid} / 非法 {c_invalid}")
            if sources_data:
                with st.expander(f"📚 来源引用 ({len(sources_data)})"):
                    for s in sources_data:
                        st.markdown(f"**{s['source_file']}** `{s['chunk_id']}` score=`{s['score']}`")
                        st.text(s["text_preview"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources_data,
        })