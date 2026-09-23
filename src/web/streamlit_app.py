"""RAG Interview Agent - Streamlit 前端（SSE 流式版）"""
from __future__ import annotations

import json

import httpx
import streamlit as st

API_BASE = "http://localhost:8001"

st.set_page_config(
    page_title="个人求职 RAG 助手",
    page_icon="🤖",
    layout="wide",
)

# ── 节点中文标签映射 ──────────────────────────────
NODE_LABELS = {
    "query_rewrite": "🔍 Query 改写",
    "hybrid_retrieve": "📚 混合检索",
    "context_build": "📝 上下文构造",
    "generate": "🤖 LLM 生成",
}


def format_node_output(node: str, output: dict) -> str:
    """把节点的摘要输出转成可读字符串"""
    if node == "query_rewrite":
        qs = output.get("rewritten_queries", [])
        return f"改写成 {len(qs)} 个 query"
    if node == "hybrid_retrieve":
        return f"检索到 {output.get('doc_count', 0)} 条"
    if node == "context_build":
        return f"上下文 {output.get('context_chars', 0)} 字符"
    if node == "generate":
        total = output.get("citation_total", 0)
        invalid = output.get("citation_invalid", 0)
        return f"引用 {total} 处，非法 {invalid} 处"
    return ""


# ── 侧边栏：配置 ──────────────────────────────────
with st.sidebar:
    st.title("⚙️ 配置")

    collection = st.selectbox(
        "知识库",
        options=["resume_advantages", "job_descriptions", "interview_notes"],
        format_func=lambda x: {
            "resume_advantages": "📄 简历 & 项目",
            "job_descriptions": "💼 岗位 JD",
            "interview_notes": "📝 面试笔记",
        }[x],
    )

    vis_option = st.radio(
        "可见性",
        options=["public", "private", "全部"],
        index=0,
    )
    visibility = None if vis_option == "全部" else vis_option

    st.divider()
    st.caption(
        "**public**：给 HR / 面试官看的公开资料\n\n"
        "**private**：自己的面试笔记\n\n"
        "**全部**：不过滤（仅本地调试）"
    )

    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

# ── 主区 ─────────────────────────────────────────
st.title("🤖 个人求职 RAG 助手")
st.caption("基于混合检索 + Query 改写 + Citation 溯源的求职知识库助手（SSE 流式版）")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 预设问题卡片 ──────────────────────────────────
PRESETS = [
    "用 3 句话介绍你自己",
    "你最匹配这个岗位的三个证据",
    "讲一个最有挑战的项目",
    "你熟悉哪些技术栈，项目里怎么用的",
    "你的短板是什么，怎么补",
]

st.markdown("#### 💡 不知道问什么？试试这些：")
cols = st.columns(len(PRESETS))
pending_query: str | None = None
for i, q in enumerate(PRESETS):
    with cols[i]:
        if st.button(q, use_container_width=True, key=f"preset_{i}"):
            pending_query = q

st.divider()

# ── 渲染对话历史 ──────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 来源引用 ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**{s['source_file']}** `{s['chunk_id']}` "
                        f"score=`{s['score']}`"
                    )
                    st.caption(s["text_preview"])

# ── 输入 ─────────────────────────────────────────
user_input = st.chat_input("问我任何关于我的问题...")
if pending_query:
    user_input = pending_query

if user_input:
    # 写入用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 助手回答（SSE 流式）
    with st.chat_message("assistant"):
        status = st.status("正在思考...", expanded=True)
        answer_text = ""
        sources_data: list[dict] = []
        citation_info: dict = {}

        params = {"query": user_input, "collection": collection}
        if visibility:
            params["visibility"] = visibility

        try:
            with httpx.stream(
                "GET",
                f"{API_BASE}/api/ask/stream",
                params=params,
                timeout=120.0,
            ) as resp:
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

                        elif current_event == "error":
                            st.error(f"服务错误：{data.get('message', '未知错误')}")

            status.update(label="✅ 完成", state="complete", expanded=False)

        except httpx.HTTPStatusError as e:
            status.update(label="❌ 服务错误", state="error")
            st.error(f"服务返回错误：{e.response.status_code}\n{e.response.text}")
            st.stop()
        except Exception as e:
            status.update(label="❌ 请求失败", state="error")
            st.error(f"请求失败：{e}\n请确认 uvicorn 服务已启动在 {API_BASE}")
            st.stop()

        # 渲染答案
        if answer_text:
            st.markdown(answer_text)

            # Citation 徽章
            c_total = citation_info.get("total", 0)
            c_valid = citation_info.get("valid", 0)
            c_invalid = citation_info.get("invalid", 0)
            if c_total > 0:
                badge = "✅" if c_invalid == 0 else "⚠️"
                st.caption(
                    f"{badge} 引用：总计 {c_total} / 合法 {c_valid} / 非法 {c_invalid}"
                )

            # 来源卡片
            if sources_data:
                with st.expander(f"📚 来源引用 ({len(sources_data)})"):
                    for s in sources_data:
                        st.markdown(
                            f"**{s['source_file']}** `{s['chunk_id']}` "
                            f"score=`{s['score']}`"
                        )
                        st.caption(s["text_preview"])

        # 写入历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources_data,
        })