import os

import ollama
import streamlit as st

import config
from ragsys.agent_loop import answer_question
from ragsys.ingest import ingest_pdf
from ragsys.registry import get_active, get_entry, pdf_hash, set_active

st.set_page_config(page_title="PDF RAG Q&A", page_icon="📄", layout="wide")


def ollama_reachable() -> bool:
    try:
        ollama.Client(host=config.OLLAMA_HOST).list()
        return True
    except Exception:
        return False


def list_workspace_pdfs() -> list[str]:
    os.makedirs(config.PDFS_DIR, exist_ok=True)
    return sorted(f for f in os.listdir(config.PDFS_DIR) if f.lower().endswith(".pdf"))


def render_trace_markdown(trace: list[dict]) -> str:
    lines = []
    for entry in trace:
        judge = entry["judge"]
        decision = "✅ **Sufficient**" if judge["sufficient"] else "🔄 **Needs more context**"
        lines.append(f"#### Iteration {entry['iteration']}")
        lines.append(f"**Query:** `{entry['query']}`")
        lines.append("")
        lines.append(f"- Fused candidates: `{', '.join(entry['fused_ids']) or '—'}`")
        reranked_str = ", ".join(f"`{cid}` ({score:.3f})" for cid, score in entry["reranked"]) or "—"
        lines.append(f"- Reranked top: {reranked_str}")
        lines.append(f"- Judge: {decision} — {judge['reasoning']}")
        if judge.get("next_query"):
            lines.append(f"- Next query: `{judge['next_query']}`")
        lines.append("")
    return "\n".join(lines)


st.title("📄 Agentic RAG — PDF Q&A")
st.caption(
    "Hybrid search (vector + BM25) → cross-encoder rerank → agentic judge/retry loop, "
    "powered by Qwen3 via Ollama."
)

if not ollama_reachable():
    st.error(f"Could not reach Ollama at {config.OLLAMA_HOST}. Run `ollama serve`, then refresh this page.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = get_active()

with st.sidebar:
    st.header("Document")

    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded is not None:
        os.makedirs(config.PDFS_DIR, exist_ok=True)
        dest = os.path.join(config.PDFS_DIR, uploaded.name)
        if not os.path.exists(dest):
            with open(dest, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Saved to {dest}")

    files = list_workspace_pdfs()
    if not files:
        st.info(f"Drop a PDF into '{config.PDFS_DIR}/' or upload one above to get started.")
    else:
        selected = st.selectbox("PDF", files, index=0)
        path = os.path.join(config.PDFS_DIR, selected)
        pid = pdf_hash(path)
        entry = get_entry(pid)

        if entry:
            st.success(f"Ingested — {entry['num_chunks']} chunks")
        else:
            st.warning("Not ingested yet")

        force = st.checkbox("Force re-ingest", value=False)
        if st.button("Ingest / Activate", type="primary"):
            with st.spinner("Ingesting PDF (extract → chunk → embed → index)..."):
                pid = ingest_pdf(path, force=force)
            st.session_state.active_pdf = pid
            st.session_state.messages = []
            st.rerun()

        if entry and st.session_state.active_pdf != pid:
            if st.button("Use this PDF"):
                set_active(pid)
                st.session_state.active_pdf = pid
                st.session_state.messages = []
                st.rerun()

    st.divider()
    st.session_state.max_iterations = st.slider("Max agentic iterations", 1, 6, config.MAX_ITERATIONS)
    show_thinking = st.checkbox("Show model thinking", value=True)
    show_trace = st.checkbox("Show retrieval/judge trace", value=True)

active_pdf = st.session_state.active_pdf
if not active_pdf:
    st.info("Ingest a PDF from the sidebar to start asking questions.")
    st.stop()

active_entry = get_entry(active_pdf)
st.subheader(f"Chatting with: {active_entry['source_path'] if active_entry else active_pdf}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("thinking") and show_thinking:
                with st.expander("🧠 Model thinking"):
                    st.markdown(msg["thinking"])
            if msg.get("trace_md") and show_trace:
                with st.expander(f"🔍 Retrieval trace ({msg['iterations']} iteration(s))"):
                    st.markdown(msg["trace_md"])
            if msg.get("citations"):
                st.caption(f"Pages cited: {msg['citations']}")

question = st.chat_input("Ask a question about the PDF...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        config.MAX_ITERATIONS = st.session_state.get("max_iterations", config.MAX_ITERATIONS)
        with st.spinner("Retrieving, reranking, and reasoning..."):
            result = answer_question(question, active_pdf, verbose=False)

        st.markdown(result.answer)
        if result.thinking and show_thinking:
            with st.expander("🧠 Model thinking"):
                st.markdown(result.thinking)

        trace_md = render_trace_markdown(result.trace)
        if show_trace:
            with st.expander(f"🔍 Retrieval trace ({result.iterations_used} iteration(s))"):
                st.markdown(trace_md)
        if result.citations:
            st.caption(f"Pages cited: {result.citations}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "thinking": result.thinking,
            "trace_md": trace_md,
            "iterations": result.iterations_used,
            "citations": result.citations,
        }
    )
