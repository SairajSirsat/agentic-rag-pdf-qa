import html
import re

from flask import Flask, jsonify, request

import config
from ragsys.agent_loop import answer_question
from ragsys.registry import get_active, get_entry

app = Flask(__name__)


def esc(s) -> str:
    return html.escape(str(s))


def format_answer_html(text: str) -> str:
    text = esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    paragraphs = text.split("\n\n")
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs if p.strip())


def format_trace_html(trace: list[dict]) -> str:
    parts = []
    for entry in trace:
        judge = entry["judge"]
        decision = "Sufficient" if judge["sufficient"] else "Needs more context"
        badge_class = "ok" if judge["sufficient"] else "retry"
        reranked_str = ", ".join(f"{esc(cid)} ({score:.3f})" for cid, score in entry["reranked"]) or "—"
        fused_str = esc(", ".join(entry["fused_ids"])) or "—"
        parts.append(f"""
        <div class="iter">
          <h4>Iteration {entry['iteration']}</h4>
          <p><b>Query:</b> {esc(entry['query'])}</p>
          <p><b>Fused candidates:</b> <span class="mono">{fused_str}</span></p>
          <p><b>Reranked:</b> <span class="mono">{reranked_str}</span></p>
          <p><b>Judge:</b> <span class="badge {badge_class}">{decision}</span> — {esc(judge['reasoning'])}</p>
          {f"<p><b>Next query:</b> {esc(judge['next_query'])}</p>" if judge.get('next_query') else ""}
        </div>""")
    return "".join(parts)


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Agentic RAG - PDF Q&A</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 860px; margin: 0 auto;
         padding: 24px; background: #0e1117; color: #e6e6e6; }
  h1 { font-size: 1.6rem; margin-bottom: 4px; }
  .subtitle { color: #9a9a9a; margin-bottom: 4px; font-size: 0.95rem; }
  .doc { color: #7fd0ff; margin-bottom: 20px; font-size: 0.95rem; }
  #chat { display: flex; flex-direction: column; gap: 16px; margin-bottom: 90px; }
  .msg { padding: 12px 16px; border-radius: 10px; line-height: 1.5; }
  .msg.user { background: #1e2530; align-self: flex-end; max-width: 80%; }
  .msg.assistant { background: #161b22; border: 1px solid #2a2f3a; }
  .msg.assistant p { margin: 0.4em 0; }
  .citations { color: #9a9a9a; font-size: 0.85rem; margin-top: 8px; }
  details { margin-top: 10px; background: #10141c; border: 1px solid #262b36; border-radius: 8px; padding: 8px 12px; }
  summary { cursor: pointer; font-weight: 600; color: #c9c9c9; }
  .iter { border-top: 1px solid #262b36; padding-top: 8px; margin-top: 8px; }
  .iter:first-child { border-top: none; margin-top: 0; }
  .iter h4 { margin: 0 0 4px 0; color: #7fd0ff; }
  .mono { font-family: ui-monospace, monospace; font-size: 0.82rem; color: #b8b8b8; }
  .badge { padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; }
  .badge.ok { background: #16351f; color: #7fe0a0; }
  .badge.retry { background: #3a2a10; color: #f0c070; }
  #inputRow { position: fixed; bottom: 0; left: 0; right: 0; background: #0e1117; padding: 16px;
              display: flex; gap: 8px; justify-content: center; border-top: 1px solid #262b36; }
  #inputRow > div { display: flex; gap: 8px; width: 100%; max-width: 860px; }
  #question { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #2a2f3a;
              background: #161b22; color: #e6e6e6; font-size: 1rem; }
  button { padding: 10px 18px; border-radius: 8px; border: none; background: #2b6fd6; color: white;
           font-size: 1rem; cursor: pointer; }
  button:disabled { background: #3a3f4a; cursor: default; }
  .status { color: #9a9a9a; font-style: italic; }
</style>
</head>
<body>
  <h1>Agentic RAG &mdash; PDF Q&amp;A</h1>
  <div class="subtitle">Hybrid search (vector + BM25) &rarr; cross-encoder rerank &rarr; agentic judge/retry loop, powered by Qwen3 via Ollama.</div>
  <div class="doc" id="docLabel">Loading document info...</div>
  <div id="chat"></div>
  <div id="inputRow">
    <div>
      <input id="question" type="text" placeholder="Ask a question about the PDF..." autocomplete="off">
      <button id="sendBtn" onclick="ask()">Ask</button>
    </div>
  </div>

<script>
async function loadInfo() {
  const r = await fetch('/api/info');
  const data = await r.json();
  document.getElementById('docLabel').textContent = data.active_pdf
    ? 'Chatting with: ' + data.active_pdf
    : 'No PDF ingested yet.';
}

function addUserMessage(q) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg user';
  div.textContent = q;
  chat.appendChild(div);
  window.scrollTo(0, document.body.scrollHeight);
}

function addStatusMessage(text) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg assistant status';
  div.id = 'statusMsg';
  div.textContent = text;
  chat.appendChild(div);
  window.scrollTo(0, document.body.scrollHeight);
  return div;
}

function addAssistantMessage(data) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg assistant';
  let html = data.answer_html;
  if (data.thinking) {
    html += `<details><summary>Model thinking</summary><p>${data.thinking_html}</p></details>`;
  }
  html += `<details><summary>Retrieval trace (${data.iterations} iteration(s))</summary>${data.trace_html}</details>`;
  if (data.citations && data.citations.length) {
    html += `<div class="citations">Pages cited: ${data.citations.join(', ')}</div>`;
  }
  div.innerHTML = html;
  chat.appendChild(div);
  window.scrollTo(0, document.body.scrollHeight);
}

async function ask() {
  const input = document.getElementById('question');
  const btn = document.getElementById('sendBtn');
  const q = input.value.trim();
  if (!q) return;
  addUserMessage(q);
  input.value = '';
  input.disabled = true;
  btn.disabled = true;
  const statusMsg = addStatusMessage('Retrieving, reranking, and reasoning...');
  try {
    const r = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await r.json();
    statusMsg.remove();
    if (data.error) {
      addStatusMessage('Error: ' + data.error);
    } else {
      addAssistantMessage(data);
    }
  } catch (e) {
    statusMsg.remove();
    addStatusMessage('Request failed: ' + e);
  } finally {
    input.disabled = false;
    btn.disabled = false;
    input.focus();
  }
}

document.getElementById('question').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') ask();
});

loadInfo();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/info")
def info():
    pid = get_active()
    entry = get_entry(pid) if pid else None
    return jsonify({"active_pdf": entry["source_path"] if entry else None})


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "empty question"}), 400

    pid = get_active()
    if not pid:
        return jsonify({"error": "no PDF has been ingested"}), 400

    result = answer_question(question, pid, verbose=False)

    return jsonify(
        {
            "answer_html": format_answer_html(result.answer),
            "thinking": bool(result.thinking),
            "thinking_html": format_answer_html(result.thinking) if result.thinking else "",
            "trace_html": format_trace_html(result.trace),
            "citations": result.citations,
            "iterations": result.iterations_used,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
