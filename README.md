# Agentic RAG for Single-PDF Q&A

Ask questions about one PDF at a time using hybrid (vector + BM25) retrieval, cross-encoder
reranking, and an agentic loop that judges whether retrieved context is sufficient before
answering — reformulating and re-retrieving if not. All models run locally via
[Ollama](https://ollama.com) (Qwen3 embeddings + LLM); reranking runs locally via
`sentence-transformers`.

## Setup

1. Install [Ollama](https://ollama.com) and make sure `ollama serve` is running.
2. `pip install -r requirements.txt`
   - For GPU-accelerated reranking on Windows, install the CUDA build of torch explicitly:
     `pip install torch --index-url https://download.pytorch.org/whl/cu121` (match to your CUDA version).
3. `python main.py setup` — pulls the embedding and LLM models via Ollama.

## Usage

```
python main.py list                                  # list PDFs in pdfs/ and their status
python main.py ingest <name>.pdf [--force]            # ingest a PDF placed in pdfs/
python main.py ask [--pdf <name>.pdf] [--question "..."] [--verbose]
```

Drop a PDF into the `pdfs/` folder, then:

```
python main.py ingest report.pdf
python main.py ask --pdf report.pdf --verbose
```

Omitting `--question` starts an interactive loop (`exit`/`quit` to stop). Omitting `--pdf`
targets whichever PDF was most recently ingested/asked. `--verbose` prints the retrieval,
rerank, and judge trace for each loop iteration.

## Web UI

Two options:

- `python webapp.py` (Flask, `http://127.0.0.1:5000`) — recommended. A plain HTML/JS page over
  a simple request/response API, no persistent connection. Shows the answer, model "thinking",
  and full retrieval/judge trace per question.
- `streamlit run app.py` (`http://localhost:8501`) — richer chat-style UI, but on some Windows
  setups Streamlit's own PyArrow-based session serialization can intermittently crash the
  process after a few minutes of use (unrelated to this project's RAG code, which runs fine
  standalone via the CLI or `webapp.py`). Prefer `webapp.py` if you hit that.

## Configuration

All tunables (model names, chunk size, top-k values at each stage, max loop iterations) are
in `config.py` and overridable via environment variables — see that file for the full list.
