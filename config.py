import os

DATA_DIR = os.environ.get("RAG_DATA_DIR", "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
BM25_DIR = os.path.join(DATA_DIR, "bm25")
REGISTRY_PATH = os.path.join(DATA_DIR, "registry.json")
PDFS_DIR = os.environ.get("RAG_PDFS_DIR", "pdfs")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "qwen3-embedding:0.6b")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "qwen3:8b")
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE = os.environ.get("RAG_RERANK_DEVICE", "cuda")

CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", 1200))       # chars
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", 200))  # chars

DENSE_TOP_K = int(os.environ.get("RAG_DENSE_TOP_K", 20))
BM25_TOP_K = int(os.environ.get("RAG_BM25_TOP_K", 20))
RRF_K = int(os.environ.get("RAG_RRF_K", 60))                 # RRF smoothing constant
FUSED_TOP_K = int(os.environ.get("RAG_FUSED_TOP_K", 15))     # into reranker
RERANK_TOP_K = int(os.environ.get("RAG_RERANK_TOP_K", 5))    # kept per iteration
CONTEXT_TOP_K = int(os.environ.get("RAG_CONTEXT_TOP_K", 8))  # max chunks sent to answerer

MAX_ITERATIONS = int(os.environ.get("RAG_MAX_ITERATIONS", 4))
JUDGE_THINK = False
ANSWER_THINK = os.environ.get("RAG_ANSWER_THINK", "true").lower() == "true"

QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve relevant passages that answer it\nQuery: {query}"
)
