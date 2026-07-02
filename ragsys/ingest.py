from ragsys.bm25_index import BM25Index
from ragsys.chunking import chunk_pages
from ragsys.embeddings import embed_documents
from ragsys.pdf_extract import extract_pages
from ragsys.registry import is_ingested, pdf_hash, register_pdf, set_active
from ragsys.vectorstore import VectorStore


def ingest_pdf(pdf_path: str, force: bool = False, verbose: bool = False) -> str:
    pid = pdf_hash(pdf_path)
    if is_ingested(pid) and not force:
        set_active(pid)
        if verbose:
            print(f"'{pdf_path}' already ingested (pdf_id={pid}); reusing existing index.")
        return pid

    if verbose:
        print(f"Extracting text from {pdf_path}...")
    pages = extract_pages(pdf_path)

    if verbose:
        print(f"Chunking {len(pages)} pages...")
    chunks = chunk_pages(pages, pid)

    if verbose:
        print(f"Embedding {len(chunks)} chunks...")
    vectors = embed_documents([c["text"] for c in chunks])

    VectorStore(pid).upsert(chunks, vectors)

    idx = BM25Index(pid)
    idx.build(chunks)

    register_pdf(pid, pdf_path, len(chunks))
    set_active(pid)

    if verbose:
        print(f"Ingested '{pdf_path}' as pdf_id={pid} ({len(chunks)} chunks).")
    return pid
