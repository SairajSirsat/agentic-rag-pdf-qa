import pymupdf4llm


def extract_pages(pdf_path: str) -> list[dict]:
    """Return [{"page": int, "text": str}, ...], one entry per PDF page.

    Uses pymupdf4llm for table-preserving markdown extraction; falls back to
    raw PyMuPDF text extraction if that fails on a malformed PDF.
    """
    try:
        page_dicts = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        pages = []
        for i, pd in enumerate(page_dicts):
            text = pd.get("text", "") if isinstance(pd, dict) else str(pd)
            page_num = pd.get("metadata", {}).get("page", i + 1) if isinstance(pd, dict) else i + 1
            pages.append({"page": page_num, "text": text})
        if any(p["text"].strip() for p in pages):
            return pages
    except Exception:
        pass

    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    try:
        return [{"page": i + 1, "text": doc[i].get_text()} for i in range(len(doc))]
    finally:
        doc.close()
