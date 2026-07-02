import config

# Split boundaries tried in order, coarsest to finest. A hard character cut is
# always the final fallback so an oversized token-free blob still terminates.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_on(text: str, sep: str) -> list[str]:
    parts = text.split(sep)
    return [p + sep for p in parts[:-1]] + [parts[-1]] if len(parts) > 1 else [text]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest_seps = separators[0], separators[1:]
    pieces = _split_on(text, sep)

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if len(buf) + len(piece) <= chunk_size:
            buf += piece
        else:
            if buf.strip():
                chunks.append(buf)
            if len(piece) > chunk_size:
                chunks.extend(_recursive_split(piece, chunk_size, rest_seps))
                buf = ""
            else:
                buf = piece
    if buf.strip():
        chunks.append(buf)
    return chunks


def split_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Recursively split text on paragraph/line/sentence/word boundaries with char overlap."""
    raw_chunks = _recursive_split(text, chunk_size, _SEPARATORS)

    if overlap <= 0 or len(raw_chunks) <= 1:
        return [c.strip() for c in raw_chunks if c.strip()]

    overlapped = []
    for i, c in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(c)
            continue
        prev_tail = raw_chunks[i - 1][-overlap:]
        overlapped.append(prev_tail + c)
    return [c.strip() for c in overlapped if c.strip()]


def chunk_pages(pages: list[dict], pdf_id: str) -> list[dict]:
    """Chunk each page's text independently, tagging chunks with page + pdf_id metadata."""
    chunks = []
    for page in pages:
        page_num = page["page"]
        for i, text in enumerate(split_text(page["text"])):
            chunks.append(
                {
                    "id": f"{pdf_id}_{page_num}_{i}",
                    "text": text,
                    "metadata": {"page": page_num, "pdf_id": pdf_id},
                }
            )
    return chunks
