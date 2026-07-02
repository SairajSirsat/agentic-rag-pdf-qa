import os
import pickle
import re

from rank_bm25 import BM25Okapi

import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, pdf_id: str):
        self.pdf_id = pdf_id
        self.path = os.path.join(config.BM25_DIR, f"{pdf_id}.pkl")
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.chunk_lookup: dict[str, dict] = {}

    def build(self, chunks: list[dict]) -> None:
        os.makedirs(config.BM25_DIR, exist_ok=True)
        self.chunk_ids = [c["id"] for c in chunks]
        self.chunk_lookup = {c["id"]: c for c in chunks}
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        with open(self.path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids, "chunk_lookup": self.chunk_lookup}, f)

    def load(self) -> bool:
        if not os.path.exists(self.path):
            return False
        with open(self.path, "rb") as f:
            state = pickle.load(f)
        self.bm25 = state["bm25"]
        self.chunk_ids = state["chunk_ids"]
        self.chunk_lookup = state["chunk_lookup"]
        return True

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: -x[1])
        return [(cid, score) for cid, score in ranked[:k] if score > 0]
