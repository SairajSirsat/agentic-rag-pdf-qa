import chromadb

import config

_client = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return _client


class VectorStore:
    def __init__(self, pdf_id: str):
        self.pdf_id = pdf_id
        self.collection = _get_client().get_or_create_collection(name=f"pdf_{pdf_id}")

    def upsert(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        self.collection.upsert(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

    def query(self, query_embedding: list[float], k: int) -> list[dict]:
        result = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        if not result["ids"] or not result["ids"][0]:
            return []
        out = []
        for i in range(len(result["ids"][0])):
            out.append(
                {
                    "id": result["ids"][0][i],
                    "text": result["documents"][0][i],
                    "metadata": result["metadatas"][0][i],
                    "distance": result["distances"][0][i],
                }
            )
        return out
