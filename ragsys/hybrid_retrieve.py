import config
from ragsys.bm25_index import BM25Index
from ragsys.embeddings import embed_query
from ragsys.vectorstore import VectorStore


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = config.RRF_K) -> dict[str, float]:
    """RRF over multiple ranked ID lists. Union of all lists; missing-from-one-list is fine."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_retrieve(
    query: str,
    vectorstore: VectorStore,
    bm25: BM25Index,
    dense_k: int = config.DENSE_TOP_K,
    bm25_k: int = config.BM25_TOP_K,
    fused_k: int = config.FUSED_TOP_K,
) -> list[dict]:
    dense_results = vectorstore.query(embed_query(query), dense_k)
    bm25_results = bm25.search(query, bm25_k)

    lookup: dict[str, dict] = {r["id"]: r for r in dense_results}
    for cid, _ in bm25_results:
        if cid not in lookup and cid in bm25.chunk_lookup:
            chunk = bm25.chunk_lookup[cid]
            lookup[cid] = {"id": cid, "text": chunk["text"], "metadata": chunk["metadata"]}

    dense_ids = [r["id"] for r in dense_results]
    bm25_ids = [cid for cid, _ in bm25_results]
    fused_scores = reciprocal_rank_fusion([dense_ids, bm25_ids])

    ranked_ids = sorted(fused_scores.keys(), key=lambda cid: -fused_scores[cid])[:fused_k]
    return [lookup[cid] for cid in ranked_ids if cid in lookup]
