from sentence_transformers import CrossEncoder

import config

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        try:
            _model = CrossEncoder(config.RERANK_MODEL, device=config.RERANK_DEVICE)
        except Exception:
            _model = CrossEncoder(config.RERANK_MODEL, device="cpu")
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = config.RERANK_TOP_K) -> list[dict]:
    if not candidates:
        return []
    model = _get_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)
    return sorted(candidates, key=lambda c: -c["rerank_score"])[:top_k]
