import ollama

import config

_client = ollama.Client(host=config.OLLAMA_HOST)


def embed_documents(texts: list[str], model: str = config.EMBED_MODEL, batch_size: int = 32) -> list[list[float]]:
    """Embed raw document chunk texts (no instruction prefix)."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = _client.embed(model=model, input=batch)
        vectors.extend(response.embeddings)
    return vectors


def embed_query(query: str, model: str = config.EMBED_MODEL) -> list[float]:
    """Embed a query, prefixed with Qwen3's asymmetric retrieval instruction template."""
    prefixed = config.QUERY_INSTRUCTION.format(query=query)
    response = _client.embed(model=model, input=[prefixed])
    return response.embeddings[0]
