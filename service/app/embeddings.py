from __future__ import annotations

import re
import unicodedata

from app.config import settings
from app.models import EMBEDDING_DIM

class EmbeddingUnavailable(RuntimeError):
    pass

_client = None

def _get_client():
    global _client
    if not settings.cohere_api_key:
        raise EmbeddingUnavailable(
            "No COHERE_API_KEY configured — retrieval is disabled."
        )
    if _client is None:
        import cohere

        _client = cohere.ClientV2(api_key=settings.cohere_api_key)
    return _client

_DIACRITICS = re.compile(r"[ً-ْٰٓ-ٕـ]")
_ALEF = re.compile(r"[آأإٱ]")
_ALEF_MAKSURA = "ى"
_TEH_MARBUTA = "ة"
_WHITESPACE = re.compile(r"\s+")

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = _ALEF.sub("ا", text)
    text = text.replace(_ALEF_MAKSURA, "ي").replace(_TEH_MARBUTA, "ه")
    return _WHITESPACE.sub(" ", text).strip()

def _embed(texts: list[str], input_type: str) -> list[list[float]]:
    client = _get_client()
    prepared = [normalize(text) for text in texts]

    vectors: list[list[float]] = []
    batch = max(1, settings.embed_batch_size)

    for start in range(0, len(prepared), batch):
        response = client.embed(
            texts=prepared[start : start + batch],
            model=settings.embedding_model,
            input_type=input_type,
            embedding_types=["float"],
        )
        vectors.extend(list(v) for v in response.embeddings.float_)

    for vector in vectors:
        if len(vector) != EMBEDDING_DIM:
            raise EmbeddingUnavailable(
                f"{settings.embedding_model} returned {len(vector)} dimensions, "
                f"but the column is fixed at {EMBEDDING_DIM}"
            )
    return vectors

def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed(texts, "search_document") if texts else []

def embed_query(text: str) -> list[float]:
    return _embed([text], "search_query")[0]

def rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    if not documents:
        return []

    client = _get_client()
    response = client.rerank(
        model=settings.rerank_model,
        query=normalize(query),
        documents=[normalize(doc) for doc in documents],
        top_n=min(top_n, len(documents)),
    )
    return [(result.index, float(result.relevance_score)) for result in response.results]

def embed_for_similarity(texts: list[str]) -> list[list[float]]:
    return _embed(texts, "search_document") if texts else []

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
