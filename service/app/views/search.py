from fastapi import APIRouter, HTTPException

from app.config import settings
from app.deps import SessionDep
from app.embeddings import EmbeddingUnavailable, embed_query, rerank
from app.models import EMBEDDING_DIM
from app.repositories import retrieval as retrieval_repo
from app.schemas import SearchIn, SearchOut, SearchResultOut

router = APIRouter()

@router.post("", response_model=SearchOut)
def create_search(payload: SearchIn, session: SessionDep):
    if (payload.q is None) == (payload.embedding is None):
        raise HTTPException(400, "Send exactly one of `q` or `embedding`")

    model = payload.model or settings.embedding_model

    if payload.embedding is not None:
        if len(payload.embedding) != EMBEDDING_DIM:
            raise HTTPException(
                400,
                f"Embedding must have {EMBEDDING_DIM} dimensions, "
                f"got {len(payload.embedding)}",
            )
        vector = payload.embedding
    else:
        try:
            vector = embed_query(payload.q or "")
        except EmbeddingUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc

    use_rerank = payload.rerank if payload.rerank is not None else settings.rerank_enabled
    use_rerank = use_rerank and payload.q is not None
    fetch = max(payload.limit, settings.rerank_candidates) if use_rerank else payload.limit

    hits = retrieval_repo.similar_chunks(
        session,
        vector,
        model=model,
        story_id=payload.story_id,
        lang=payload.lang,
        limit=fetch,
    )

    scores: dict[str, float] = {}
    if use_rerank and hits:
        try:
            ordering = rerank(payload.q or "", [hit.chunk.text for hit in hits], payload.limit)
        except EmbeddingUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        reordered = []
        for index, score in ordering:
            hit = hits[index]
            scores[hit.chunk.id] = score
            reordered.append(hit)
        hits = reordered
    else:
        hits = hits[: payload.limit]

    return SearchOut(
        reranked=bool(scores),
        results=[
            SearchResultOut(
                chunk_id=hit.chunk.id,
                story_id=hit.chunk.story_id,
                document_id=hit.chunk.document_id,
                paragraph=retrieval_repo.chunk_paragraph_position(session, hit.chunk),
                lang=hit.chunk.lang,
                text=hit.chunk.text,
                distance=hit.distance,
                score=hit.score,
                rerank_score=scores.get(hit.chunk.id),
            )
            for hit in hits
        ],
    )
