from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app import embeddings
from app.config import settings
from app.models import Chunk, ChunkEmbedding, ExtractionRun, Paragraph

@dataclass(slots=True)
class Retrieved:
    chunk: Chunk
    distance: float

    @property
    def score(self) -> float:
        return max(0.0, 1.0 - self.distance / 2.0)

def similar_chunks(
    session: Session,
    embedding: list[float],
    *,
    model: str,
    story_id: str | None = None,
    lang: str | None = None,
    limit: int = 5,
) -> list[Retrieved]:
    distance = ChunkEmbedding.embedding.cosine_distance(embedding).label("distance")

    stmt = (
        select(Chunk, distance)
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .where(ChunkEmbedding.model == model)
        .order_by(distance)
        .limit(limit)
    )
    if story_id is not None:
        stmt = stmt.where(Chunk.story_id == story_id)
    if lang is not None:
        stmt = stmt.where(Chunk.lang == lang)

    return [Retrieved(chunk=chunk, distance=float(dist)) for chunk, dist in session.execute(stmt)]

def chunk_paragraph_position(session: Session, chunk: Chunk) -> int | None:
    if chunk.paragraph_id is None:
        return None
    paragraph = session.get(Paragraph, chunk.paragraph_id)
    return paragraph.position if paragraph else None

def embed_chunks(
    session: Session, chunks: list[Chunk], *, run: ExtractionRun | None = None
) -> int:
    if not chunks:
        return 0

    model = settings.embedding_model
    vectors = embeddings.embed_documents([chunk.text for chunk in chunks])

    session.execute(
        delete(ChunkEmbedding).where(
            ChunkEmbedding.chunk_id.in_([chunk.id for chunk in chunks]),
            ChunkEmbedding.model == model,
        )
    )

    for chunk, vector in zip(chunks, vectors, strict=True):
        session.add(
            ChunkEmbedding(
                chunk_id=chunk.id,
                model=model,
                dim=len(vector),
                embedding=vector,
                run_id=run.id if run else None,
            )
        )

    session.flush()
    return len(vectors)

def unembedded_chunks(session: Session, model: str, limit: int | None = None) -> list[Chunk]:
    indexed = select(ChunkEmbedding.chunk_id).where(ChunkEmbedding.model == model)
    stmt = select(Chunk).where(Chunk.id.not_in(indexed)).order_by(Chunk.position)
    if limit:
        stmt = stmt.limit(limit)
    return list(session.scalars(stmt))
