from __future__ import annotations

import re
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.embeddings import EmbeddingUnavailable
from app.config import settings
from app.models import (
    Chunk,
    ChunkEmbedding,
    Document,
    ExtractionRun,
    Paragraph,
    ParagraphTranslation,
    Story,
    StoryTranslation,
)
from app.ml.kg_bridge import run_kg_extraction
from app.repositories.kg_write import (
    persist_prepared_records,
    retract_paragraph_kg,
)

def slugify(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned[:50] or fallback

def start_run(session: Session, stage: str, model: str, params: dict | None = None) -> ExtractionRun:
    run = ExtractionRun(stage=stage, model=model, params=params or {})
    session.add(run)
    session.flush()
    return run

def finish_run(session: Session, run: ExtractionRun) -> None:
    run.finished_at = datetime.now(timezone.utc)
    session.flush()

def split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def build_story(
    session: Session,
    *,
    document: Document,
    text: str,
    story_id: str | None,
    title: str,
    summary: str,
    level: str,
    minutes: int,
    uploaded_by: str,
) -> Story:
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        raise ValueError("No paragraphs found ? separate them with blank lines")

    target = story_id or slugify(title, document.id[:8])
    lang = document.lang

    story = session.get(Story, target)
    if story is None:
        story = Story(
            id=target,
            level=level,
            minutes=minutes or max(3, len(paragraphs) // 2),
            character_count=0,
            position=0,
            published=False,
        )
        session.add(story)
        session.flush()
    else:
        story.level = level
        story.minutes = minutes or story.minutes

    #
    # Story metadata is accepted at ingestion.
    # Paragraph text is reviewed independently.
    #
    session.merge(
        StoryTranslation(
            story_id=story.id,
            lang=lang,
            title=title,
            summary=summary,
            heading=title,
            status="approved",
        )
    )
    session.flush()

    existing_by_position = {
        row.position: row
        for row in session.scalars(
            select(Paragraph)
            .where(Paragraph.story_id == story.id)
            .order_by(Paragraph.position)
        )
    }

    #
    # Merge the uploaded language by paragraph position.
    #
    # This is critical for bilingual stories:
    # uploading English must not delete Arabic and vice versa.
    #
    for position, body in enumerate(paragraphs):
        paragraph = existing_by_position.get(position)

        if paragraph is None:
            paragraph = Paragraph(
                story_id=story.id,
                position=position,
                is_pending=False,
            )
            session.add(paragraph)
            session.flush()
            existing_by_position[position] = paragraph
        else:
            paragraph.is_pending = False

        translation = session.get(
            ParagraphTranslation,
            {
                "paragraph_id": paragraph.id,
                "lang": lang,
            },
        )

        if translation is None:
            session.add(
                ParagraphTranslation(
                    paragraph_id=paragraph.id,
                    lang=lang,
                    body=body,
                    status="draft",
                    source_id=document.source_id,
                    approved_by=None,
                    approved_at=None,
                )
            )
            continue

        translation.source_id = document.source_id

        #
        # Preserve unchanged approved paragraphs.
        #
        # Changed content returns to draft and stale RAG/KG state is
        # retracted before it can be approved again.
        #
        if translation.body != body:
            if lang == "ar":
                retract_paragraph_kg(
                    session,
                    paragraph_id=paragraph.id,
                    lang=lang,
                )

            translation.body = body
            translation.status = "draft"
            translation.approved_by = None
            translation.approved_at = None

            session.flush()

            sync_paragraph_chunk(
                session,
                paragraph.id,
                lang,
            )

    #
    # If this language is re-uploaded with fewer paragraphs,
    # remove only that language's extra translations.
    #
    for position, paragraph in list(existing_by_position.items()):
        if position < len(paragraphs):
            continue

        translation = session.get(
            ParagraphTranslation,
            {
                "paragraph_id": paragraph.id,
                "lang": lang,
            },
        )

        if translation is None:
            continue

        if lang == "ar":
            retract_paragraph_kg(
                session,
                paragraph_id=paragraph.id,
                lang=lang,
            )

        session.delete(translation)
        session.flush()

        sync_paragraph_chunk(
            session,
            paragraph.id,
            lang,
        )

        remaining_translation = session.scalars(
            select(ParagraphTranslation).where(
                ParagraphTranslation.paragraph_id == paragraph.id
            )
        ).first()

        if remaining_translation is None:
            session.delete(paragraph)

    document.story_id = story.id
    document.status = "approved"

    session.flush()

    #
    # No indexing or embedding here.
    #
    # approve_paragraph() is the single activation path:
    #
    # English -> RAG
    # Arabic  -> KG + RAG
    #
    return story

def index_story_chunks(
    session: Session, story_id: str, lang: str, *, document_id: str | None = None
) -> list[Chunk]:
    for existing in session.scalars(
        select(Chunk).where(Chunk.story_id == story_id, Chunk.lang == lang)
    ).all():
        session.delete(existing)
    session.flush()

    paragraphs = list(
        session.scalars(
            select(Paragraph).where(Paragraph.story_id == story_id).order_by(Paragraph.position)
        )
    )

    chunks: list[Chunk] = []
    for paragraph in paragraphs:
        translation = session.get(
            ParagraphTranslation, {"paragraph_id": paragraph.id, "lang": lang}
        )
        if translation is None or not translation.body.strip():
            continue
        if translation.status != "approved":
            continue
        chunk = Chunk(
            story_id=story_id,
            paragraph_id=paragraph.id,
            document_id=document_id,
            lang=lang,
            position=paragraph.position,
            text=translation.body,
            token_count=len(translation.body) // 4,
        )
        session.add(chunk)
        chunks.append(chunk)

    session.flush()
    return chunks

def sync_paragraph_chunk(
    session: Session, paragraph_id: str, lang: str, *, document_id: str | None = None
) -> Chunk | None:
    paragraph = session.get(Paragraph, paragraph_id)
    translation = session.get(
        ParagraphTranslation, {"paragraph_id": paragraph_id, "lang": lang}
    )
    if paragraph is None:
        return None

    existing = session.scalars(
        select(Chunk).where(Chunk.paragraph_id == paragraph_id, Chunk.lang == lang)
    ).first()

    approved = (
        translation is not None
        and translation.status == "approved"
        and bool(translation.body.strip())
    )
    if not approved:
        if existing is not None:
            session.delete(existing)
            session.flush()
        return None

    if existing is None:
        existing = Chunk(
            story_id=paragraph.story_id,
            paragraph_id=paragraph.id,
            document_id=document_id,
            lang=lang,
            position=paragraph.position,
        )
        session.add(existing)

    if existing.text != translation.body:
        session.execute(
            delete(ChunkEmbedding).where(ChunkEmbedding.chunk_id == existing.id)
        )
    existing.text = translation.body
    existing.position = paragraph.position
    existing.token_count = len(translation.body) // 4
    session.flush()
    return existing

def embed_story(session: Session, story_id: str) -> int:
    from app.repositories import retrieval as retrieval_repo

    pending = [
        chunk
        for chunk in session.scalars(select(Chunk).where(Chunk.story_id == story_id))
        if not any(e.model == settings.embedding_model for e in chunk.embeddings)
    ]
    if not pending:
        return 0

    try:
        # Embedding is secondary to paragraph approval/KG persistence.
        # Isolate it in a savepoint so an unavailable embedding provider
        # does not roll back the outer approval transaction.
        with session.begin_nested():
            run = start_run(
                session,
                stage="embedding",
                model=settings.embedding_model,
                params={"story_id": story_id, "chunks": len(pending)},
            )
            count = retrieval_repo.embed_chunks(
                session,
                pending,
                run=run,
            )
            finish_run(session, run)

        return count

    except EmbeddingUnavailable:
        return 0

def indexing_state(session: Session, story_id: str) -> tuple[int, int]:
    chunks = list(session.scalars(select(Chunk).where(Chunk.story_id == story_id)))
    embedded = sum(
        1
        for chunk in chunks
        if any(e.model == settings.embedding_model for e in chunk.embeddings)
    )
    return embedded, len(chunks)

def approve_paragraph(
    session: Session, paragraph_id: str, lang: str, reviewer: str
) -> ParagraphTranslation | None:
    row = session.get(
        ParagraphTranslation,
        {
            "paragraph_id": paragraph_id,
            "lang": lang,
        },
    )
    if row is None:
        return None

    paragraph = session.get(Paragraph, paragraph_id)

    #
    # Arabic KG extraction is part of the same DB transaction as approval.
    #
    # Re-approval first retracts the previous paragraph-derived KG state,
    # then rebuilds it from the current paragraph body.
    #
    # If extraction/persistence raises, get_session() rolls back the whole
    # request, including this retraction and the approval state change.
    #
    if lang == "ar" and paragraph is not None:
        retract_paragraph_kg(
            session,
            paragraph_id=paragraph_id,
            lang=lang,
        )

        kg_result = run_kg_extraction(
            story_id=paragraph.story_id,
            story_key=paragraph.story_id,
            paragraph_id=paragraph_id,
            lang=lang,
            text=row.body,
        )

        persist_prepared_records(
            session,
            records=kg_result.records,
        )

    row.status = "approved"
    row.approved_by = reviewer
    row.approved_at = datetime.now(timezone.utc)
    session.flush()

    sync_paragraph_chunk(
        session,
        paragraph_id,
        lang,
    )

    if paragraph is not None:
        embed_story(
            session,
            paragraph.story_id,
        )

    return row

def reject_paragraph(
    session: Session,
    paragraph_id: str,
    lang: str,
) -> ParagraphTranslation | None:
    row = session.get(
        ParagraphTranslation,
        {
            "paragraph_id": paragraph_id,
            "lang": lang,
        },
    )
    if row is None:
        return None

    #
    # Only Arabic paragraphs currently contribute to the research KG.
    #
    # Remove this paragraph's mentions/evidence. Shared StoryRelation
    # appearances remain when another paragraph still supports them.
    #
    if lang == "ar":
        retract_paragraph_kg(
            session,
            paragraph_id=paragraph_id,
            lang=lang,
        )

    row.status = "draft"
    row.approved_by = None
    row.approved_at = None
    session.flush()

    sync_paragraph_chunk(
        session,
        paragraph_id,
        lang,
    )

    return row

def approval_state(session: Session, story_id: str) -> tuple[int, int]:
    stmt = (
        select(ParagraphTranslation.status)
        .join(Paragraph, Paragraph.id == ParagraphTranslation.paragraph_id)
        .where(Paragraph.story_id == story_id)
    )
    statuses = list(session.scalars(stmt))
    return sum(1 for s in statuses if s == "approved"), len(statuses)
