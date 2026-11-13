from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy import select

from app.deps import AdminDep, SessionDep
from app.models import Chunk, Document, Paragraph, ParagraphTranslation, Question, Story
from app.repositories import authoring as authoring_repo
from app.repositories import composition as composition_repo
from app.repositories import ingest as ingest_repo
from app.repositories import questions as questions_repo
from app.questions import GenerationUnavailable
from app.schemas import (
    DocumentOut,
    QuizShape,
    GenerateQuestionsRequest,
    QuestionOptionOut,
    QuestionReviewOut,
    QuestionSetOut,
    PublishRequest,
    ReviewParagraphOut,
    StoryReviewOut,
)

router = APIRouter()

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}

def _document_out(document: Document, chunk_count: int) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        lang=document.lang,
        status=document.status,
        source_id=document.source_id,
        story_id=document.story_id,
        chunk_count=chunk_count,
    )

def _review_row(session, row: ParagraphTranslation) -> ReviewParagraphOut:
    paragraph = session.get(Paragraph, row.paragraph_id)
    return ReviewParagraphOut(
        paragraph_id=row.paragraph_id,
        position=paragraph.position if paragraph else 0,
        lang=row.lang,
        body=row.body,
        status=row.status,
        approved_by=row.approved_by,
    )

def _story_review(session, story: Story) -> StoryReviewOut:
    rows = list(
        session.scalars(
            select(ParagraphTranslation)
            .join(Paragraph, Paragraph.id == ParagraphTranslation.paragraph_id)
            .where(Paragraph.story_id == story.id)
            .order_by(Paragraph.position)
        )
    )
    approved, total = authoring_repo.approval_state(session, story.id)
    embedded, chunks = authoring_repo.indexing_state(session, story.id)
    title = next((t.title for t in story.translations if t.title), story.id)

    return StoryReviewOut(
        story_id=story.id,
        title=title,
        published=story.published,
        approved=approved,
        total=total,
        chunk_count=chunks,
        embedded_count=embedded,
        quiz_size=story.quiz_size,
        quiz_relationship=story.quiz_relationship,
        quiz_identification=story.quiz_identification,
        quiz_comprehension=story.quiz_comprehension,
        paragraphs=[_review_row(session, row) for row in rows],
    )

@router.post("/stories", response_model=StoryReviewOut)
async def create_story(
    session: SessionDep,
    uploader: AdminDep,
    file: UploadFile | None = File(None),
    text: str = Form(""),
    title: str = Form(...),
    summary: str = Form(""),
    lang: str = Form("en"),
    level: str = Form("beginner"),
    minutes: int = Form(0),
    story_id: str = Form(""),
    source_title: str = Form(...),
):
    pasted = text.strip()
    has_file = file is not None and bool(file.filename)

    if has_file == bool(pasted):
        raise HTTPException(400, "Send either a file or pasted text, not both")

    if has_file:
        assert file is not None
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(400, f"Unsupported file type {suffix or '(none)'}")
        filename = file.filename or "upload"
        media_type = file.content_type
        payload = await file.read()
    else:
        filename = f"{authoring_repo.slugify(title, 'story')}.txt"
        media_type = "text/plain"
        payload = pasted.encode("utf-8")

    source = ingest_repo.get_or_create_source(session, source_title)

    document = Document(
        source_id=source.id,
        filename=filename,
        media_type=media_type,
        stored_path="",
        lang=lang,
        status="draft",
    )
    session.add(document)
    session.flush()

    path = ingest_repo.store_upload(filename, payload, document.id)
    document.stored_path = str(path)

    if has_file:
        try:
            body = ingest_repo.extract_text(path, media_type)
        except ingest_repo.UnsupportedDocument as exc:
            raise HTTPException(400, str(exc)) from exc
    else:
        body = pasted

    if not body.strip():
        raise HTTPException(400, "No text could be read")

    try:
        story = authoring_repo.build_story(
            session,
            document=document,
            text=body,
            story_id=story_id.strip() or None,
            title=title.strip(),
            summary=summary.strip(),
            level=level,
            minutes=minutes,
            uploaded_by=uploader,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    return _story_review(session, story)

@router.get("/documents", response_model=list[DocumentOut])
def list_documents(session: SessionDep):
    documents = list(session.scalars(select(Document).order_by(Document.created_at.desc())))
    return [
        _document_out(doc, len(ingest_repo.document_chunks(session, doc.id)))
        for doc in documents
    ]

@router.get("/stories/{story_id}/review", response_model=StoryReviewOut)
def retrieve_review(story_id: str, session: SessionDep):
    story = session.get(Story, story_id)
    if story is None:
        raise HTTPException(404, "Story not found")
    return _story_review(session, story)

@router.post("/paragraphs/{paragraph_id}/{lang}/approve", response_model=ReviewParagraphOut)
def approve_paragraph(paragraph_id: str, lang: str, session: SessionDep, reviewer: AdminDep):
    row = authoring_repo.approve_paragraph(session, paragraph_id, lang, reviewer)
    if row is None:
        raise HTTPException(404, "Paragraph translation not found")
    return _review_row(session, row)

@router.post("/paragraphs/{paragraph_id}/{lang}/reject", response_model=ReviewParagraphOut)
def reject_paragraph(paragraph_id: str, lang: str, session: SessionDep):
    row = authoring_repo.reject_paragraph(session, paragraph_id, lang)
    if row is None:
        raise HTTPException(404, "Paragraph translation not found")
    return _review_row(session, row)

@router.post("/stories/{story_id}/publish", response_model=StoryReviewOut)
def publish_story(story_id: str, payload: PublishRequest, session: SessionDep):
    story = session.get(Story, story_id)
    if story is None:
        raise HTTPException(404, "Story not found")

    if payload.published:
        approved, total = authoring_repo.approval_state(session, story.id)
        if total == 0 or approved < total:
            raise HTTPException(409, f"{total - approved} paragraph(s) still need approval")

        authoring_repo.embed_story(session, story.id)
        embedded, chunks = authoring_repo.indexing_state(session, story.id)
        if chunks == 0 or embedded < chunks:
            raise HTTPException(
                409,
                f"{chunks - embedded} paragraph(s) not indexed yet — "
                "check COHERE_API_KEY, then publish again",
            )

    story.published = payload.published
    session.flush()
    return _story_review(session, story)

def _question_out(session, question: Question, lang: str) -> QuestionReviewOut:
    translation = next(
        (t for t in question.translations if t.lang == lang), None
    ) or (question.translations[0] if question.translations else None)

    chunk = session.get(Chunk, question.evidence_chunk_id) if question.evidence_chunk_id else None
    paragraph = (
        session.get(Paragraph, question.evidence_paragraph_id)
        if question.evidence_paragraph_id
        else None
    )

    options: list[QuestionOptionOut] = []
    for option in question.options:
        label = next((t for t in option.translations if t.lang == lang), None) or (
            option.translations[0] if option.translations else None
        )
        options.append(
            QuestionOptionOut(
                id=option.id,
                label=label.label if label else "",
                is_correct=option.is_correct,
            )
        )

    child = questions_repo.followup_of(session, question.id) if not question.followup_of else None

    return QuestionReviewOut(
        id=question.id,
        type=question.type,
        answer_mode=question.answer_mode,
        position=question.position,
        status=question.status,
        lang=translation.lang if translation else lang,
        prompt=translation.prompt if translation else "",
        explanation=translation.explanation if translation else "",
        reference_answer=translation.reference_answer if translation else "",
        options=options,
        evidence_paragraph=paragraph.position if paragraph else None,
        source_text=chunk.text if chunk else None,
        follow_up=_question_out(session, child, lang) if child else None,
    )

def _question_set(session, story_id: str, lang: str, warnings: list[str]) -> QuestionSetOut:
    rows = questions_repo.list_questions(session, story_id)
    covered, paragraphs = composition_repo.coverage(session, story_id)
    return QuestionSetOut(
        story_id=story_id,
        approved=sum(1 for q in rows if q.status == "approved"),
        total=len(rows),
        covered=covered,
        paragraphs=paragraphs,
        questions=[_question_out(session, q, lang) for q in rows],
        warnings=warnings,
    )

@router.post("/stories/{story_id}/questions", response_model=QuestionSetOut)
def create_questions(story_id: str, payload: GenerateQuestionsRequest, session: SessionDep):
    story = session.get(Story, story_id)
    if story is None:
        raise HTTPException(404, "Story not found")

    try:
        _, warnings = questions_repo.generate_for_story(
            session,
            story_id,
            payload.lang,
            count=payload.count,
            age_range=payload.age_range,
            answer_mode=payload.answer_mode,
            replace=payload.replace,
            cover_all=payload.cover_all,
        )
    except GenerationUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, f"Question generation failed: {exc}") from exc

    return _question_set(session, story_id, payload.lang, warnings)

@router.get("/stories/{story_id}/questions", response_model=QuestionSetOut)
def retrieve_questions(story_id: str, session: SessionDep, lang: str = "en"):
    story = session.get(Story, story_id)
    if story is None:
        raise HTTPException(404, "Story not found")
    return _question_set(session, story_id, lang, [])

@router.post("/questions/{question_id}/approve", response_model=QuestionReviewOut)
def approve_question(question_id: str, session: SessionDep, reviewer: AdminDep):
    question = questions_repo.set_status(session, question_id, "approved")
    if question is None:
        raise HTTPException(404, "Question not found")
    lang = question.translations[0].lang if question.translations else "en"
    return _question_out(session, question, lang)

@router.post("/questions/{question_id}/reject", response_model=QuestionReviewOut)
def reject_question(question_id: str, session: SessionDep):
    question = questions_repo.set_status(session, question_id, "proposed")
    if question is None:
        raise HTTPException(404, "Question not found")
    lang = question.translations[0].lang if question.translations else "en"
    return _question_out(session, question, lang)

@router.delete("/questions/{question_id}")
def destroy_question(question_id: str, session: SessionDep):
    if not questions_repo.delete_question(session, question_id):
        raise HTTPException(404, "Question not found")
    return {"ok": True}

@router.put("/stories/{story_id}/quiz-shape", response_model=StoryReviewOut)
def update_quiz_shape(story_id: str, payload: QuizShape, session: SessionDep):
    story = session.get(Story, story_id)
    if story is None:
        raise HTTPException(404, "Story not found")

    story.quiz_size = payload.quiz_size
    story.quiz_relationship = payload.quiz_relationship
    story.quiz_identification = payload.quiz_identification
    story.quiz_comprehension = payload.quiz_comprehension
    session.flush()
    return _story_review(session, story)
