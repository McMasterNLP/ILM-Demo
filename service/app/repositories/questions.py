from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import embeddings as embed
from app import questions as generator
from app.config import settings
from app.models import (
    Chunk,
    QuestionEmbedding,
    Option,
    OptionTranslation,
    Question,
    QuestionTranslation,
)
from app.repositories.authoring import finish_run, start_run

PROPOSED = "proposed"

_cosine = embed.cosine

def _check_repeat(
    session: Session, story_id: str, lang: str, prompt: str, asked: list[str]
) -> tuple[bool, str | None, list[float] | None]:
    if generator.is_duplicate(prompt, asked):
        return True, None, None

    resembles, vector = most_similar(session, story_id, lang, prompt)
    return False, resembles, vector

def prompt_vectors(
    session: Session, story_id: str, lang: str
) -> list[tuple[str, list[float]]]:
    stmt = (
        select(QuestionTranslation.prompt, QuestionEmbedding.embedding)
        .join(Question, Question.id == QuestionEmbedding.question_id)
        .join(
            QuestionTranslation,
            (QuestionTranslation.question_id == Question.id)
            & (QuestionTranslation.lang == lang),
        )
        .where(
            Question.story_id == story_id,
            QuestionEmbedding.model == settings.embedding_model,
            QuestionEmbedding.lang == lang,
        )
    )
    return [(prompt, list(vector)) for prompt, vector in session.execute(stmt)]

def most_similar(
    session: Session, story_id: str, lang: str, prompt: str
) -> tuple[str | None, list[float] | None]:
    try:
        vector = embed.embed_for_similarity([prompt])[0]
    except embed.EmbeddingUnavailable:
        return None, None

    best_prompt, best_score = None, 0.0
    for other_prompt, other_vector in prompt_vectors(session, story_id, lang):
        score = _cosine(vector, other_vector)
        if score > best_score:
            best_prompt, best_score = other_prompt, score

    if best_score >= settings.similarity_warning:
        return best_prompt, vector
    return None, vector

def store_prompt_vector(
    session: Session, question_id: str, lang: str, vector: list[float] | None
) -> None:
    if vector is None:
        return
    session.add(
        QuestionEmbedding(
            question_id=question_id,
            model=settings.embedding_model,
            lang=lang,
            embedding=vector,
        )
    )
    session.flush()

def existing_prompts(session: Session, chunk_id: str) -> list[str]:
    stmt = (
        select(QuestionTranslation.prompt)
        .join(Question, Question.id == QuestionTranslation.question_id)
        .where(Question.evidence_chunk_id == chunk_id, Question.followup_of.is_(None))
    )
    return [p for p in session.scalars(stmt) if p]

def _coverage(session: Session, story_id: str) -> dict[str, int]:
    stmt = (
        select(Question.evidence_chunk_id)
        .where(Question.story_id == story_id, Question.followup_of.is_(None))
    )
    counts: dict[str, int] = {}
    for chunk_id in session.scalars(stmt):
        if chunk_id:
            counts[chunk_id] = counts.get(chunk_id, 0) + 1
    return counts

def uncovered_chunks(session: Session, story_id: str, lang: str) -> list[Chunk]:
    counts = _coverage(session, story_id)
    stmt = (
        select(Chunk)
        .where(Chunk.story_id == story_id, Chunk.lang == lang)
        .order_by(Chunk.position)
    )
    return [c for c in session.scalars(stmt) if counts.get(c.id, 0) == 0]

def select_chunks(session: Session, story_id: str, lang: str, limit: int) -> list[Chunk]:
    chunks = list(
        session.scalars(
            select(Chunk)
            .where(Chunk.story_id == story_id, Chunk.lang == lang)
            .order_by(Chunk.position)
        )
    )
    if not chunks:
        return []

    counts = _coverage(session, story_id)
    if counts:
        chunks.sort(key=lambda c: (counts.get(c.id, 0), c.position))

    if len(chunks) <= limit:
        return chunks

    vectors: dict[str, list[float]] = {}
    for chunk in chunks:
        for embedding in chunk.embeddings:
            if embedding.model == settings.embedding_model:
                vectors[chunk.id] = list(embedding.embedding)
                break

    if len(vectors) < len(chunks):
        step = max(1, len(chunks) // limit)
        return chunks[::step][:limit]

    chosen = [chunks[0]]
    remaining = chunks[1:]
    while len(chosen) < limit and remaining:
        furthest = min(
            remaining,
            key=lambda c: max(_cosine(vectors[c.id], vectors[p.id]) for p in chosen),
        )
        chosen.append(furthest)
        remaining.remove(furthest)

    return sorted(chosen, key=lambda c: c.position)

def clear_proposed(session: Session, story_id: str) -> int:
    stale = list(
        session.scalars(
            select(Question).where(
                Question.story_id == story_id,
                Question.status == PROPOSED,
                Question.type == "comprehension",
            )
        )
    )
    for question in stale:
        session.delete(question)
    session.flush()
    return len(stale)

def generate_for_story(
    session: Session,
    story_id: str,
    lang: str,
    *,
    count: int = 6,
    age_range: str = "7-10",
    answer_mode: str = "choice",
    replace: bool = False,
    cover_all: bool = False,
) -> tuple[list[Question], list[str]]:
    if replace:
        clear_proposed(session, story_id)

    if cover_all:
        chunks = uncovered_chunks(session, story_id, lang)
        if not chunks:
            raise ValueError("Every paragraph already has a question")
    else:
        chunks = select_chunks(session, story_id, lang, count)
    if not chunks:
        raise ValueError("Story has no indexed paragraphs to ask about")

    run = start_run(
        session,
        stage="question-generation",
        model=settings.gemini_model,
        params={
            "story_id": story_id,
            "lang": lang,
            "chunks": len(chunks),
            "answer_mode": answer_mode,
            "replace": replace,
            "cover_all": cover_all,
        },
    )

    highest = session.scalar(
        select(Question.position)
        .where(Question.story_id == story_id)
        .order_by(Question.position.desc())
        .limit(1)
    )
    position = (highest or 0) + 1

    created: list[Question] = []
    failures: list[str] = []

    for chunk in chunks:
        if answer_mode == "open":
            created_open = _draft_open(
                session,
                story_id,
                chunk,
                lang,
                age_range,
                position,
                run.id,
                failures,
                existing_prompts(session, chunk.id),
            )
            if created_open is not None:
                created.append(created_open)
                position += 1
            continue

        asked = existing_prompts(session, chunk.id)
        try:
            drafted = generator.generate_question(chunk.text, lang, age_range, asked)
        except generator.GenerationUnavailable as exc:
            failures.append(f"paragraph {chunk.position + 1}: {exc}")
            continue

        reject, resembles, vector = _check_repeat(
            session, story_id, lang, drafted.prompt, asked
        )
        if reject:
            try:
                drafted = generator.generate_question(
                    chunk.text, lang, age_range, [*asked, drafted.prompt]
                )
            except generator.GenerationUnavailable as exc:
                failures.append(f"paragraph {chunk.position + 1}: {exc}")
                continue
            reject, resembles, vector = _check_repeat(
                session, story_id, lang, drafted.prompt, asked
            )
            if reject:
                failures.append(
                    f"paragraph {chunk.position + 1}: skipped — reworded a question "
                    "already in the bank"
                )
                continue

        if resembles:
            failures.append(
                f"paragraph {chunk.position + 1}: close to an existing question "
                f"({resembles}) — check they are not asking the same thing"
            )

        question = Question(
            story_id=story_id,
            type="comprehension",
            position=position,
            evidence_chunk_id=chunk.id,
            evidence_paragraph_id=chunk.paragraph_id,
            status=PROPOSED,
            run_id=run.id,
        )
        session.add(question)
        session.flush()

        session.add(
            QuestionTranslation(
                question_id=question.id,
                lang=lang,
                prompt=drafted.prompt,
                explanation=drafted.explanation,
            )
        )
        for index, label in enumerate(drafted.options):
            option = Option(
                question_id=question.id,
                position=index,
                is_correct=index == drafted.correct_index,
            )
            session.add(option)
            session.flush()
            session.add(
                OptionTranslation(option_id=option.id, lang=lang, label=label)
            )

        followup = None
        if settings.generate_followups:
            try:
                followup = generator.generate_followup(
                    chunk.text, drafted, lang, age_range
                )
            except generator.GenerationUnavailable as exc:
                failures.append(f"paragraph {chunk.position + 1}: no follow-up ({exc})")

        if followup is not None:
            child = Question(
                story_id=story_id,
                type="comprehension",
                position=position,
                evidence_chunk_id=chunk.id,
                evidence_paragraph_id=chunk.paragraph_id,
                followup_of=question.id,
                status=PROPOSED,
                run_id=run.id,
            )
            session.add(child)
            session.flush()
            session.add(
                QuestionTranslation(
                    question_id=child.id,
                    lang=lang,
                    prompt=followup.prompt,
                    explanation=followup.explanation,
                )
            )
            for index, label in enumerate(followup.options):
                option = Option(
                    question_id=child.id,
                    position=index,
                    is_correct=index == followup.correct_index,
                )
                session.add(option)
                session.flush()
                session.add(
                    OptionTranslation(option_id=option.id, lang=lang, label=label)
                )

        if not generator.looks_grounded(drafted, chunk.text):
            failures.append(
                f"paragraph {chunk.position + 1}: answer shares no wording with the "
                "source — check it closely"
            )

        store_prompt_vector(session, question.id, lang, vector)
        created.append(question)
        position += 1

    finish_run(session, run)
    session.flush()
    return created, failures

def list_questions(session: Session, story_id: str) -> list[Question]:
    stmt = (
        select(Question)
        .where(Question.story_id == story_id, Question.followup_of.is_(None))
        .order_by(Question.position)
    )
    return list(session.scalars(stmt))

def set_status(session: Session, question_id: str, status: str) -> Question | None:
    question = session.get(Question, question_id)
    if question is None:
        return None

    question.status = status
    for child in session.scalars(
        select(Question).where(Question.followup_of == question.id)
    ):
        child.status = status

    session.flush()
    return question

def followup_of(session: Session, question_id: str) -> Question | None:
    return session.scalars(
        select(Question).where(Question.followup_of == question_id)
    ).first()

def delete_question(session: Session, question_id: str) -> bool:
    question = session.get(Question, question_id)
    if question is None:
        return False
    session.delete(question)
    session.flush()
    return True

def _draft_open(
    session: Session,
    story_id: str,
    chunk,
    lang: str,
    age_range: str,
    position: int,
    run_id: str,
    failures: list[str],
    asked: list[str],
) -> Question | None:
    try:
        drafted = generator.generate_open_question(chunk.text, lang, age_range, asked)
    except generator.GenerationUnavailable as exc:
        failures.append(f"paragraph {chunk.position + 1}: {exc}")
        return None

    reject, resembles, vector = _check_repeat(
        session, story_id, lang, drafted.prompt, asked
    )
    if reject:
        failures.append(
            f"paragraph {chunk.position + 1}: skipped — reworded a question already "
            "in the bank"
        )
        return None
    if resembles:
        failures.append(
            f"paragraph {chunk.position + 1}: close to an existing question "
            f"({resembles}) — check they are not asking the same thing"
        )

    question = Question(
        story_id=story_id,
        type="comprehension",
        answer_mode="open",
        position=position,
        evidence_chunk_id=chunk.id,
        evidence_paragraph_id=chunk.paragraph_id,
        status=PROPOSED,
        run_id=run_id,
    )
    session.add(question)
    session.flush()

    session.add(
        QuestionTranslation(
            question_id=question.id,
            lang=lang,
            prompt=drafted.prompt,
            explanation=drafted.explanation,
            reference_answer=drafted.reference_answer,
        )
    )
    store_prompt_vector(session, question.id, lang, vector)
    session.flush()
    return question
