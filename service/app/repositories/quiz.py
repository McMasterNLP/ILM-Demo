from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    AttemptAnswer,
    AttemptQuestion,
    Chunk,
    ConceptMastery,
    Option,
    Paragraph,
    Question,
    QuizAttempt,
)
from app.repositories.content import APPROVED, localized
from app.schemas import OptionOut, QuestionOut

def _paragraph_position(session: Session, paragraph_id: str | None) -> int | None:
    if paragraph_id is None:
        return None
    paragraph = session.get(Paragraph, paragraph_id)
    return paragraph.position if paragraph else None

def to_question_out(session: Session, question: Question) -> QuestionOut:
    return QuestionOut(
        id=question.id,
        type=question.type,
        position=question.position,
        prompt=localized(question.translations, "prompt"),
        answer_mode=question.answer_mode,
        options=[
            OptionOut(id=option.id, label=localized(option.translations, "label"))
            for option in question.options
        ],
        evidence_paragraph=_paragraph_position(session, question.evidence_paragraph_id),
    )

def story_questions(session: Session, story_id: str) -> list[Question]:
    stmt = (
        select(Question)
        .where(
            Question.story_id == story_id,
            Question.status == APPROVED,
            Question.followup_of.is_(None),
        )
        .order_by(Question.position, Question.id)
    )
    return list(session.scalars(stmt))

def followup_for(session: Session, question_id: str) -> Question | None:
    stmt = select(Question).where(
        Question.followup_of == question_id, Question.status == APPROVED
    )
    return session.scalars(stmt).first()

def start_attempt(session: Session, user_id: str, story_id: str, total: int) -> QuizAttempt:
    attempt = QuizAttempt(user_id=user_id, story_id=story_id, score=0, total=total)
    session.add(attempt)
    session.flush()
    return attempt

def find_unfinished_attempt(
    session: Session, user_id: str, story_id: str
) -> QuizAttempt | None:
    stmt = (
        select(QuizAttempt)
        .where(
            QuizAttempt.user_id == user_id,
            QuizAttempt.story_id == story_id,
            QuizAttempt.finished_at.is_(None),
        )
        .order_by(QuizAttempt.started_at.desc())
    )
    return session.scalars(stmt).first()

def answered_question_ids(attempt: QuizAttempt) -> list[str]:
    return [a.question_id for a in attempt.answers if not a.was_followup]


def selected_question_ids(
    session: Session,
    attempt_id: str,
) -> list[str]:
    stmt = (
        select(AttemptQuestion.question_id)
        .where(AttemptQuestion.attempt_id == attempt_id)
        .order_by(AttemptQuestion.position)
    )
    return list(session.scalars(stmt))


def selected_questions(
    session: Session,
    attempt_id: str,
) -> list[Question]:
    stmt = (
        select(Question)
        .join(
            AttemptQuestion,
            AttemptQuestion.question_id == Question.id,
        )
        .where(AttemptQuestion.attempt_id == attempt_id)
        .order_by(AttemptQuestion.position)
    )
    return list(session.scalars(stmt))


def bind_selected_questions(
    session: Session,
    attempt: QuizAttempt,
    questions: list[Question],
) -> None:
    existing = selected_question_ids(
        session,
        attempt.id,
    )

    if existing:
        return

    for position, question in enumerate(questions):
        session.add(
            AttemptQuestion(
                attempt_id=attempt.id,
                question_id=question.id,
                position=position,
            )
        )

    attempt.total = len(questions)
    session.flush()


def evidence_text(session: Session, question: Question) -> str | None:
    if question.evidence_chunk_id is None:
        return None
    chunk = session.get(Chunk, question.evidence_chunk_id)
    return chunk.text if chunk else None


def retrieved_evidence_text(
    session: Session, evidence_chunk_ids: list[str]
) -> str | None:
    """Return the highest-ranked retrieved passage actually used by the judge."""
    for chunk_id in evidence_chunk_ids:
        chunk = session.get(Chunk, chunk_id)
        if chunk is not None:
            return chunk.text
    return None

def kg_edge_text(question: Question) -> str | None:
    relation = question.relation
    if relation is None:
        return None
    subject = localized(relation.subject.translations, "label").get("en", relation.subject.key)
    predicate = localized(relation.predicate.translations, "label").get(
        "en", relation.predicate_id
    )
    obj = localized(relation.object.translations, "label").get("en", relation.object.key)
    return f"{subject} → ({predicate}) → {obj}"

def _record_mastery(session: Session, user_id: str, question: Question, is_correct: bool) -> None:
    if question.relation_id is None:
        return

    row = session.get(
        ConceptMastery, {"user_id": user_id, "relation_id": question.relation_id}
    )
    if row is None:
        row = ConceptMastery(
            user_id=user_id,
            relation_id=question.relation_id,
            correct_count=0,
            wrong_count=0,
        )
        session.add(row)
    if is_correct:
        row.correct_count += 1
    else:
        row.wrong_count += 1
    session.flush()

def _record(
    session: Session,
    attempt: QuizAttempt,
    question: Question,
    *,
    is_correct: bool,
    was_followup: bool,
    option_id: str | None = None,
    text_answer: str | None = None,
    feedback: str | None = None,
) -> None:
    existing = next((a for a in attempt.answers if a.question_id == question.id), None)
    if existing is None:
        attempt.answers.append(
            AttemptAnswer(
                attempt_id=attempt.id,
                question_id=question.id,
                option_id=option_id,
                text_answer=text_answer,
                feedback=feedback,
                is_correct=is_correct,
                was_followup=was_followup,
            )
        )
        if is_correct and not was_followup:
            attempt.score += 1

    _record_mastery(session, attempt.user_id, question, is_correct)
    session.flush()

def grade(
    session: Session,
    attempt: QuizAttempt,
    question: Question,
    option_id: str | None,
    was_followup: bool,
) -> tuple[bool, Option]:
    correct_option = next((o for o in question.options if o.is_correct), None)
    if correct_option is None:
        raise ValueError(f"question {question.id} has no correct option")

    is_correct = option_id is not None and option_id == correct_option.id
    _record(
        session,
        attempt,
        question,
        is_correct=is_correct,
        was_followup=was_followup,
        option_id=option_id,
    )
    return is_correct, correct_option

def grade_open(
    session: Session,
    attempt: QuizAttempt,
    question: Question,
    text_answer: str,
    was_followup: bool,
    lang: str,
) -> tuple[str, bool, str, list[str]]:
    from app.evaluation import judge_answer

    translation = next((t for t in question.translations if t.lang == lang), None) or (
        question.translations[0] if question.translations else None
    )

    verdict = judge_answer(
        session,
        story_id=question.story_id,
        question=translation.prompt if translation else "",
        learner_answer=text_answer,
        reference_answer=translation.reference_answer if translation else "",
        lang=lang,
    )

    # A judgement with insufficient approved evidence is neutral. Do not
    # record it as a wrong answer and do not affect concept mastery.
    if verdict.status != "not_enough_evidence":
        _record(
            session,
            attempt,
            question,
            is_correct=verdict.is_correct,
            was_followup=was_followup,
            text_answer=text_answer,
            feedback=verdict.feedback,
        )

    return (
        verdict.status,
        verdict.is_correct,
        verdict.feedback,
        verdict.evidence,
    )

def finish_attempt(session: Session, attempt: QuizAttempt) -> QuizAttempt:
    if attempt.finished_at is None:
        attempt.finished_at = datetime.now(timezone.utc)
        session.flush()
    return attempt

