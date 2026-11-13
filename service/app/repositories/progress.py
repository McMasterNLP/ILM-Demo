from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import AttemptAnswer, Paragraph, QuizAttempt, ReadingProgress, Story
from app.schemas import QuizResultOut, StoryProgressOut

def progress_rows(session: Session, user_id: str) -> dict[str, ReadingProgress]:
    stmt = select(ReadingProgress).where(ReadingProgress.user_id == user_id)
    return {row.story_id: row for row in session.scalars(stmt)}

def latest_results(session: Session, user_id: str) -> dict[str, QuizResultOut]:
    newest = (
        select(
            QuizAttempt.story_id,
            func.max(QuizAttempt.finished_at).label("finished_at"),
        )
        .where(QuizAttempt.user_id == user_id, QuizAttempt.finished_at.is_not(None))
        .group_by(QuizAttempt.story_id)
        .subquery()
    )
    stmt = select(QuizAttempt).join(
        newest,
        (QuizAttempt.story_id == newest.c.story_id)
        & (QuizAttempt.finished_at == newest.c.finished_at),
    )
    return {
        attempt.story_id: QuizResultOut(correct=attempt.score, total=attempt.total)
        for attempt in session.scalars(stmt)
        if attempt.user_id == user_id
    }

def questions_answered_count(session: Session, user_id: str) -> int:
    stmt = (
        select(func.count(AttemptAnswer.id))
        .join(QuizAttempt, QuizAttempt.id == AttemptAnswer.attempt_id)
        .where(QuizAttempt.user_id == user_id)
    )
    return session.scalar(stmt) or 0

def unfinished_story_ids(session: Session, user_id: str) -> set[str]:
    latest = (
        select(
            QuizAttempt.story_id,
            func.max(QuizAttempt.started_at).label("started_at"),
        )
        .where(QuizAttempt.user_id == user_id)
        .group_by(QuizAttempt.story_id)
        .subquery()
    )
    stmt = (
        select(QuizAttempt.story_id)
        .join(
            latest,
            (QuizAttempt.story_id == latest.c.story_id)
            & (QuizAttempt.started_at == latest.c.started_at),
        )
        .where(QuizAttempt.user_id == user_id, QuizAttempt.finished_at.is_(None))
    )
    return set(session.scalars(stmt))

def story_progress(
    rows: dict[str, ReadingProgress],
    results: dict[str, QuizResultOut],
    story_id: str,
    in_progress: set[str] = frozenset(),
) -> StoryProgressOut:
    row = rows.get(story_id)
    return StoryProgressOut(
        paragraph=row.furthest_paragraph if row else 0,
        completed=bool(row and row.completed_at),
        quiz=results.get(story_id),
        quiz_in_progress=story_id in in_progress,
    )

def paragraph_total(session: Session, story_id: str) -> int:
    stmt = select(func.count(Paragraph.id)).where(Paragraph.story_id == story_id)
    return session.scalar(stmt) or 0

def record_paragraph(
    session: Session, user_id: str, story: Story, paragraph: int
) -> ReadingProgress:
    total = paragraph_total(session, story.id)
    index = max(0, min(paragraph, max(total - 1, 0)))

    row = session.get(ReadingProgress, {"user_id": user_id, "story_id": story.id})
    if row is None:
        row = ReadingProgress(user_id=user_id, story_id=story.id, furthest_paragraph=index)
        session.add(row)
    else:
        row.furthest_paragraph = max(row.furthest_paragraph, index)

    if total > 0 and row.furthest_paragraph >= total - 1 and row.completed_at is None:
        row.completed_at = datetime.now(timezone.utc)

    session.flush()
    return row

def restart_story(session: Session, user_id: str, story_id: str) -> ReadingProgress | None:
    row = session.get(ReadingProgress, {"user_id": user_id, "story_id": story_id})
    if row is None:
        return None
    row.furthest_paragraph = 0
    row.completed_at = None
    session.flush()
    return row

def has_completed(session: Session, user_id: str, story_id: str) -> bool:
    row = session.get(ReadingProgress, {"user_id": user_id, "story_id": story_id})
    return bool(row and row.completed_at)

def completed_count(session: Session, user_id: str) -> int:
    stmt = select(func.count(ReadingProgress.story_id)).where(
        ReadingProgress.user_id == user_id, ReadingProgress.completed_at.is_not(None)
    )
    return session.scalar(stmt) or 0
