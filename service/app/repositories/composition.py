from __future__ import annotations

import random

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Question, Story

APPROVED = "approved"

TYPE_ORDER = {"relationship": 0, "identification": 1, "comprehension": 2}

DEFAULT_SIZE = 6

def _answerable(question: Question, lang: str) -> bool:
    prompt = next(
        (t.prompt for t in question.translations if t.lang == lang and t.prompt.strip()),
        None,
    )
    if not prompt:
        return False

    if question.answer_mode == "open":
        return True

    if not question.options:
        return False
    return all(
        any(t.lang == lang and t.label.strip() for t in option.translations)
        for option in question.options
    )

def _pool(session: Session, story_id: str, lang: str) -> list[Question]:
    stmt = (
        select(Question)
        .where(
            Question.story_id == story_id,
            Question.status == APPROVED,
            Question.followup_of.is_(None),
        )
        .order_by(Question.position)
    )
    return [q for q in session.scalars(stmt) if _answerable(q, lang)]

def _take_spread(
    candidates: list[Question], count: int, used_chunks: set[str]
) -> list[Question]:
    chosen: list[Question] = []

    for question in candidates:
        if len(chosen) >= count:
            break
        chunk = question.evidence_chunk_id
        if chunk is None or chunk not in used_chunks:
            chosen.append(question)
            if chunk:
                used_chunks.add(chunk)

    if len(chosen) < count:
        for question in candidates:
            if len(chosen) >= count:
                break
            if question not in chosen:
                chosen.append(question)

    return chosen

def compose_quiz(
    session: Session,
    story_id: str,
    *,
    size: int = DEFAULT_SIZE,
    mix: dict[str, int] | None = None,
    lang: str = "en",
) -> list[Question]:
    pool = _pool(session, story_id, lang)
    if not pool:
        return []

    used_chunks: set[str] = set()
    chosen: list[Question] = []

    rng = random.SystemRandom()
    effective_mix = dict(mix or {})

    if not effective_mix:
        available_types = [
            question_type
            for question_type, order in sorted(
                TYPE_ORDER.items(),
                key=lambda item: item[1],
            )
            if any(q.type == question_type for q in pool)
        ]

        if available_types:
            base = size // len(available_types)
            extra = size % len(available_types)

            effective_mix = {
                question_type: base + (1 if index < extra else 0)
                for index, question_type in enumerate(available_types)
            }

    for question_type, wanted in effective_mix.items():
        if wanted <= 0:
            continue

        candidates = [
            q
            for q in pool
            if q.type == question_type and q not in chosen
        ]

        rng.shuffle(candidates)

        chosen.extend(
            _take_spread(
                candidates,
                wanted,
                used_chunks,
            )
        )

    if len(chosen) < size:
        remaining = [
            q for q in pool
            if q not in chosen
        ]

        rng.shuffle(remaining)

        chosen.extend(
            _take_spread(
                remaining,
                size - len(chosen),
                used_chunks,
            )
        )

    chosen = chosen[:size]

    chosen.sort(
        key=lambda q: (
            TYPE_ORDER.get(q.type, 99),
            q.position,
        )
    )

    return chosen

def coverage(session: Session, story_id: str) -> tuple[int, int]:
    from app.models import Chunk

    chunk_ids = set(
        session.scalars(select(Chunk.id).where(Chunk.story_id == story_id))
    )
    if not chunk_ids:
        return 0, 0

    covered = {
        q.evidence_chunk_id
        for q in _pool(session, story_id, "en") + _pool(session, story_id, "ar")
        if q.evidence_chunk_id in chunk_ids
    }
    return len(covered), len(chunk_ids)

def shape_for(session: Session, story_id: str) -> tuple[int, dict[str, int]]:
    story = session.get(Story, story_id)
    if story is None:
        return DEFAULT_SIZE, {}

    mix = {
        "relationship": story.quiz_relationship,
        "identification": story.quiz_identification,
        "comprehension": story.quiz_comprehension,
    }
    return story.quiz_size or DEFAULT_SIZE, {k: v for k, v in mix.items() if v > 0}
