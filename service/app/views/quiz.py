from fastapi import APIRouter, HTTPException

from app.deps import SessionDep, UserDep
from app.questions import GenerationUnavailable
from app.models import Question, QuizAttempt
from app.repositories import content as content_repo
from app.repositories import progress as progress_repo
from app.repositories import composition as composition_repo
from app.repositories import quiz as quiz_repo
from app.schemas import AnswerIn, AnswerOut, AttemptOut, AttemptStartIn, QuizOut

router = APIRouter()

attempts_router = APIRouter()

def _require_unlocked(session, user_id: str, story) -> None:
    if story.unlock_after is not None:
        if progress_repo.completed_count(session, user_id) < story.unlock_after:
            raise HTTPException(403, "Story is locked")
    if not progress_repo.has_completed(session, user_id, story.id):
        raise HTTPException(403, "Finish reading the story first")

@router.get("/{story_id}", response_model=QuizOut)
def retrieve_quiz(
    story_id: str,
    session: SessionDep,
    user: UserDep,
    size: int | None = None,
    relationship: int | None = None,
    identification: int | None = None,
    comprehension: int | None = None,
    lang: str = "en",
):
    story = content_repo.get_story(session, story_id)
    if story is None:
        raise HTTPException(404, "Story not found")
    _require_unlocked(session, user.id, story)

    existing = quiz_repo.find_unfinished_attempt(
        session,
        user.id,
        story_id,
    )

    if existing is not None:
        existing_questions = quiz_repo.selected_questions(
            session,
            existing.id,
        )

        if existing_questions:
            return QuizOut(
                story_id=story_id,
                questions=[
                    quiz_repo.to_question_out(session, q)
                    for q in existing_questions
                ],
            )

    configured_size, configured_mix = composition_repo.shape_for(session, story_id)
    overrides = {
        "relationship": relationship,
        "identification": identification,
        "comprehension": comprehension,
    }
    mix = {k: v for k, v in overrides.items() if v} or configured_mix

    questions = composition_repo.compose_quiz(
        session,
        story_id,
        size=max(1, size or configured_size),
        mix=mix or None,
        lang=lang,
    )
    if not questions:
        raise HTTPException(404, f"No questions for this story in '{lang}'")

    return QuizOut(
        story_id=story_id,
        questions=[quiz_repo.to_question_out(session, q) for q in questions],
    )

@router.post("/{story_id}/attempts", response_model=AttemptOut)
def create_attempt(
    story_id: str,
    payload: AttemptStartIn,
    session: SessionDep,
    user: UserDep,
    lang: str = "en",
):
    story = content_repo.get_story(session, story_id)
    if story is None:
        raise HTTPException(404, "Story not found")
    _require_unlocked(session, user.id, story)

    question_ids = payload.question_ids

    if not question_ids:
        raise HTTPException(
            400,
            "At least one question is required",
        )

    if len(question_ids) != len(set(question_ids)):
        raise HTTPException(
            400,
            "Duplicate question IDs are not allowed",
        )

    eligible_questions = {
        question.id: question
        for question in quiz_repo.story_questions(
            session,
            story_id,
        )
    }

    try:
        questions = [
            eligible_questions[question_id]
            for question_id in question_ids
        ]
    except KeyError:
        raise HTTPException(
            400,
            "Quiz contains an invalid question",
        )

    existing = quiz_repo.find_unfinished_attempt(
        session,
        user.id,
        story_id,
    )

    if existing is not None:
        stored_ids = quiz_repo.selected_question_ids(
            session,
            existing.id,
        )

        if stored_ids:
            if stored_ids != question_ids:
                raise HTTPException(
                    409,
                    "An unfinished attempt already has a different question set",
                )
        else:
            quiz_repo.bind_selected_questions(
                session,
                existing,
                questions,
            )

        return AttemptOut(
            id=existing.id,
            story_id=story_id,
            score=existing.score,
            total=existing.total,
            answered_question_ids=quiz_repo.answered_question_ids(existing),
        )

    attempt = quiz_repo.start_attempt(
        session,
        user.id,
        story_id,
        len(questions),
    )

    quiz_repo.bind_selected_questions(
        session,
        attempt,
        questions,
    )

    return AttemptOut(
        id=attempt.id,
        story_id=story_id,
        score=0,
        total=attempt.total,
    )

@attempts_router.post("/{attempt_id}/answers", response_model=AnswerOut)
def create_answer(attempt_id: str, payload: AnswerIn, session: SessionDep, user: UserDep):
    attempt = session.get(QuizAttempt, attempt_id)
    if attempt is None or attempt.user_id != user.id:
        raise HTTPException(404, "Attempt not found")
    if attempt.finished_at is not None:
        raise HTTPException(409, "Attempt already finished")

    question = session.get(Question, payload.question_id)
    if question is None or question.story_id != attempt.story_id:
        raise HTTPException(404, "Question not found")

    selected_ids = set(
        quiz_repo.selected_question_ids(
            session,
            attempt.id,
        )
    )

    if selected_ids:
        if payload.was_followup:
            allowed = question.followup_of in selected_ids
        else:
            allowed = question.id in selected_ids

        if not allowed:
            raise HTTPException(
                409,
                "Question is not part of this attempt",
            )

    correct_option_id = ""
    feedback = ""
    reference = ""
    grading_status = ""
    source_text = None

    if question.answer_mode == "open":
        if not (payload.text or "").strip():
            raise HTTPException(400, "Write an answer first")
        try:
            grading_status, is_correct, feedback, _evidence = quiz_repo.grade_open(
                session,
                attempt,
                question,
                payload.text or "",
                payload.was_followup,
                payload.lang,
            )
            source_text = quiz_repo.retrieved_evidence_text(session, _evidence)
        except GenerationUnavailable:
            # AI availability must never turn a learner answer into a wrong
            # answer. Nothing is persisted and no mastery score is changed.
            grading_status = "grading_unavailable"
            is_correct = False
            feedback = (
                "Ø§Ù„ØªÙ‚ÙŠÙŠÙ… Ø§Ù„Ø¯Ù„Ø§Ù„ÙŠ ØºÙŠØ± Ù…ØªØ§Ø­ Ù…Ø¤Ù‚ØªÙ‹Ø§. ÙŠÙ…ÙƒÙ†Ùƒ Ø§Ù„Ù…Ø­Ø§ÙˆÙ„Ø© Ù…Ø±Ø© Ø£Ø®Ø±Ù‰ Ù„Ø§Ø­Ù‚Ù‹Ø§ Ø£Ùˆ ØªØ®Ø·ÙŠ Ù‡Ø°Ø§ Ø§Ù„Ø³Ø¤Ø§Ù„."
                if payload.lang == "ar"
                else "Semantic grading is temporarily unavailable. "
                     "Please try again later or skip this question."
            )
            source_text = quiz_repo.evidence_text(session, question)
        except Exception as exc:
            raise HTTPException(503, f"Could not grade that answer: {exc}") from exc

        translation = next(
            (t for t in question.translations if t.lang == payload.lang), None
        ) or (question.translations[0] if question.translations else None)
        reference = translation.reference_answer if translation else ""
    else:
        if payload.option_id is not None and not any(
            o.id == payload.option_id for o in question.options
        ):
            raise HTTPException(400, "Option does not belong to that question")

        is_correct, correct_option = quiz_repo.grade(
            session, attempt, question, payload.option_id, payload.was_followup
        )
        grading_status = "correct" if is_correct else "incorrect"
        correct_option_id = correct_option.id

    follow_up = None
    if grading_status == "incorrect" and not payload.was_followup:
        next_question = quiz_repo.followup_for(session, question.id)
        if next_question is not None:
            follow_up = quiz_repo.to_question_out(session, next_question)

    if follow_up is None and len(quiz_repo.answered_question_ids(attempt)) >= attempt.total:
        quiz_repo.finish_attempt(session, attempt)

    return AnswerOut(
        is_correct=is_correct,
        grading_status=grading_status,
        correct_option_id=correct_option_id,
        feedback=feedback,
        reference_answer=reference,
        explanation=content_repo.localized(question.translations, "explanation"),
        kg_edge=quiz_repo.kg_edge_text(question),
        source_text=(
            source_text
            if question.answer_mode == "open"
            else quiz_repo.evidence_text(session, question)
        ),
        follow_up=follow_up,
        score=attempt.score,
        answered=len(attempt.answers),
        total=attempt.total,
    )

@attempts_router.post("/{attempt_id}/finish", response_model=AttemptOut)
def finish_attempt(attempt_id: str, session: SessionDep, user: UserDep):
    attempt = session.get(QuizAttempt, attempt_id)
    if attempt is None or attempt.user_id != user.id:
        raise HTTPException(404, "Attempt not found")

    quiz_repo.finish_attempt(session, attempt)
    return AttemptOut(
        id=attempt.id,
        story_id=attempt.story_id,
        score=attempt.score,
        total=attempt.total,
        finished=True,
    )

