from fastapi import APIRouter, HTTPException

from app.deps import SessionDep, UserDep
from app.repositories import content as content_repo
from app.repositories import progress as progress_repo
from app.schemas import LearnerSummaryOut, ProgressIn, ProgressOut

router = APIRouter()

def _out(story_id: str, row, results) -> ProgressOut:
    return ProgressOut(
        story_id=story_id,
        paragraph=row.furthest_paragraph if row else 0,
        completed=bool(row and row.completed_at),
        quiz=results.get(story_id),
    )

@router.get("/progress", response_model=LearnerSummaryOut)
def retrieve_progress(session: SessionDep, user: UserDep):
    rows = progress_repo.progress_rows(session, user.id)
    results = progress_repo.latest_results(session, user.id)

    return LearnerSummaryOut(
        user_id=user.id,
        stories_started=len(rows),
        stories_completed=sum(1 for row in rows.values() if row.completed_at),
        questions_answered=progress_repo.questions_answered_count(session, user.id),
        progress=[_out(story_id, row, results) for story_id, row in sorted(rows.items())],
    )

@router.put("/progress/{story_id}", response_model=ProgressOut)
def update_progress(story_id: str, payload: ProgressIn, session: SessionDep, user: UserDep):
    story = content_repo.get_story(session, story_id)
    if story is None:
        raise HTTPException(404, "Story not found")

    row = progress_repo.record_paragraph(session, user.id, story, payload.paragraph)
    results = progress_repo.latest_results(session, user.id)
    return _out(story_id, row, results)

@router.delete("/progress/{story_id}", response_model=ProgressOut)
def destroy_progress(story_id: str, session: SessionDep, user: UserDep):
    story = content_repo.get_story(session, story_id)
    if story is None:
        raise HTTPException(404, "Story not found")

    row = progress_repo.restart_story(session, user.id, story_id)
    results = progress_repo.latest_results(session, user.id)
    return _out(story_id, row, results)
