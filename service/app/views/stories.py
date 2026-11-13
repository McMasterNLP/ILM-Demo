from fastapi import APIRouter, HTTPException

from app.deps import SessionDep, UserDep
from app.repositories import content as content_repo
from app.repositories import progress as progress_repo
from app.schemas import StoryDetailOut, StoryGraphOut, StoryOut

router = APIRouter()

def _story_out(story, paragraphs: int, questions: int, progress) -> StoryOut:
    return StoryOut(
        id=story.id,
        level=story.level,
        minutes=story.minutes,
        character_count=story.character_count,
        unlock_after=story.unlock_after,
        paragraph_count=paragraphs,
        question_count=questions,
        title=content_repo.localized(story.translations, "title", only_approved=True),
        summary=content_repo.localized(story.translations, "summary", only_approved=True),
        progress=progress,
    )

@router.get("", response_model=list[StoryOut])
def list_stories(session: SessionDep, user: UserDep):
    stories = content_repo.list_stories(session)
    paragraphs = content_repo.paragraph_counts(session)
    questions = content_repo.question_counts(session)
    rows = progress_repo.progress_rows(session, user.id)
    results = progress_repo.latest_results(session, user.id)
    in_progress = progress_repo.unfinished_story_ids(session, user.id)

    return [
        _story_out(
            story,
            paragraphs.get(story.id, 0),
            questions.get(story.id, 0),
            progress_repo.story_progress(rows, results, story.id, in_progress),
        )
        for story in stories
    ]

@router.get("/{story_id}", response_model=StoryDetailOut)
def retrieve_story(story_id: str, session: SessionDep, user: UserDep):
    story = content_repo.get_story(session, story_id)
    if story is None:
        raise HTTPException(404, "Story not found")

    rows = progress_repo.progress_rows(session, user.id)
    results = progress_repo.latest_results(session, user.id)
    in_progress = progress_repo.unfinished_story_ids(session, user.id)
    paragraphs = content_repo.story_paragraphs(session, story)
    questions = content_repo.question_counts(session)

    base = _story_out(
        story,
        len(paragraphs),
        questions.get(story.id, 0),
        progress_repo.story_progress(rows, results, story.id, in_progress),
    )
    return StoryDetailOut(
        **base.model_dump(),
        heading=content_repo.story_heading(story),
        paragraphs=paragraphs,
    )

@router.get("/{story_id}/graph", response_model=StoryGraphOut)
def retrieve_graph(story_id: str, session: SessionDep):
    story = content_repo.get_story(session, story_id)
    if story is None:
        raise HTTPException(404, "Story not found")
    return content_repo.story_graph(session, story_id)
