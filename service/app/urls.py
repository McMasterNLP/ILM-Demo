from fastapi import APIRouter, Depends

from app.deps import require_admin
from app.views import authoring, entities, progress, quiz, search, stories

api_router = APIRouter()

api_router.include_router(stories.router, prefix="/stories", tags=["stories"])
api_router.include_router(entities.router, prefix="/entities", tags=["stories"])
api_router.include_router(quiz.router, prefix="/quiz", tags=["quiz"])
api_router.include_router(quiz.attempts_router, prefix="/attempts", tags=["quiz"])
api_router.include_router(progress.router, prefix="/me", tags=["learner"])
api_router.include_router(search.router, prefix="/search", tags=["retrieval"])
api_router.include_router(
    authoring.router,
    prefix="/authoring",
    tags=["authoring"],
    dependencies=[Depends(require_admin)],
)
