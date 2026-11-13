from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import User

SessionDep = Annotated[Session, Depends(get_session)]

def current_user(
    session: SessionDep,
    x_learner_id: Annotated[str | None, Header()] = None,
) -> User:
    learner_id = (x_learner_id or "").strip()[:64] or "anonymous"

    user = session.get(User, learner_id)
    if user is None:
        user = User(id=learner_id)
        session.add(user)
        session.flush()
    return user

UserDep = Annotated[User, Depends(current_user)]

def require_admin(request: Request) -> str:
    user = request.session.get("admin_user") if hasattr(request, "session") else None
    if not user:
        raise HTTPException(401, "Sign in at /admin first")
    return str(user)

AdminDep = Annotated[str, Depends(require_admin)]
