from __future__ import annotations

import secrets
from datetime import datetime, timezone

from fastapi import FastAPI
from sqladmin import Admin, BaseView, ModelView, action, expose
from sqladmin.filters import AllUniqueStringValuesFilter, BooleanFilter
from sqladmin.authentication import AuthenticationBackend
from sqlalchemy import Engine, update
from starlette.requests import Request
from starlette.responses import RedirectResponse

from app.config import settings
from app.models import (
    AttemptAnswer,
    Chunk,
    ChunkEmbedding,
    ConceptMastery,
    Document,
    Entity,
    EntityMention,
    EntityTranslation,
    ExtractionRun,
    Option,
    OptionTranslation,
    Paragraph,
    ParagraphTranslation,
    Predicate,
    PredicateTranslation,
    Question,
    QuestionTranslation,
    QuizAttempt,
    ReadingProgress,
    Relation,
    Source,
    Story,
    StoryEntity,
    StoryRelation,
    StoryTranslation,
    User,
)

class AdminAuth(AuthenticationBackend):

    async def login(self, request: Request) -> bool:
        form = await request.form()
        username = str(form.get("username", ""))
        password = str(form.get("password", ""))

        if not settings.admin_password:
            return False
        if not (
            secrets.compare_digest(username, settings.admin_username)
            and secrets.compare_digest(password, settings.admin_password)
        ):
            return False

        request.session.update({"admin_user": username})
        return True

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return bool(request.session.get("admin_user"))

def reviewer_of(request: Request) -> str:
    return str(request.session.get("admin_user") or "unknown")

class AuthoringLink(BaseView):

    name = "Authoring console"
    icon = "fa-solid fa-pen-to-square"

    @expose("/authoring-console", methods=["GET"])
    async def index(self, request: Request) -> RedirectResponse:
        return RedirectResponse("/admin/authoring")

class StatusActionsMixin:

    def _set_status(self, request: Request, status: str) -> RedirectResponse:
        pks = [pk for pk in request.query_params.get("pks", "").split(",") if pk]
        if pks:
            model = self.model  # type: ignore[attr-defined]
            with self.session_maker() as session:  # type: ignore[attr-defined]
                session.execute(
                    update(model).where(model.id.in_(pks)).values(status=status)
                )
                session.commit()
        return RedirectResponse(request.url_for("admin:list", identity=self.identity))  # type: ignore[attr-defined]

    @action(name="approve", label="Approve", confirmation_message="Approve the selected rows?")
    async def approve(self, request: Request) -> RedirectResponse:
        return self._set_status(request, "approved")

    @action(name="reject", label="Send back to draft")
    async def reject(self, request: Request) -> RedirectResponse:
        return self._set_status(request, "draft")

class SourceAdmin(ModelView, model=Source):
    name_plural = "Sources"
    category = "Content"
    column_list = [Source.id, Source.title, Source.citation, Source.url]
    column_searchable_list = [Source.title]

class StoryAdmin(ModelView, model=Story):
    name_plural = "Stories"
    category = "Content"
    column_list = [
        Story.id,
        Story.level,
        Story.minutes,
        Story.character_count,
        Story.unlock_after,
        Story.published,
        Story.position,
    ]
    column_sortable_list = [Story.position, Story.level]
    column_searchable_list = [Story.id]

class StoryTranslationAdmin(StatusActionsMixin, ModelView, model=StoryTranslation):
    name_plural = "Story translations"
    category = "Content"
    pk_columns = [StoryTranslation.story_id, StoryTranslation.lang]
    column_list = [
        StoryTranslation.story_id,
        StoryTranslation.lang,
        StoryTranslation.title,
        StoryTranslation.status,
    ]
    column_filters = [AllUniqueStringValuesFilter(StoryTranslation.status), AllUniqueStringValuesFilter(StoryTranslation.lang)]

class ParagraphAdmin(ModelView, model=Paragraph):
    name_plural = "Paragraphs"
    category = "Content"
    column_list = [Paragraph.id, Paragraph.story_id, Paragraph.position, Paragraph.is_pending]
    column_sortable_list = [Paragraph.position]
    column_filters = [AllUniqueStringValuesFilter(Paragraph.story_id), BooleanFilter(Paragraph.is_pending)]

class ParagraphTranslationAdmin(ModelView, model=ParagraphTranslation):

    name_plural = "Paragraph translations"
    category = "Content"
    pk_columns = [ParagraphTranslation.paragraph_id, ParagraphTranslation.lang]
    column_list = [
        ParagraphTranslation.paragraph_id,
        ParagraphTranslation.lang,
        ParagraphTranslation.status,
        ParagraphTranslation.body,
        ParagraphTranslation.source_id,
        ParagraphTranslation.approved_by,
        ParagraphTranslation.approved_at,
    ]
    column_filters = [AllUniqueStringValuesFilter(ParagraphTranslation.status), AllUniqueStringValuesFilter(ParagraphTranslation.lang)]
    column_searchable_list = [ParagraphTranslation.body]

    def _review(self, request: Request, status: str) -> RedirectResponse:
        pks = [pk for pk in request.query_params.get("pks", "").split(",") if pk]
        reviewer = reviewer_of(request)
        now = datetime.now(timezone.utc)

        with self.session_maker() as session:
            for pk in pks:
                paragraph_id, _, lang = pk.partition(";")
                row = session.get(
                    ParagraphTranslation,
                    {"paragraph_id": paragraph_id, "lang": lang},
                )
                if row is None:
                    continue
                row.status = status
                row.approved_by = reviewer if status == "approved" else None
                row.approved_at = now if status == "approved" else None
            session.commit()

        return RedirectResponse(request.url_for("admin:list", identity=self.identity))

    @action(
        name="approve_text",
        label="Approve for learners",
        confirmation_message="This text becomes visible to children. Approve?",
    )
    async def approve_text(self, request: Request) -> RedirectResponse:
        return self._review(request, "approved")

    @action(name="reject_text", label="Send back to draft")
    async def reject_text(self, request: Request) -> RedirectResponse:
        return self._review(request, "draft")

class PredicateAdmin(ModelView, model=Predicate):
    name_plural = "Predicates"
    category = "Knowledge graph"
    column_list = [Predicate.id, Predicate.symmetric]

class PredicateTranslationAdmin(ModelView, model=PredicateTranslation):
    name_plural = "Predicate translations"
    category = "Knowledge graph"
    pk_columns = [PredicateTranslation.predicate_id, PredicateTranslation.lang]
    column_list = [
        PredicateTranslation.predicate_id,
        PredicateTranslation.lang,
        PredicateTranslation.label,
        PredicateTranslation.phrase,
    ]

class EntityAdmin(StatusActionsMixin, ModelView, model=Entity):
    name_plural = "Entities"
    category = "Knowledge graph"
    column_list = [Entity.id, Entity.key, Entity.type, Entity.status, Entity.run_id]
    column_filters = [AllUniqueStringValuesFilter(Entity.status), AllUniqueStringValuesFilter(Entity.type)]
    column_searchable_list = [Entity.key]

class StoryEntityAdmin(ModelView, model=StoryEntity):
    name_plural = "Story entities"
    category = "Knowledge graph"
    pk_columns = [StoryEntity.story_id, StoryEntity.entity_id]
    column_list = [
        StoryEntity.story_id,
        StoryEntity.entity_id,
        StoryEntity.layout_x,
        StoryEntity.layout_y,
    ]
    column_filters = [AllUniqueStringValuesFilter(StoryEntity.story_id)]

class EntityTranslationAdmin(ModelView, model=EntityTranslation):
    name_plural = "Entity translations"
    category = "Knowledge graph"
    pk_columns = [EntityTranslation.entity_id, EntityTranslation.lang]
    column_list = [EntityTranslation.entity_id, EntityTranslation.lang, EntityTranslation.label]

class RelationAdmin(StatusActionsMixin, ModelView, model=Relation):
    name_plural = "Relations"
    category = "Knowledge graph"
    column_list = [
        Relation.id,
        Relation.subject_id,
        Relation.predicate_id,
        Relation.object_id,
        Relation.confidence,
        Relation.status,
        Relation.run_id,
    ]
    column_filters = [AllUniqueStringValuesFilter(Relation.status), AllUniqueStringValuesFilter(Relation.predicate_id)]
    column_sortable_list = [Relation.confidence]

class StoryRelationAdmin(ModelView, model=StoryRelation):
    name_plural = "Story relations"
    category = "Knowledge graph"
    pk_columns = [StoryRelation.story_id, StoryRelation.relation_id]
    column_list = [
        StoryRelation.story_id,
        StoryRelation.relation_id,
        StoryRelation.evidence_paragraph_id,
    ]
    column_filters = [AllUniqueStringValuesFilter(StoryRelation.story_id)]

class EntityMentionAdmin(ModelView, model=EntityMention):
    name_plural = "Entity mentions"
    category = "Knowledge graph"
    column_list = [
        EntityMention.id,
        EntityMention.entity_id,
        EntityMention.paragraph_id,
        EntityMention.lang,
        EntityMention.score,
    ]
    column_filters = [AllUniqueStringValuesFilter(EntityMention.lang)]

class QuestionAdmin(StatusActionsMixin, ModelView, model=Question):
    name_plural = "Questions"
    category = "Quiz"
    column_list = [
        Question.id,
        Question.story_id,
        Question.type,
        Question.position,
        Question.relation_id,
        Question.followup_of,
        Question.status,
    ]
    column_filters = [AllUniqueStringValuesFilter(Question.status), AllUniqueStringValuesFilter(Question.story_id), AllUniqueStringValuesFilter(Question.type)]
    column_sortable_list = [Question.position]

class QuestionTranslationAdmin(ModelView, model=QuestionTranslation):
    name_plural = "Question translations"
    category = "Quiz"
    pk_columns = [QuestionTranslation.question_id, QuestionTranslation.lang]
    column_list = [
        QuestionTranslation.question_id,
        QuestionTranslation.lang,
        QuestionTranslation.prompt,
    ]

class OptionAdmin(ModelView, model=Option):
    name_plural = "Options"
    category = "Quiz"
    column_list = [Option.id, Option.question_id, Option.position, Option.is_correct]
    column_filters = [BooleanFilter(Option.is_correct)]

class OptionTranslationAdmin(ModelView, model=OptionTranslation):
    name_plural = "Option translations"
    category = "Quiz"
    pk_columns = [OptionTranslation.option_id, OptionTranslation.lang]
    column_list = [OptionTranslation.option_id, OptionTranslation.lang, OptionTranslation.label]

class DocumentAdmin(StatusActionsMixin, ModelView, model=Document):
    name_plural = "Documents"
    category = "Retrieval"
    column_list = [
        Document.id,
        Document.filename,
        Document.story_id,
        Document.source_id,
        Document.lang,
        Document.status,
    ]
    column_filters = [AllUniqueStringValuesFilter(Document.status), AllUniqueStringValuesFilter(Document.lang)]

class ChunkAdmin(ModelView, model=Chunk):
    name_plural = "Chunks"
    category = "Retrieval"
    column_list = [
        Chunk.id,
        Chunk.story_id,
        Chunk.paragraph_id,
        Chunk.document_id,
        Chunk.lang,
        Chunk.position,
        Chunk.token_count,
    ]
    column_filters = [AllUniqueStringValuesFilter(Chunk.story_id), AllUniqueStringValuesFilter(Chunk.lang)]

class ChunkEmbeddingAdmin(ModelView, model=ChunkEmbedding):
    name_plural = "Chunk embeddings"
    category = "Retrieval"
    pk_columns = [ChunkEmbedding.chunk_id, ChunkEmbedding.model]
    column_list = [
        ChunkEmbedding.chunk_id,
        ChunkEmbedding.model,
        ChunkEmbedding.dim,
        ChunkEmbedding.run_id,
    ]
    column_filters = [AllUniqueStringValuesFilter(ChunkEmbedding.model)]
    can_edit = False

class ExtractionRunAdmin(ModelView, model=ExtractionRun):
    name_plural = "Extraction runs"
    category = "Retrieval"
    column_list = [
        ExtractionRun.id,
        ExtractionRun.stage,
        ExtractionRun.model,
        ExtractionRun.started_at,
        ExtractionRun.finished_at,
    ]
    column_sortable_list = [ExtractionRun.started_at]

class UserAdmin(ModelView, model=User):
    name_plural = "Learners"
    category = "Learners"
    column_list = [User.id, User.display_name, User.created_at]
    can_create = False

class ReadingProgressAdmin(ModelView, model=ReadingProgress):
    name_plural = "Reading progress"
    category = "Learners"
    pk_columns = [ReadingProgress.user_id, ReadingProgress.story_id]
    column_list = [
        ReadingProgress.user_id,
        ReadingProgress.story_id,
        ReadingProgress.furthest_paragraph,
        ReadingProgress.completed_at,
    ]
    can_create = False
    can_edit = False

class QuizAttemptAdmin(ModelView, model=QuizAttempt):
    name_plural = "Quiz attempts"
    category = "Learners"
    column_list = [
        QuizAttempt.id,
        QuizAttempt.user_id,
        QuizAttempt.story_id,
        QuizAttempt.score,
        QuizAttempt.total,
        QuizAttempt.finished_at,
    ]
    can_create = False
    can_edit = False

class AttemptAnswerAdmin(ModelView, model=AttemptAnswer):
    name_plural = "Attempt answers"
    category = "Learners"
    column_list = [
        AttemptAnswer.attempt_id,
        AttemptAnswer.question_id,
        AttemptAnswer.is_correct,
        AttemptAnswer.was_followup,
    ]
    can_create = False
    can_edit = False

class ConceptMasteryAdmin(ModelView, model=ConceptMastery):
    name_plural = "Concept mastery"
    category = "Learners"
    pk_columns = [ConceptMastery.user_id, ConceptMastery.relation_id]
    column_list = [
        ConceptMastery.user_id,
        ConceptMastery.relation_id,
        ConceptMastery.correct_count,
        ConceptMastery.wrong_count,
        ConceptMastery.last_seen_at,
    ]
    can_create = False
    can_edit = False

VIEWS = [
    SourceAdmin,
    StoryAdmin,
    StoryTranslationAdmin,
    ParagraphAdmin,
    ParagraphTranslationAdmin,
    PredicateAdmin,
    PredicateTranslationAdmin,
    EntityAdmin,
    StoryEntityAdmin,
    EntityTranslationAdmin,
    RelationAdmin,
    StoryRelationAdmin,
    EntityMentionAdmin,
    QuestionAdmin,
    QuestionTranslationAdmin,
    OptionAdmin,
    OptionTranslationAdmin,
    DocumentAdmin,
    ChunkAdmin,
    ChunkEmbeddingAdmin,
    ExtractionRunAdmin,
    UserAdmin,
    ReadingProgressAdmin,
    QuizAttemptAdmin,
    AttemptAnswerAdmin,
    ConceptMasteryAdmin,
]

def mount_admin(app: FastAPI, engine: Engine) -> Admin:
    admin = Admin(
        app,
        engine,
        title="Ilm curation",
        base_url="/admin",
        authentication_backend=AdminAuth(secret_key=settings.session_secret),
    )
    admin.add_base_view(AuthoringLink)
    for view in VIEWS:
        admin.add_view(view)
    return admin
