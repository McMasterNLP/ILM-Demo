from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

def _uuid() -> str:
    return uuid.uuid4().hex

LANGS = ("en", "ar")
STATUSES = ("draft", "proposed", "approved")
ENTITY_TYPES = ("person", "place", "group")
QUESTION_TYPES = ("relationship", "identification", "comprehension")
ANSWER_MODES = ("choice", "open")
LEVELS = ("beginner", "intermediate", "advanced")

EMBEDDING_DIM = 1024

def _lang_column() -> Mapped[str]:
    return mapped_column(String(2), primary_key=True)

class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

class Source(Base, TimestampMixin):

    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    citation: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)

class ExtractionRun(Base):

    __tablename__ = "extraction_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    stage: Mapped[str] = mapped_column(String(40), nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(60))
    params: Mapped[dict | None] = mapped_column(JSONB)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

class Story(Base, TimestampMixin):
    __tablename__ = "stories"
    __table_args__ = (CheckConstraint(f"level IN {LEVELS}", name="ck_stories_level"),)

    id: Mapped[str] = mapped_column(String(60), primary_key=True)
    level: Mapped[str] = mapped_column(String(20), nullable=False)
    minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    character_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unlock_after: Mapped[int | None] = mapped_column(Integer)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    published: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    quiz_size: Mapped[int] = mapped_column(Integer, nullable=False, default=6)
    quiz_relationship: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quiz_identification: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quiz_comprehension: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    translations: Mapped[list[StoryTranslation]] = relationship(
        back_populates="story", cascade="all, delete-orphan", lazy="selectin"
    )
    paragraphs: Mapped[list[Paragraph]] = relationship(
        back_populates="story",
        cascade="all, delete-orphan",
        order_by="Paragraph.position",
        lazy="selectin",
    )

class StoryTranslation(Base):
    __tablename__ = "story_translations"
    __table_args__ = (
        CheckConstraint(f"lang IN {LANGS}", name="ck_story_tr_lang"),
        CheckConstraint(f"status IN {STATUSES}", name="ck_story_tr_status"),
    )

    story_id: Mapped[str] = mapped_column(
        ForeignKey("stories.id", ondelete="CASCADE"), primary_key=True
    )
    lang: Mapped[str] = _lang_column()
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    heading: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="draft")

    story: Mapped[Story] = relationship(back_populates="translations")

class Paragraph(Base):
    __tablename__ = "paragraphs"
    __table_args__ = (UniqueConstraint("story_id", "position", name="uq_paragraph_position"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    story_id: Mapped[str] = mapped_column(
        ForeignKey("stories.id", ondelete="CASCADE"), nullable=False, index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    is_pending: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    story: Mapped[Story] = relationship(back_populates="paragraphs")
    translations: Mapped[list[ParagraphTranslation]] = relationship(
        back_populates="paragraph", cascade="all, delete-orphan", lazy="selectin"
    )
    mentions: Mapped[list[EntityMention]] = relationship(
        back_populates="paragraph", cascade="all, delete-orphan", lazy="selectin"
    )

class ParagraphTranslation(Base):
    __tablename__ = "paragraph_translations"
    __table_args__ = (
        CheckConstraint(f"lang IN {LANGS}", name="ck_paragraph_tr_lang"),
        CheckConstraint(f"status IN {STATUSES}", name="ck_paragraph_tr_status"),
    )

    paragraph_id: Mapped[str] = mapped_column(
        ForeignKey("paragraphs.id", ondelete="CASCADE"), primary_key=True
    )
    lang: Mapped[str] = _lang_column()
    body: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="draft")
    source_id: Mapped[str | None] = mapped_column(ForeignKey("sources.id", ondelete="SET NULL"))
    approved_by: Mapped[str | None] = mapped_column(String(120))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    paragraph: Mapped[Paragraph] = relationship(back_populates="translations")

class Predicate(Base):

    __tablename__ = "predicates"

    id: Mapped[str] = mapped_column(String(60), primary_key=True)
    symmetric: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    translations: Mapped[list[PredicateTranslation]] = relationship(
        back_populates="predicate", cascade="all, delete-orphan", lazy="selectin"
    )

class PredicateTranslation(Base):
    __tablename__ = "predicate_translations"
    __table_args__ = (CheckConstraint(f"lang IN {LANGS}", name="ck_predicate_tr_lang"),)

    predicate_id: Mapped[str] = mapped_column(
        ForeignKey("predicates.id", ondelete="CASCADE"), primary_key=True
    )
    lang: Mapped[str] = _lang_column()
    label: Mapped[str] = mapped_column(Text, nullable=False)
    phrase: Mapped[str] = mapped_column(Text, nullable=False)

    predicate: Mapped[Predicate] = relationship(back_populates="translations")

class Entity(Base):

    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("key", name="uq_entity_key"),
        CheckConstraint(f"type IN {ENTITY_TYPES}", name="ck_entity_type"),
        CheckConstraint(f"status IN {STATUSES}", name="ck_entity_status"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    key: Mapped[str] = mapped_column(String(60), nullable=False)
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="approved")
    run_id: Mapped[str | None] = mapped_column(ForeignKey("extraction_runs.id", ondelete="SET NULL"))

    translations: Mapped[list[EntityTranslation]] = relationship(
        back_populates="entity", cascade="all, delete-orphan", lazy="selectin"
    )
    appearances: Mapped[list[StoryEntity]] = relationship(
        back_populates="entity", cascade="all, delete-orphan", lazy="selectin"
    )

class StoryEntity(Base):

    __tablename__ = "story_entities"

    story_id: Mapped[str] = mapped_column(
        ForeignKey("stories.id", ondelete="CASCADE"), primary_key=True
    )
    entity_id: Mapped[str] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    )
    layout_x: Mapped[float | None] = mapped_column(Float)
    layout_y: Mapped[float | None] = mapped_column(Float)

    entity: Mapped[Entity] = relationship(back_populates="appearances")

class EntityTranslation(Base):
    __tablename__ = "entity_translations"
    __table_args__ = (CheckConstraint(f"lang IN {LANGS}", name="ck_entity_tr_lang"),)

    entity_id: Mapped[str] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    )
    lang: Mapped[str] = _lang_column()
    label: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    entity: Mapped[Entity] = relationship(back_populates="translations")

class EntityMention(Base):

    __tablename__ = "entity_mentions"
    __table_args__ = (Index("ix_mention_paragraph_entity", "paragraph_id", "entity_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    entity_id: Mapped[str] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    paragraph_id: Mapped[str] = mapped_column(
        ForeignKey("paragraphs.id", ondelete="CASCADE"), nullable=False
    )
    lang: Mapped[str] = mapped_column(String(2), nullable=False, default="en")
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    score: Mapped[float | None] = mapped_column(Float)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("extraction_runs.id", ondelete="SET NULL"))

    paragraph: Mapped[Paragraph] = relationship(back_populates="mentions")
    entity: Mapped[Entity] = relationship()

class Relation(Base):

    __tablename__ = "relations"
    __table_args__ = (
        UniqueConstraint("subject_id", "predicate_id", "object_id", name="uq_relation_triple"),
        CheckConstraint(f"status IN {STATUSES}", name="ck_relation_status"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    subject_id: Mapped[str] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    predicate_id: Mapped[str] = mapped_column(
        ForeignKey("predicates.id", ondelete="RESTRICT"), nullable=False
    )
    object_id: Mapped[str] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="approved")
    run_id: Mapped[str | None] = mapped_column(ForeignKey("extraction_runs.id", ondelete="SET NULL"))

    subject: Mapped[Entity] = relationship(foreign_keys=[subject_id], lazy="selectin")
    object: Mapped[Entity] = relationship(foreign_keys=[object_id], lazy="selectin")
    predicate: Mapped[Predicate] = relationship(lazy="selectin")
    appearances: Mapped[list[StoryRelation]] = relationship(
        back_populates="relation", cascade="all, delete-orphan", lazy="selectin"
    )

class StoryRelation(Base):

    __tablename__ = "story_relations"
    __table_args__ = (Index("ix_story_relation_story", "story_id"),)

    story_id: Mapped[str] = mapped_column(
        ForeignKey("stories.id", ondelete="CASCADE"), primary_key=True
    )
    relation_id: Mapped[str] = mapped_column(
        ForeignKey("relations.id", ondelete="CASCADE"), primary_key=True
    )
    evidence_paragraph_id: Mapped[str | None] = mapped_column(
        ForeignKey("paragraphs.id", ondelete="SET NULL")
    )

    # Story-specific extraction provenance.
    #
    # These belong here rather than on Relation because the same
    # canonical KG edge may be supported by different evidence in
    # different stories.
    evidence_source: Mapped[str | None] = mapped_column(String(60))
    source_relation: Mapped[str | None] = mapped_column(String(100))
    resolution_rule: Mapped[str | None] = mapped_column(String(120))

    relation: Mapped[Relation] = relationship(back_populates="appearances")


class StoryRelationEvidence(Base):
    """
    Paragraph-level evidence supporting a relation appearance in a story.

    A StoryRelation can have multiple supporting paragraphs. Keeping evidence
    in a separate table allows one paragraph to be reprocessed or rejected
    without removing a relation that is still supported elsewhere.
    """

    __tablename__ = "story_relation_evidence"
    __table_args__ = (
        Index(
            "ix_story_relation_evidence_paragraph",
            "paragraph_id",
        ),
    )

    story_id: Mapped[str] = mapped_column(
        ForeignKey("stories.id", ondelete="CASCADE"),
        primary_key=True,
    )
    relation_id: Mapped[str] = mapped_column(
        ForeignKey("relations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    paragraph_id: Mapped[str] = mapped_column(
        ForeignKey("paragraphs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    lang: Mapped[str] = mapped_column(
        String(2),
        primary_key=True,
        default="ar",
    )

    evidence_source: Mapped[str | None] = mapped_column(String(60))
    source_relation: Mapped[str | None] = mapped_column(String(100))
    resolution_rule: Mapped[str | None] = mapped_column(String(120))

class Question(Base, TimestampMixin):
    __tablename__ = "questions"
    __table_args__ = (
        CheckConstraint(f"type IN {QUESTION_TYPES}", name="ck_question_type"),
        CheckConstraint(f"answer_mode IN {ANSWER_MODES}", name="ck_question_answer_mode"),
        CheckConstraint(f"status IN {STATUSES}", name="ck_question_status"),
        Index("ix_question_story", "story_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    story_id: Mapped[str] = mapped_column(
        ForeignKey("stories.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = mapped_column(String(30), nullable=False)
    answer_mode: Mapped[str] = mapped_column(String(20), nullable=False, default="choice")
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    relation_id: Mapped[str | None] = mapped_column(ForeignKey("relations.id", ondelete="SET NULL"))
    evidence_paragraph_id: Mapped[str | None] = mapped_column(
        ForeignKey("paragraphs.id", ondelete="SET NULL")
    )
    evidence_chunk_id: Mapped[str | None] = mapped_column(
        ForeignKey("chunks.id", ondelete="SET NULL")
    )
    followup_of: Mapped[str | None] = mapped_column(ForeignKey("questions.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="approved")
    run_id: Mapped[str | None] = mapped_column(ForeignKey("extraction_runs.id", ondelete="SET NULL"))

    translations: Mapped[list[QuestionTranslation]] = relationship(
        back_populates="question", cascade="all, delete-orphan", lazy="selectin"
    )
    options: Mapped[list[Option]] = relationship(
        back_populates="question",
        cascade="all, delete-orphan",
        order_by="Option.position",
        lazy="selectin",
    )
    followup: Mapped[Question | None] = relationship(
        remote_side=[id], back_populates="parent", uselist=False
    )
    parent: Mapped[Question | None] = relationship(back_populates="followup", remote_side=[followup_of])
    relation: Mapped[Relation | None] = relationship(lazy="selectin")

class QuestionTranslation(Base):
    __tablename__ = "question_translations"
    __table_args__ = (CheckConstraint(f"lang IN {LANGS}", name="ck_question_tr_lang"),)

    question_id: Mapped[str] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), primary_key=True
    )
    lang: Mapped[str] = _lang_column()
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reference_answer: Mapped[str] = mapped_column(Text, nullable=False, default="")

    question: Mapped[Question] = relationship(back_populates="translations")

class Option(Base):
    __tablename__ = "options"
    __table_args__ = (UniqueConstraint("question_id", "position", name="uq_option_position"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    question_id: Mapped[str] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    question: Mapped[Question] = relationship(back_populates="options")
    translations: Mapped[list[OptionTranslation]] = relationship(
        back_populates="option", cascade="all, delete-orphan", lazy="selectin"
    )

class OptionTranslation(Base):
    __tablename__ = "option_translations"
    __table_args__ = (CheckConstraint(f"lang IN {LANGS}", name="ck_option_tr_lang"),)

    option_id: Mapped[str] = mapped_column(
        ForeignKey("options.id", ondelete="CASCADE"), primary_key=True
    )
    lang: Mapped[str] = _lang_column()
    label: Mapped[str] = mapped_column(Text, nullable=False)

    option: Mapped[Option] = relationship(back_populates="translations")

class User(Base, TimestampMixin):

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String(120))

class ReadingProgress(Base):
    __tablename__ = "reading_progress"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    story_id: Mapped[str] = mapped_column(
        ForeignKey("stories.id", ondelete="CASCADE"), primary_key=True
    )
    furthest_paragraph: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"
    __table_args__ = (Index("ix_attempt_user_story", "user_id", "story_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    story_id: Mapped[str] = mapped_column(
        ForeignKey("stories.id", ondelete="CASCADE"), nullable=False
    )
    score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    answers: Mapped[list[AttemptAnswer]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan", lazy="selectin"
    )

    selected_questions: Mapped[list["AttemptQuestion"]] = relationship(
        back_populates="attempt",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="AttemptQuestion.position",
    )

class AttemptQuestion(Base):
    __tablename__ = "attempt_questions"
    __table_args__ = (
        UniqueConstraint(
            "attempt_id",
            "position",
            name="uq_attempt_question_position",
        ),
        Index(
            "ix_attempt_questions_attempt_position",
            "attempt_id",
            "position",
        ),
    )

    attempt_id: Mapped[str] = mapped_column(
        ForeignKey("quiz_attempts.id", ondelete="CASCADE"),
        primary_key=True,
    )
    question_id: Mapped[str] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    attempt: Mapped["QuizAttempt"] = relationship(
        back_populates="selected_questions"
    )
    question: Mapped["Question"] = relationship()



class AttemptAnswer(Base):
    __tablename__ = "attempt_answers"
    __table_args__ = (
        UniqueConstraint("attempt_id", "question_id", name="uq_attempt_question"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    attempt_id: Mapped[str] = mapped_column(
        ForeignKey("quiz_attempts.id", ondelete="CASCADE"), nullable=False
    )
    question_id: Mapped[str] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), nullable=False
    )
    option_id: Mapped[str | None] = mapped_column(ForeignKey("options.id", ondelete="SET NULL"))
    text_answer: Mapped[str | None] = mapped_column(Text)
    feedback: Mapped[str | None] = mapped_column(Text)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    was_followup: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    answered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    attempt: Mapped[QuizAttempt] = relationship(back_populates="answers")

class ConceptMastery(Base):

    __tablename__ = "concept_mastery"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    relation_id: Mapped[str] = mapped_column(
        ForeignKey("relations.id", ondelete="CASCADE"), primary_key=True
    )
    correct_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    wrong_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

class Document(Base, TimestampMixin):

    __tablename__ = "documents"
    __table_args__ = (CheckConstraint(f"status IN {STATUSES}", name="ck_document_status"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    source_id: Mapped[str | None] = mapped_column(ForeignKey("sources.id", ondelete="SET NULL"))
    story_id: Mapped[str | None] = mapped_column(ForeignKey("stories.id", ondelete="SET NULL"))
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    media_type: Mapped[str | None] = mapped_column(String(120))
    stored_path: Mapped[str] = mapped_column(Text, nullable=False)
    lang: Mapped[str] = mapped_column(String(2), nullable=False, default="en")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="draft")

class Chunk(Base):

    __tablename__ = "chunks"
    __table_args__ = (
        CheckConstraint(
            "paragraph_id IS NOT NULL OR document_id IS NOT NULL",
            name="ck_chunk_has_origin",
        ),
        Index("ix_chunk_story_lang", "story_id", "lang"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    story_id: Mapped[str | None] = mapped_column(ForeignKey("stories.id", ondelete="CASCADE"))
    paragraph_id: Mapped[str | None] = mapped_column(
        ForeignKey("paragraphs.id", ondelete="CASCADE")
    )
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    lang: Mapped[str] = mapped_column(String(2), nullable=False, default="en")
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("extraction_runs.id", ondelete="SET NULL"))

    embeddings: Mapped[list[ChunkEmbedding]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )

class ChunkEmbedding(Base):

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        Index(
            "ix_chunk_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(120), primary_key=True)
    dim: Mapped[int] = mapped_column(Integer, nullable=False, default=EMBEDDING_DIM)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("extraction_runs.id", ondelete="SET NULL"))

    chunk: Mapped[Chunk] = relationship(back_populates="embeddings")

class QuestionEmbedding(Base):

    __tablename__ = "question_embeddings"
    __table_args__ = (
        Index(
            "ix_question_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    question_id: Mapped[str] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(120), primary_key=True)
    lang: Mapped[str] = mapped_column(String(2), nullable=False, default="en")
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
