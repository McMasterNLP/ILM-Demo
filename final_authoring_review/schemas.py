from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

Localized = dict[str, str]

class QuizResultOut(BaseModel):
    correct: int
    total: int

class StoryProgressOut(BaseModel):

    paragraph: int = 0
    completed: bool = False
    quiz: QuizResultOut | None = None
    quiz_in_progress: bool = False

class StoryOut(BaseModel):
    id: str
    level: str
    minutes: int
    character_count: int
    unlock_after: int | None = None
    paragraph_count: int
    question_count: int
    title: Localized
    summary: Localized
    progress: StoryProgressOut

class ParagraphOut(BaseModel):
    position: int
    pending: bool
    text: Localized
    characters: list[str] = []

class StoryDetailOut(StoryOut):
    heading: Localized
    paragraphs: list[ParagraphOut]

class GraphNodeOut(BaseModel):
    id: str
    type: str
    label: Localized
    category: Localized
    description: Localized
    x: float
    y: float

class GraphEdgeOut(BaseModel):
    from_: str = Field(alias="from")
    to: str
    label: Localized
    phrase: Localized

    model_config = ConfigDict(populate_by_name=True)

class EntityAppearanceOut(BaseModel):
    story_id: str
    title: Localized

class EntityRelationOut(BaseModel):
    other_key: str
    other_label: Localized
    label: Localized
    outgoing: bool

class EntityDetailOut(BaseModel):

    key: str
    type: str
    label: Localized
    category: Localized
    description: Localized
    appearances: list[EntityAppearanceOut]
    relations: list[EntityRelationOut]

class StoryGraphOut(BaseModel):
    story_id: str
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]

class OptionOut(BaseModel):

    id: str
    label: Localized

class QuestionOut(BaseModel):
    id: str
    type: str
    answer_mode: str = "choice"
    position: int
    prompt: Localized
    options: list[OptionOut]
    evidence_paragraph: int | None = None

class QuizOut(BaseModel):
    story_id: str
    questions: list[QuestionOut]

class AttemptOut(BaseModel):
    id: str
    story_id: str
    score: int
    total: int
    finished: bool = False
    answered_question_ids: list[str] = []

class AnswerIn(BaseModel):
    question_id: str
    option_id: str | None = None
    text: str | None = None
    lang: str = "en"
    was_followup: bool = False

class AnswerOut(BaseModel):
    is_correct: bool
    correct_option_id: str = ""
    feedback: str = ""
    reference_answer: str = ""
    explanation: Localized
    kg_edge: str | None = None
    source_text: str | None = None
    follow_up: QuestionOut | None = None
    score: int
    answered: int
    total: int

class ProgressIn(BaseModel):
    paragraph: int

class ProgressOut(BaseModel):
    story_id: str
    paragraph: int
    completed: bool
    quiz: QuizResultOut | None = None

class LearnerSummaryOut(BaseModel):
    user_id: str
    stories_started: int
    stories_completed: int
    questions_answered: int
    progress: list[ProgressOut]

class SearchIn(BaseModel):

    q: str | None = None
    embedding: list[float] | None = None
    model: str | None = None
    story_id: str | None = None
    lang: str | None = None
    limit: int = Field(default=5, ge=1, le=50)
    rerank: bool | None = None

class SearchResultOut(BaseModel):
    chunk_id: str
    story_id: str | None
    document_id: str | None
    paragraph: int | None
    lang: str
    text: str
    distance: float
    score: float
    rerank_score: float | None = None

class SearchOut(BaseModel):
    reranked: bool = False
    results: list[SearchResultOut]

class DocumentOut(BaseModel):
    id: str
    filename: str
    lang: str
    status: str
    source_id: str | None
    story_id: str | None
    chunk_count: int

class ReviewParagraphOut(BaseModel):

    paragraph_id: str
    position: int
    lang: str
    body: str
    status: str
    approved_by: str | None = None

class StoryReviewOut(BaseModel):
    story_id: str
    title: str
    published: bool
    approved: int
    total: int
    chunk_count: int = 0
    embedded_count: int = 0
    quiz_size: int = 6
    quiz_relationship: int = 0
    quiz_identification: int = 0
    quiz_comprehension: int = 0
    paragraphs: list[ReviewParagraphOut]

class PublishRequest(BaseModel):
    published: bool = True

class GenerateQuestionsRequest(BaseModel):
    lang: str = "en"
    count: int = Field(default=6, ge=1, le=20)
    age_range: str = "7-10"
    answer_mode: str = "choice"
    replace: bool = False
    cover_all: bool = False

class QuestionOptionOut(BaseModel):
    id: str
    label: str
    is_correct: bool

class QuestionReviewOut(BaseModel):

    id: str
    type: str
    answer_mode: str = "choice"
    position: int
    status: str
    lang: str
    prompt: str
    explanation: str
    reference_answer: str = ""
    options: list[QuestionOptionOut]
    evidence_paragraph: int | None = None
    source_text: str | None = None
    follow_up: "QuestionReviewOut | None" = None

QuestionReviewOut.model_rebuild()

class QuestionSetOut(BaseModel):
    story_id: str
    approved: int
    total: int
    covered: int = 0
    paragraphs: int = 0
    questions: list[QuestionReviewOut]
    warnings: list[str] = []

class QuizShape(BaseModel):

    quiz_size: int = Field(default=6, ge=1, le=50)
    quiz_relationship: int = Field(default=0, ge=0, le=50)
    quiz_identification: int = Field(default=0, ge=0, le=50)
    quiz_comprehension: int = Field(default=0, ge=0, le=50)
