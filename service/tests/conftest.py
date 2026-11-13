from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db import get_session
from app.main import app
from app.models import (
    Base,
    Entity,
    EntityMention,
    EntityTranslation,
    Option,
    OptionTranslation,
    Paragraph,
    ParagraphTranslation,
    Predicate,
    PredicateTranslation,
    Question,
    QuestionTranslation,
    Relation,
    Story,
    StoryEntity,
    StoryRelation,
    StoryTranslation,
)

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+psycopg://ilm:ilm@localhost:5432/ilm_test"
)

@pytest.fixture(scope="session")
def engine():
    engine = create_engine(TEST_DATABASE_URL, future=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    except Exception as exc:
        pytest.skip(f"No test database at {TEST_DATABASE_URL}: {exc}")

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()

@pytest.fixture
def session(engine):
    connection = engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(bind=connection, autoflush=False, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()

@pytest.fixture
def client(session):

    def override():
        yield session

    app.dependency_overrides[get_session] = override
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

def learner(client: TestClient, learner_id: str = "tester") -> dict[str, str]:
    return {"X-Learner-Id": learner_id}

@pytest.fixture
def yusuf(session):
    story = Story(
        id="yusuf",
        level="intermediate",
        minutes=12,
        character_count=2,
        position=0,
        published=True,
    )
    session.add(story)
    session.add(
        StoryTranslation(
            story_id="yusuf",
            lang="en",
            title="Prophet Yusuf (AS)",
            summary="A journey.",
            heading="The Story of Prophet Yusuf (AS)",
            status="approved",
        )
    )

    paragraphs = []
    for position in range(3):
        paragraph = Paragraph(
            id=f"p{position}", story_id="yusuf", position=position, is_pending=False
        )
        session.add(paragraph)
        session.add(
            ParagraphTranslation(
                paragraph_id=paragraph.id,
                lang="en",
                body=f"Paragraph {position}.",
                status="approved",
            )
        )
        paragraphs.append(paragraph)

    session.add(Predicate(id="son_of"))
    session.add(
        PredicateTranslation(
            predicate_id="son_of", lang="en", label="son of", phrase="Son of {object}"
        )
    )

    for key, label in (("yusuf", "Yusuf (AS)"), ("yaqub", "Yaqub (AS)")):
        session.add(Entity(id=key, key=key, type="person", status="approved"))
        session.add(EntityTranslation(entity_id=key, lang="en", label=label))
        session.add(StoryEntity(story_id="yusuf", entity_id=key))

    session.add(EntityMention(id="m0", entity_id="yusuf", paragraph_id="p0", lang="en"))

    session.add(
        Relation(
            id="r1",
            subject_id="yusuf",
            predicate_id="son_of",
            object_id="yaqub",
            status="approved",
        )
    )
    session.add(StoryRelation(story_id="yusuf", relation_id="r1"))

    question = Question(
        id="q1",
        story_id="yusuf",
        type="relationship",
        position=0,
        relation_id="r1",
        status="approved",
    )
    session.add(question)
    session.add(
        QuestionTranslation(
            question_id="q1",
            lang="en",
            prompt="Who is the father of Yusuf?",
            explanation="Yusuf is the son of Yaqub.",
        )
    )
    for position, (option_id, label, correct) in enumerate(
        [("o1", "Yaqub", True), ("o2", "Al-Aziz", False)]
    ):
        session.add(
            Option(id=option_id, question_id="q1", position=position, is_correct=correct)
        )
        session.add(OptionTranslation(option_id=option_id, lang="en", label=label))

    followup = Question(
        id="q1f",
        story_id="yusuf",
        type="relationship",
        position=0,
        relation_id="r1",
        followup_of="q1",
        status="approved",
    )
    session.add(followup)
    session.add(
        QuestionTranslation(
            question_id="q1f", lang="en", prompt="Yusuf is the son of which prophet?"
        )
    )
    session.add(Option(id="o3", question_id="q1f", position=0, is_correct=True))
    session.add(OptionTranslation(option_id="o3", lang="en", label="Yaqub"))

    session.flush()
    return story
