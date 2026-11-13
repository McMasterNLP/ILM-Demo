from sqlalchemy import select

from app.db import SessionLocal
from app.models import (
    Entity,
    EntityMention,
    ParagraphTranslation,
    Predicate,
    Relation,
    StoryRelation,
)


PARAGRAPH_ID = "yusuf-p2"
LANG = "ar"
STORY_ID = "yusuf"


with SessionLocal() as session:
    row = session.get(
        ParagraphTranslation,
        {
            "paragraph_id": PARAGRAPH_ID,
            "lang": LANG,
        },
    )

    print("STATUS:", row.status if row else None)
    print("TEXT:", row.body if row else None)

    mentions = session.scalars(
        select(EntityMention).where(
            EntityMention.paragraph_id == PARAGRAPH_ID,
            EntityMention.lang == LANG,
        )
    ).all()

    print("\nMENTIONS:")
    for mention in mentions:
        entity = session.get(Entity, mention.entity_id)

        text = None
        if row is not None and mention.char_start is not None and mention.char_end is not None:
            text = row.body[mention.char_start:mention.char_end]

        print(
            mention.entity_id,
            "| type=",
            entity.type if entity else None,
            "| text=",
            repr(text),
            "| offsets=",
            (mention.char_start, mention.char_end),
            "| score=",
            mention.score,
        )

    story_relations = session.scalars(
        select(StoryRelation).where(
            StoryRelation.story_id == STORY_ID,
            StoryRelation.evidence_paragraph_id == PARAGRAPH_ID,
        )
    ).all()

    print("\nSTORY RELATIONS:")
    for story_relation in story_relations:
        relation = session.get(Relation, story_relation.relation_id)

        if relation is None:
            print("Missing relation:", story_relation.relation_id)
            continue

        predicate = session.get(Predicate, relation.predicate_id)

        print(
            relation.subject_id,
            "--",
            predicate.id if predicate else relation.predicate_id,
            "-->",
            relation.object_id,
            "| confidence=",
            relation.confidence,
            "| source=",
            story_relation.evidence_source,
            "| source_relation=",
            story_relation.source_relation,
            "| rule=",
            story_relation.resolution_rule,
        )
