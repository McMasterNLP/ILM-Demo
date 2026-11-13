from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    Entity,
    EntityMention,
    Paragraph,
    Question,
    Relation,
    Story,
    StoryEntity,
    StoryRelation,
)
from app.schemas import (
    GraphEdgeOut,
    GraphNodeOut,
    ParagraphOut,
    StoryGraphOut,
)

APPROVED = "approved"

def localized(rows, field: str, *, only_approved: bool = False) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        if only_approved and getattr(row, "status", APPROVED) != APPROVED:
            continue
        out[row.lang] = getattr(row, field)
    return out

PENDING_NOTICE = {
    "en": "This paragraph is waiting for text from a curated source.",
    "ar": "هذه الفقرة بانتظار نص من مصدر معتمد.",
}

def _pending_notice() -> dict[str, str]:
    return dict(PENDING_NOTICE)

def list_stories(session: Session) -> list[Story]:
    stmt = select(Story).where(Story.published.is_(True)).order_by(Story.position, Story.id)
    return list(session.scalars(stmt))

def get_story(session: Session, story_id: str) -> Story | None:
    stmt = select(Story).where(Story.id == story_id, Story.published.is_(True))
    return session.scalars(stmt).one_or_none()

def paragraph_counts(session: Session) -> dict[str, int]:
    stmt = select(Paragraph.story_id, func.count(Paragraph.id)).group_by(Paragraph.story_id)
    return {story_id: count for story_id, count in session.execute(stmt)}

def question_counts(session: Session) -> dict[str, int]:
    stmt = (
        select(Question.story_id, func.count(Question.id))
        .where(Question.followup_of.is_(None), Question.status == APPROVED)
        .group_by(Question.story_id)
    )
    return {story_id: count for story_id, count in session.execute(stmt)}

def story_paragraphs(session: Session, story: Story) -> list[ParagraphOut]:
    mention_stmt = (
        select(EntityMention.paragraph_id, Entity.key)
        .join(Entity, Entity.id == EntityMention.entity_id)
        .join(Paragraph, Paragraph.id == EntityMention.paragraph_id)
        .where(Paragraph.story_id == story.id)
        .distinct()
    )
    characters: dict[str, list[str]] = {}
    for paragraph_id, key in session.execute(mention_stmt):
        characters.setdefault(paragraph_id, []).append(key)

    out: list[ParagraphOut] = []
    for paragraph in story.paragraphs:
        text = localized(paragraph.translations, "body", only_approved=True)
        out.append(
            ParagraphOut(
                position=paragraph.position,
                pending=paragraph.is_pending or not text,
                text=text or _pending_notice(),
                characters=characters.get(paragraph.id, []),
            )
        )
    return out

def story_heading(story: Story) -> dict[str, str]:
    titles = localized(story.translations, "title", only_approved=True)
    headings = localized(story.translations, "heading", only_approved=True)
    return {lang: headings.get(lang) or titles.get(lang, "") for lang in titles}

def _render_phrase(templates: dict[str, str], object_labels: dict[str, str]) -> dict[str, str]:
    rendered: dict[str, str] = {}
    for lang, template in templates.items():
        rendered[lang] = template.replace("{object}", object_labels.get(lang, ""))
    return rendered

def story_graph(
    session: Session,
    story_id: str,
) -> StoryGraphOut:

    automatic_stmt = (
        select(Relation)
        .join(
            StoryRelation,
            StoryRelation.relation_id == Relation.id,
        )
        .where(
            StoryRelation.story_id == story_id,
            (
                StoryRelation.evidence_paragraph_id.is_not(None)
                | StoryRelation.evidence_source.startswith("quran:")
            ),
            Relation.status == APPROVED,
        )
    )

    relations = list(
        session.scalars(automatic_stmt)
    )

    automatic = bool(relations)

    #
    # Backward compatibility for stories that have not yet been
    # migrated to automatic paragraph-derived KG extraction.
    #
    if not relations:
        legacy_stmt = (
            select(Relation)
            .join(
                StoryRelation,
                StoryRelation.relation_id == Relation.id,
            )
            .where(
                StoryRelation.story_id == story_id,
                Relation.status == APPROVED,
            )
        )

        relations = list(
            session.scalars(legacy_stmt)
        )

    node_ids = {
        entity_id
        for relation in relations
        for entity_id in (
            relation.subject_id,
            relation.object_id,
        )
    }

    #
    # For automatically generated graphs, include entities actually
    # mentioned in the current story even when they do not participate
    # in one of the educational relations.
    #
    if automatic:
        mention_ids = session.scalars(
            select(EntityMention.entity_id)
            .join(
                Paragraph,
                Paragraph.id == EntityMention.paragraph_id,
            )
            .where(
                Paragraph.story_id == story_id
            )
            .distinct()
        )

        node_ids.update(mention_ids)

    entity_stmt = (
        select(
            Entity,
            StoryEntity,
        )
        .join(
            StoryEntity,
            StoryEntity.entity_id == Entity.id,
        )
        .where(
            StoryEntity.story_id == story_id,
            Entity.status == APPROVED,
        )
        .order_by(
            Entity.key
        )
    )

    if automatic:
        entity_stmt = entity_stmt.where(
            Entity.id.in_(node_ids)
        )

    nodes = [
        GraphNodeOut(
            id=entity.key,
            type=entity.type,
            label=localized(
                entity.translations,
                "label",
            ),
            category=localized(
                entity.translations,
                "category",
            ),
            description=localized(
                entity.translations,
                "description",
            ),
            x=appearance.layout_x or 0.0,
            y=appearance.layout_y or 0.0,
        )
        for entity, appearance
        in session.execute(entity_stmt)
    ]

    relation_layers: dict[str, str] = {}

    if relations:
        story_relation_rows = session.scalars(
            select(StoryRelation).where(
                StoryRelation.story_id == story_id,
                StoryRelation.relation_id.in_([relation.id for relation in relations]),
            )
        )

        for story_relation in story_relation_rows:
            source = story_relation.evidence_source or ""

            if source.startswith("quran:"):
                layer = "quran"
            elif story_relation.evidence_paragraph_id is not None:
                layer = "automatic"
            else:
                layer = "legacy"

            relation_layers[story_relation.relation_id] = layer

    edges = [
        GraphEdgeOut(
            from_=relation.subject.key,
            to=relation.object.key,
            label=localized(
                relation.predicate.translations,
                "label",
            ),
            phrase=_render_phrase(
                localized(
                    relation.predicate.translations,
                    "phrase",
                ),
                localized(
                    relation.object.translations,
                    "label",
                ),
            ),
            layer=relation_layers.get(relation.id, "legacy"),
        )
        for relation in relations
    ]

    return StoryGraphOut(
        story_id=story_id,
        nodes=nodes,
        edges=edges,
    )

def entity_by_key(session: Session, key: str) -> Entity | None:
    return session.scalars(select(Entity).where(Entity.key == key)).one_or_none()

def stories_featuring(session: Session, entity: Entity) -> list[Story]:
    stmt = (
        select(Story)
        .join(StoryEntity, StoryEntity.story_id == Story.id)
        .where(StoryEntity.entity_id == entity.id, Story.published.is_(True))
        .order_by(Story.position, Story.id)
    )
    return list(session.scalars(stmt))

def relations_for_entity(session: Session, entity: Entity) -> list[Relation]:
    stmt = select(Relation).where(
        (Relation.subject_id == entity.id) | (Relation.object_id == entity.id),
        Relation.status == APPROVED,
    )
    return list(session.scalars(stmt))
