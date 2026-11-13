from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.educational_ontology import get_relation
from app.ml.kg_records import KGPreparedRecords
from app.models import (
    Entity,
    EntityMention,
    EntityTranslation,
    Predicate,
    PredicateTranslation,
    Relation,
    StoryEntity,
    StoryRelation,
    StoryRelationEvidence,
)


def _entity_by_key(
    session: Session,
    key: str,
) -> Entity | None:
    return session.scalars(
        select(Entity).where(Entity.key == key)
    ).one_or_none()


def _relation_by_triple(
    session: Session,
    *,
    subject_id: str,
    predicate_id: str,
    object_id: str,
) -> Relation | None:
    return session.scalars(
        select(Relation).where(
            Relation.subject_id == subject_id,
            Relation.predicate_id == predicate_id,
            Relation.object_id == object_id,
        )
    ).one_or_none()


def _fallback_en_label(key: str) -> str:
    return key.replace("_", " ").strip().title()


def _entity_category(
    record_type: str,
    lang: str,
) -> str:
    if record_type == "group":
        return (
            "Person ? Group"
            if lang == "en"
            else "شخصية ? مجموعة"
        )

    if record_type == "place":
        return (
            "Place"
            if lang == "en"
            else "مكان"
        )

    return (
        "Person"
        if lang == "en"
        else "شخصية"
    )


def _ensure_predicate(
    session: Session,
    predicate_id: str,
) -> Predicate:
    definition = get_relation(predicate_id)

    predicate = session.get(
        Predicate,
        predicate_id,
    )

    if predicate is None:
        predicate = Predicate(
            id=predicate_id,
            symmetric=definition.symmetric,
        )
        session.add(predicate)
        session.flush()

    elif predicate.symmetric != definition.symmetric:
        predicate.symmetric = definition.symmetric

    values = {
        "en": (
            definition.label_en,
            definition.phrase_en
            or f"{definition.label_en} {{object}}",
        ),
        "ar": (
            definition.label_ar,
            definition.phrase_ar
            or f"{definition.label_ar} {{object}}",
        ),
    }

    for lang, (label, phrase) in values.items():
        translation = session.get(
            PredicateTranslation,
            {
                "predicate_id": predicate_id,
                "lang": lang,
            },
        )

        if translation is None:
            session.add(
                PredicateTranslation(
                    predicate_id=predicate_id,
                    lang=lang,
                    label=label,
                    phrase=phrase,
                )
            )

        else:
            if (
                not translation.label.strip()
                or "?" in translation.label
            ):
                translation.label = label

            if (
                not translation.phrase.strip()
                or "?" in translation.phrase
            ):
                translation.phrase = phrase

    return predicate


def persist_prepared_records(
    session: Session,
    *,
    records: KGPreparedRecords,
    run_id: str | None = None,
) -> None:
    """
    Persist validated KG records into the existing application schema.

    Important:
    - Does not commit.
    - Uses existing Entity rows when keys already exist.
    - Uses existing Relation rows when triples already exist.
    - Requires predicates to already exist in the predicates table.
    - Adds story appearances and paragraph mentions idempotently.
    """

    entity_by_key: dict[str, Entity] = {}

    #
    # 1. Entities + Arabic labels
    #
    for record in records.entities:
        #
        # The research linker distinguishes collective human entities as
        # "group". The existing application/UI entity contract stores both
        # individual people and human groups under "person", with the
        # translation category carrying the group distinction.
        #
        # Keep the research representation unchanged and normalize only at
        # this persistence boundary.
        #
        storage_type = "person" if record.type == "group" else record.type

        entity = _entity_by_key(
            session,
            record.key,
        )

        if entity is None:
            entity = Entity(
                key=record.key,
                type=storage_type,
                status="approved",
                run_id=run_id,
            )
            session.add(entity)
            session.flush()

        elif entity.type != storage_type:
            raise ValueError(
                f"Entity type conflict for {record.key!r}: "
                f"DB={entity.type!r}, extracted={record.type!r}, "
                f"storage={storage_type!r}"
            )

        entity_by_key[record.key] = entity

        categories = {
            "ar": _entity_category(
                record.type,
                "ar",
            ),
            "en": _entity_category(
                record.type,
                "en",
            ),
        }

        existing_ar = session.get(
            EntityTranslation,
            {
                "entity_id": entity.id,
                "lang": "ar",
            },
        )

        if existing_ar is None:
            session.add(
                EntityTranslation(
                    entity_id=entity.id,
                    lang="ar",
                    label=record.label_ar,
                    category=categories["ar"],
                    description="",
                )
            )

        elif not existing_ar.category.strip():
            existing_ar.category = categories["ar"]

        existing_en = session.get(
            EntityTranslation,
            {
                "entity_id": entity.id,
                "lang": "en",
            },
        )

        if existing_en is None:
            session.add(
                EntityTranslation(
                    entity_id=entity.id,
                    lang="en",
                    label=_fallback_en_label(
                        record.key
                    ),
                    category=categories["en"],
                    description="",
                )
            )

        elif not existing_en.category.strip():
            existing_en.category = categories["en"]

    session.flush()

    #
    # 2. StoryEntity appearances
    #
    for record in records.story_entities:
        entity = entity_by_key[record.entity_key]

        existing = session.get(
            StoryEntity,
            {
                "story_id": record.story_id,
                "entity_id": entity.id,
            },
        )

        if existing is None:
            session.add(
                StoryEntity(
                    story_id=record.story_id,
                    entity_id=entity.id,
                )
            )

    #
    # 3. EntityMention rows
    #
    for record in records.mentions:
        entity = entity_by_key[record.entity_key]

        existing = session.scalars(
            select(EntityMention).where(
                EntityMention.entity_id == entity.id,
                EntityMention.paragraph_id == record.paragraph_id,
                EntityMention.lang == record.lang,
                EntityMention.char_start == record.char_start,
                EntityMention.char_end == record.char_end,
            )
        ).one_or_none()

        if existing is None:
            session.add(
                EntityMention(
                    entity_id=entity.id,
                    paragraph_id=record.paragraph_id,
                    lang=record.lang,
                    char_start=record.char_start,
                    char_end=record.char_end,
                    score=None,
                    run_id=run_id,
                )
            )

    session.flush()

    #
    # 4. Relations
    #
    relation_by_key: dict[
        tuple[str, str, str],
        Relation,
    ] = {}

    for record in records.relations:
        subject = entity_by_key[record.subject_key]
        object_ = entity_by_key[record.object_key]

        predicate = _ensure_predicate(
            session,
            record.predicate,
        )

        relation = _relation_by_triple(
            session,
            subject_id=subject.id,
            predicate_id=record.predicate,
            object_id=object_.id,
        )

        if relation is None:
            relation = Relation(
                subject_id=subject.id,
                predicate_id=record.predicate,
                object_id=object_.id,
                confidence=record.confidence,
                status="approved",
                run_id=run_id,
            )
            session.add(relation)
            session.flush()

        elif (
            record.confidence is not None
            and (
                relation.confidence is None
                or record.confidence > relation.confidence
            )
        ):
            relation.confidence = record.confidence

        relation_by_key[
            (
                record.subject_key,
                record.predicate,
                record.object_key,
            )
        ] = relation

    #
    # 5. StoryRelation appearances + evidence paragraph
    #
    for record in records.story_relations:
        relation = relation_by_key[
            (
                record.subject_key,
                record.predicate,
                record.object_key,
            )
        ]

        existing = session.get(
            StoryRelation,
            {
                "story_id": record.story_id,
                "relation_id": relation.id,
            },
        )

        if existing is None:
            session.add(
                StoryRelation(
                    story_id=record.story_id,
                    relation_id=relation.id,
                    evidence_paragraph_id=record.evidence_paragraph_id,
                    evidence_source=record.evidence_source,
                    source_relation=record.source_relation,
                    resolution_rule=record.resolution_rule,
                )
            )

        else:
            # Preserve existing evidence when present, but fill any
            # missing provenance fields idempotently.
            if (
                existing.evidence_paragraph_id is None
                and record.evidence_paragraph_id is not None
            ):
                existing.evidence_paragraph_id = (
                    record.evidence_paragraph_id
                )

            if (
                existing.evidence_source is None
                and record.evidence_source is not None
            ):
                existing.evidence_source = record.evidence_source

            if (
                existing.source_relation is None
                and record.source_relation is not None
            ):
                existing.source_relation = record.source_relation

            if (
                existing.resolution_rule is None
                and record.resolution_rule is not None
            ):
                existing.resolution_rule = record.resolution_rule

        # The KG extraction pipeline is currently Arabic-only.
        evidence_lang = "ar"

        evidence = session.get(
            StoryRelationEvidence,
            {
                "story_id": record.story_id,
                "relation_id": relation.id,
                "paragraph_id": record.evidence_paragraph_id,
                "lang": evidence_lang,
            },
        )

        if evidence is None:
            session.add(
                StoryRelationEvidence(
                    story_id=record.story_id,
                    relation_id=relation.id,
                    paragraph_id=record.evidence_paragraph_id,
                    lang=evidence_lang,
                    evidence_source=record.evidence_source,
                    source_relation=record.source_relation,
                    resolution_rule=record.resolution_rule,
                )
            )
        else:
            # Reprocessing the same paragraph should refresh its
            # provenance rather than create another evidence row.
            evidence.evidence_source = record.evidence_source
            evidence.source_relation = record.source_relation
            evidence.resolution_rule = record.resolution_rule

    session.flush()


def retract_paragraph_kg(
    session: Session,
    *,
    paragraph_id: str,
    lang: str,
) -> None:
    """
    Remove KG evidence derived from one paragraph/language.

    This does NOT delete global Entity or Relation rows.

    If another paragraph still supports the same StoryRelation, the
    StoryRelation is retained and its compatibility evidence fields are
    repointed to one of the remaining evidence rows.
    """

    #
    # 1. Remove paragraph-specific entity mentions.
    #
    mentions = list(
        session.scalars(
            select(EntityMention).where(
                EntityMention.paragraph_id == paragraph_id,
                EntityMention.lang == lang,
            )
        )
    )

    for mention in mentions:
        session.delete(mention)

    #
    # 2. Find and remove relation evidence belonging to this paragraph.
    #
    evidence_rows = list(
        session.scalars(
            select(StoryRelationEvidence).where(
                StoryRelationEvidence.paragraph_id == paragraph_id,
                StoryRelationEvidence.lang == lang,
            )
        )
    )

    affected = {
        (row.story_id, row.relation_id)
        for row in evidence_rows
    }

    for row in evidence_rows:
        session.delete(row)

    session.flush()

    #
    # 3. Reconcile the compatibility StoryRelation row.
    #
    for story_id, relation_id in affected:
        appearance = session.get(
            StoryRelation,
            {
                "story_id": story_id,
                "relation_id": relation_id,
            },
        )

        if appearance is None:
            continue

        remaining = session.scalars(
            select(StoryRelationEvidence)
            .where(
                StoryRelationEvidence.story_id == story_id,
                StoryRelationEvidence.relation_id == relation_id,
            )
            .order_by(
                StoryRelationEvidence.paragraph_id,
                StoryRelationEvidence.lang,
            )
        ).first()

        if remaining is None:
            # No approved paragraph evidence remains for this
            # story/relation appearance.
            session.delete(appearance)
        else:
            # Keep the existing application contract intact:
            # StoryRelation still exposes one representative evidence
            # paragraph even though the new table retains all evidence.
            appearance.evidence_paragraph_id = remaining.paragraph_id
            appearance.evidence_source = remaining.evidence_source
            appearance.source_relation = remaining.source_relation
            appearance.resolution_rule = remaining.resolution_rule

    session.flush()


__all__ = [
    "persist_prepared_records",
    "retract_paragraph_kg",
]
