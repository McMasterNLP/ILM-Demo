from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Iterable

from app.ml.educational_ontology import get_relation

if TYPE_CHECKING:
    from app.ml.islamic_entity_linker import LinkedEntityMention
    from app.ml.story_relation_pipeline import EducationalRelation


@dataclass(frozen=True)
class KGEntityRecord:
    key: str
    type: str
    label_ar: str


@dataclass(frozen=True)
class KGEntityMentionRecord:
    entity_key: str
    paragraph_id: str
    lang: str
    char_start: int
    char_end: int
    text: str
    official_ner: bool


@dataclass(frozen=True)
class KGStoryEntityRecord:
    story_id: str
    entity_key: str


@dataclass(frozen=True)
class KGRelationRecord:
    subject_key: str
    predicate: str
    object_key: str

    confidence: float | None

    evidence_source: str
    source_relation: str | None
    resolution_rule: str


@dataclass(frozen=True)
class KGStoryRelationRecord:
    story_id: str
    subject_key: str
    predicate: str
    object_key: str
    evidence_paragraph_id: str
    evidence_source: str
    source_relation: str | None
    resolution_rule: str


@dataclass(frozen=True)
class KGPreparedRecords:
    entities: tuple[KGEntityRecord, ...]
    mentions: tuple[KGEntityMentionRecord, ...]
    story_entities: tuple[KGStoryEntityRecord, ...]
    relations: tuple[KGRelationRecord, ...]
    story_relations: tuple[KGStoryRelationRecord, ...]


class KGRecordBuilder:
    """
    Convert extraction/linking output into DB-ready logical records.

    Important:
    - No SQLAlchemy session.
    - No database writes.
    - No IDs generated yet.
    - No mutation of existing Entity/Relation rows.
    - Educational predicates must exist in the controlled ontology.
    """

    @staticmethod
    def _build_entities(
        mentions: Iterable[LinkedEntityMention],
    ) -> list[KGEntityRecord]:

        best: dict[str, KGEntityRecord] = {}

        for mention in mentions:
            record = KGEntityRecord(
                key=mention.canonical_key,
                type=mention.app_type,
                label_ar=mention.normalized_text,
            )

            previous = best.get(record.key)

            if previous is None:
                best[record.key] = record
                continue

            if previous.type != record.type:
                raise ValueError(
                    "Conflicting app entity types for "
                    f"{record.key}: "
                    f"{previous.type!r} vs {record.type!r}"
                )

        return sorted(
            best.values(),
            key=lambda item: item.key,
        )

    @staticmethod
    def _add_contextual_relation_entities(
        entities: Iterable[KGEntityRecord],
        relations: Iterable[EducationalRelation],
    ) -> list[KGEntityRecord]:
        """
        Add canonical entities introduced by an explicitly approved
        contextual resolver rule without fabricating text mentions.

        At present this is intentionally restricted to the Yusuf-story
        implicit subject rule:

            ?????? ?? ???
            -> yusuf --thrown_into--> well

        The paragraph contains a pronominal reference to Yusuf, so Yusuf
        is a valid story/relation entity but does not receive a synthetic
        EntityMention span.
        """

        records = {
            entity.key: entity
            for entity in entities
        }

        for relation in relations:
            if (
                relation.resolution_rule
                != "yusuf_story_implicit_throw_subject"
            ):
                continue

            if (
                relation.subject_key != "yusuf"
                or relation.subject_type != "person"
            ):
                raise ValueError(
                    "Unexpected endpoint for "
                    "yusuf_story_implicit_throw_subject: "
                    f"{relation.subject_key!r}, "
                    f"{relation.subject_type!r}"
                )

            record = KGEntityRecord(
                key=relation.subject_key,
                type=relation.subject_type,
                label_ar=relation.subject_text,
            )

            previous = records.get(record.key)

            if previous is not None:
                if previous.type != record.type:
                    raise ValueError(
                        "Conflicting app entity types for "
                        f"{record.key}: "
                        f"{previous.type!r} vs {record.type!r}"
                    )
                continue

            records[record.key] = record

        return sorted(
            records.values(),
            key=lambda item: item.key,
        )

    @staticmethod
    def _build_mentions(
        mentions: Iterable[LinkedEntityMention],
        *,
        paragraph_id: str,
    ) -> list[KGEntityMentionRecord]:

        records = {}

        for mention in mentions:
            record = KGEntityMentionRecord(
                entity_key=mention.canonical_key,
                paragraph_id=paragraph_id,
                lang="ar",
                char_start=mention.char_start,
                char_end=mention.char_end,
                text=mention.text,
                official_ner=mention.official_ner,
            )

            key = (
                record.entity_key,
                record.paragraph_id,
                record.char_start,
                record.char_end,
            )

            records[key] = record

        return sorted(
            records.values(),
            key=lambda item: (
                item.char_start,
                item.char_end,
                item.entity_key,
            ),
        )

    @staticmethod
    def _build_story_entities(
        entities: Iterable[KGEntityRecord],
        *,
        story_id: str,
    ) -> list[KGStoryEntityRecord]:

        return [
            KGStoryEntityRecord(
                story_id=story_id,
                entity_key=entity.key,
            )
            for entity in entities
        ]

    @staticmethod
    def _build_relations(
        relations: Iterable[EducationalRelation],
    ) -> list[KGRelationRecord]:

        records = {}

        for relation in relations:
            # Fail closed if a predicate escaped ontology validation.
            get_relation(relation.predicate)

            record = KGRelationRecord(
                subject_key=relation.subject_key,
                predicate=relation.predicate,
                object_key=relation.object_key,
                confidence=relation.relation_confidence,
                evidence_source=relation.evidence_source,
                source_relation=relation.source_relation,
                resolution_rule=relation.resolution_rule,
            )

            key = (
                record.subject_key,
                record.predicate,
                record.object_key,
            )

            previous = records.get(key)

            if previous is None:
                records[key] = record
                continue

            #
            # If the same semantic edge is produced more than once,
            # retain the record with the stronger available confidence.
            # Pattern-only educational edges may have confidence=None.
            #
            previous_score = (
                previous.confidence
                if previous.confidence is not None
                else -1.0
            )

            current_score = (
                record.confidence
                if record.confidence is not None
                else -1.0
            )

            if current_score > previous_score:
                records[key] = record

        return sorted(
            records.values(),
            key=lambda item: (
                item.subject_key,
                item.predicate,
                item.object_key,
            ),
        )

    @staticmethod
    def _build_story_relations(
        relations: Iterable[KGRelationRecord],
        *,
        story_id: str,
        paragraph_id: str,
    ) -> list[KGStoryRelationRecord]:

        return [
            KGStoryRelationRecord(
                story_id=story_id,
                subject_key=relation.subject_key,
                predicate=relation.predicate,
                object_key=relation.object_key,
                evidence_paragraph_id=paragraph_id,
                evidence_source=relation.evidence_source,
                source_relation=relation.source_relation,
                resolution_rule=relation.resolution_rule,
            )
            for relation in relations
        ]

    def build(
        self,
        *,
        story_id: str,
        paragraph_id: str,
        mentions: Iterable[LinkedEntityMention],
        relations: Iterable[EducationalRelation],
    ) -> KGPreparedRecords:

        mentions = list(mentions)
        relations = list(relations)

        entities = self._build_entities(mentions)

        entities = self._add_contextual_relation_entities(
            entities,
            relations,
        )

        mention_records = self._build_mentions(
            mentions,
            paragraph_id=paragraph_id,
        )

        story_entities = self._build_story_entities(
            entities,
            story_id=story_id,
        )

        relation_records = self._build_relations(relations)

        entity_keys = {
            entity.key
            for entity in entities
        }

        for relation in relation_records:
            if relation.subject_key not in entity_keys:
                raise ValueError(
                    "Relation subject is missing from linked entities: "
                    f"{relation.subject_key}"
                )

            if relation.object_key not in entity_keys:
                raise ValueError(
                    "Relation object is missing from linked entities: "
                    f"{relation.object_key}"
                )

        story_relations = self._build_story_relations(
            relation_records,
            story_id=story_id,
            paragraph_id=paragraph_id,
        )

        return KGPreparedRecords(
            entities=tuple(entities),
            mentions=tuple(mention_records),
            story_entities=tuple(story_entities),
            relations=tuple(relation_records),
            story_relations=tuple(story_relations),
        )

    def build_dict(
        self,
        *,
        story_id: str,
        paragraph_id: str,
        mentions: Iterable[LinkedEntityMention],
        relations: Iterable[EducationalRelation],
    ) -> dict:

        result = self.build(
            story_id=story_id,
            paragraph_id=paragraph_id,
            mentions=mentions,
            relations=relations,
        )

        return {
            "entities": [
                asdict(item)
                for item in result.entities
            ],
            "mentions": [
                asdict(item)
                for item in result.mentions
            ],
            "story_entities": [
                asdict(item)
                for item in result.story_entities
            ],
            "relations": [
                asdict(item)
                for item in result.relations
            ],
            "story_relations": [
                asdict(item)
                for item in result.story_relations
            ],
        }


__all__ = [
    "KGEntityRecord",
    "KGEntityMentionRecord",
    "KGStoryEntityRecord",
    "KGRelationRecord",
    "KGStoryRelationRecord",
    "KGPreparedRecords",
    "KGRecordBuilder",
]
