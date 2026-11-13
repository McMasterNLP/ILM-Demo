from dataclasses import asdict, dataclass

from app.ml.family_relation_resolver import (
    FamilyRelationResolver,
    ResolvedFamilyRelation,
)
from app.ml.re_candidates import (
    RelationCandidate,
    RelationCandidateService,
)
from app.ml.re_positive_candidates import (
    PositiveRelationCandidateService,
    PositiveRelationPrediction,
)
from app.ml.prophetic_relation_resolver import (
    PropheticRelationResolver,
    ResolvedPropheticRelation,
)
from app.ml.major_event_relation_resolver import (
    MajorEventRelationResolver,
    ResolvedMajorEventRelation,
)
from app.ml.educational_ontology import get_relation
from app.ml.islamic_entity_linker import LinkedEntityMention


@dataclass(frozen=True)
class EducationalRelation:
    """
    Canonical relation representation consumed by the educational KG.

    This deliberately hides whether the relation came from:
      - source WOJOOD RE + semantic resolution, or
      - an Islamic educational pattern outside the WOJOOD ontology.

    Provenance is still preserved explicitly.
    """

    sentence: str

    subject_key: str
    subject_text: str
    subject_type: str

    predicate: str

    object_key: str
    object_text: str
    object_type: str

    resolution_rule: str
    evidence_source: str

    source_relation: str | None = None
    existence_probability: float | None = None
    relation_confidence: float | None = None


@dataclass(frozen=True)
class StoryRelationPipelineResult:
    """
    Complete output for one story paragraph.

    stage_a_candidates:
        Every directed linked-entity pair evaluated by the V6
        relation-existence model.

    stage_b_predictions:
        Source WOJOOD relation predictions only for pairs that passed
        Stage A.

    educational_family_relations:
        Family relations whose semantics/direction could be resolved
        safely from verified Arabic constructions.

    unresolved_source_relations:
        Stage-B predictions that have not yet been converted into the
        educational KG ontology.
    """

    linked_mentions: tuple[LinkedEntityMention, ...]
    stage_a_candidates: tuple[RelationCandidate, ...]
    stage_b_predictions: tuple[PositiveRelationPrediction, ...]
    educational_family_relations: tuple[ResolvedFamilyRelation, ...]
    educational_prophetic_relations: tuple[ResolvedPropheticRelation, ...]
    educational_major_event_relations: tuple[ResolvedMajorEventRelation, ...]
    educational_relations: tuple[EducationalRelation, ...]
    unresolved_source_relations: tuple[PositiveRelationPrediction, ...]


class StoryRelationPipeline:
    """
    Current Islamic-story relation pipeline.

    Flow:
        NER
        -> Islamic entity linking
        -> directed entity pairs
        -> V6 Stage-A existence gate
        -> 40-way source RE + NLI
        -> verified family semantic resolver

    This pipeline does NOT:
        - modify the research NER system
        - modify the source RE system
        - write to the database
        - invent educational mappings for unsupported relations
        - silently discard unresolved Stage-B predictions
    """

    def __init__(self) -> None:
        self.stage_a = RelationCandidateService()
        self.stage_b = PositiveRelationCandidateService()
        self.family_resolver = FamilyRelationResolver()
        self.prophetic_resolver = PropheticRelationResolver()
        self.major_event_resolver = MajorEventRelationResolver()

    def predict(
        self,
        text: str,
        *,
        story_key: str,
    ) -> StoryRelationPipelineResult:

        linked_mentions, stage_a_candidates = (
            self.stage_a.predict_with_mentions(
                text,
                story_key=story_key,
            )
        )

        passed = [
            candidate
            for candidate in stage_a_candidates
            if candidate.passes_existence_gate
        ]

        stage_b_predictions = self.stage_b.predict(passed)

        family_relations = self.family_resolver.resolve(
            stage_b_predictions
        )

        prophetic_relations = self.prophetic_resolver.resolve(
            sentence=text,
            candidates=stage_a_candidates,
        )

        major_event_relations = self.major_event_resolver.resolve(
            sentence=text,
            mentions=linked_mentions,
            story_key=story_key,
        )

        educational_relations = []

        for relation in family_relations:
            educational_relations.append(
                EducationalRelation(
                    sentence=relation.sentence,

                    subject_key=relation.subject_key,
                    subject_text=relation.subject_text,
                    subject_type=relation.subject_type,

                    predicate=relation.predicate,

                    object_key=relation.object_key,
                    object_text=relation.object_text,
                    object_type=relation.object_type,

                    resolution_rule=relation.resolution_rule,
                    evidence_source="source_re_resolved",

                    source_relation=relation.source_relation,
                    existence_probability=(
                        relation.existence_probability
                    ),
                    relation_confidence=(
                        relation.relation_confidence
                    ),
                )
            )

        for relation in prophetic_relations:
            educational_relations.append(
                EducationalRelation(
                    sentence=relation.sentence,

                    subject_key=relation.subject_key,
                    subject_text=relation.subject_text,
                    subject_type=relation.subject_type,

                    predicate=relation.predicate,

                    object_key=relation.object_key,
                    object_text=relation.object_text,
                    object_type=relation.object_type,

                    resolution_rule=relation.resolution_rule,
                    evidence_source=relation.source,

                    source_relation=None,
                    existence_probability=None,
                    relation_confidence=None,
                )
            )

        for relation in major_event_relations:
            educational_relations.append(
                EducationalRelation(
                    sentence=relation.sentence,

                    subject_key=relation.subject_key,
                    subject_text=relation.subject_text,
                    subject_type=relation.subject_type,

                    predicate=relation.predicate,

                    object_key=relation.object_key,
                    object_text=relation.object_text,
                    object_type=relation.object_type,

                    resolution_rule=relation.resolution_rule,
                    evidence_source=relation.source,

                    source_relation=None,
                    existence_probability=None,
                    relation_confidence=None,
                )
            )

        #
        # Validate every KG-facing predicate against the controlled
        # educational ontology before exposing it downstream.
        #
        for relation in educational_relations:
            get_relation(relation.predicate)

        educational_relations.sort(
            key=lambda item: (
                item.subject_key,
                item.predicate,
                item.object_key,
            )
        )

        #
        # Track the exact source entity pairs represented by safely
        # resolved family edges.
        #
        # Family relations such as has_parent may be predicted in both
        # directions by the source RE system. A resolved semantic edge
        # therefore consumes both source directions for that same pair,
        # but must not hide unrelated predictions that happen to share
        # the same source relation label.
        #
        resolved_family_pairs = {
            frozenset(
                (
                    relation.subject_key,
                    relation.object_key,
                )
            )
            for relation in family_relations
        }

        unresolved = [
            prediction
            for prediction in stage_b_predictions
            if not (
                prediction.relation_label
                in self.family_resolver.FAMILY_RELATIONS
                and frozenset(
                    (
                        prediction.subject_key,
                        prediction.object_key,
                    )
                )
                in resolved_family_pairs
            )
        ]

        return StoryRelationPipelineResult(
            linked_mentions=tuple(linked_mentions),
            stage_a_candidates=tuple(stage_a_candidates),
            stage_b_predictions=tuple(stage_b_predictions),
            educational_family_relations=tuple(family_relations),
            educational_prophetic_relations=tuple(prophetic_relations),
            educational_major_event_relations=tuple(major_event_relations),
            educational_relations=tuple(educational_relations),
            unresolved_source_relations=tuple(unresolved),
        )

    def predict_dict(
        self,
        text: str,
        *,
        story_key: str,
    ) -> dict:
        result = self.predict(
            text,
            story_key=story_key,
        )

        return {
            "linked_mentions": [
                asdict(item)
                for item in result.linked_mentions
            ],
            "stage_a_candidates": [
                asdict(item)
                for item in result.stage_a_candidates
            ],
            "stage_b_predictions": [
                asdict(item)
                for item in result.stage_b_predictions
            ],
            "educational_family_relations": [
                asdict(item)
                for item in result.educational_family_relations
            ],
            "educational_prophetic_relations": [
                asdict(item)
                for item in result.educational_prophetic_relations
            ],
            "educational_major_event_relations": [
                asdict(item)
                for item in result.educational_major_event_relations
            ],
            "educational_relations": [
                asdict(item)
                for item in result.educational_relations
            ],
            "unresolved_source_relations": [
                asdict(item)
                for item in result.unresolved_source_relations
            ],
        }


__all__ = [
    "EducationalRelation",
    "StoryRelationPipelineResult",
    "StoryRelationPipeline",
]
