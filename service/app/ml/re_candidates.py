from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

from app.ml.entity_pairs import (
    EntityPair,
    EntityPairGenerator,
)
from app.ml.re_inference import (
    EXISTENCE_THRESHOLD,
    RelationExistenceEnsemble,
)
from app.ml.islamic_entity_linker import LinkedEntityMention


@dataclass(frozen=True)
class RelationCandidate:
    sentence: str

    subject_key: str
    subject_text: str
    subject_type: str
    subject_char_start: int
    subject_char_end: int

    object_key: str
    object_text: str
    object_type: str
    object_char_start: int
    object_char_end: int

    existence_probability: float
    passes_existence_gate: bool


class RelationCandidateService:
    """
    Canonical entity linking + directed pair generation + V6 Stage-A gate.

    This stage intentionally does NOT:
    - predict the 40-way relation label
    - run NLI reranking
    - apply educational ontology mapping
    - write anything to the database

    A pair passes when:
        P(relation) >= EXISTENCE_THRESHOLD

    EXISTENCE_THRESHOLD remains the production value 0.5.
    """

    def __init__(
        self,
        *,
        pair_generator: Optional[EntityPairGenerator] = None,
        existence_model: Optional[RelationExistenceEnsemble] = None,
    ) -> None:
        self.pair_generator = (
            pair_generator or EntityPairGenerator()
        )

        self.existence_model = (
            existence_model or RelationExistenceEnsemble()
        )

    def predict_from_pairs(
        self,
        pairs: List[EntityPair],
    ) -> List[RelationCandidate]:
        """
        Apply the unchanged V6 Stage-A existence gate to already-built
        directed entity pairs.
        """
        results: List[RelationCandidate] = []

        for pair in pairs:
            probability = (
                self.existence_model.predict_probability(
                    sentence=pair.sentence,
                    subject=pair.subject_text,
                    object_=pair.object_text,
                )
            )

            results.append(
                RelationCandidate(
                    sentence=pair.sentence,

                    subject_key=pair.subject_key,
                    subject_text=pair.subject_text,
                    subject_type=pair.subject_type,
                    subject_char_start=pair.subject_char_start,
                    subject_char_end=pair.subject_char_end,

                    object_key=pair.object_key,
                    object_text=pair.object_text,
                    object_type=pair.object_type,
                    object_char_start=pair.object_char_start,
                    object_char_end=pair.object_char_end,

                    existence_probability=float(probability),
                    passes_existence_gate=(
                        probability >= EXISTENCE_THRESHOLD
                    ),
                )
            )

        return results

    def predict_with_mentions(
        self,
        text: str,
        *,
        story_key: str,
    ) -> tuple[List[LinkedEntityMention], List[RelationCandidate]]:
        """
        Single-pass entity-linking + Stage-A inference.

        The linked mentions returned here are the exact mentions used
        to construct the Stage-A entity pairs.
        """
        mentions = self.pair_generator.link_mentions(
            text,
            story_key=story_key,
        )

        pairs = self.pair_generator.generate_from_mentions(
            text,
            mentions=mentions,
        )

        candidates = self.predict_from_pairs(pairs)

        return mentions, candidates

    def predict(
        self,
        text: str,
        *,
        story_key: str,
    ) -> List[RelationCandidate]:

        pairs = self.pair_generator.generate(
            text,
            story_key=story_key,
        )

        return self.predict_from_pairs(pairs)

    def predict_positive(
        self,
        text: str,
        *,
        story_key: str,
    ) -> List[RelationCandidate]:
        return [
            candidate
            for candidate in self.predict(
                text,
                story_key=story_key,
            )
            if candidate.passes_existence_gate
        ]

    def predict_dicts(
        self,
        text: str,
        *,
        story_key: str,
    ) -> List[dict]:
        return [
            asdict(candidate)
            for candidate in self.predict(
                text,
                story_key=story_key,
            )
        ]


__all__ = [
    "RelationCandidate",
    "RelationCandidateService",
]
